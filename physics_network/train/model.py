import torch
import torch.nn as nn
import numpy as np

# --- [1] Fourier Feature Embedding ---
class GaussianFourierFeatureTransform(nn.Module):
    """
    좌표(x,y,z,t)를 고차원 주파수 영역으로 매핑하여
    High-Frequency 성분(급격한 농도 변화, 피크)을 학습하도록 도움
    """
    def __init__(self, input_dim, mapping_size=256, scale=10.0):
        super().__init__()
        self._input_dim = input_dim
        self._mapping_size = mapping_size
        # scale이 클수록 더 높은 주파수(세밀한 변화)를 잡음
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        # x: (Batch, N, input_dim)
        # 2pi * x * B
        x_proj = (2.0 * np.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# --- [2] Map Encoder (CNN) ---
class ConvBranch(nn.Module):
    """
    [수정됨] in_channels=4
    1. Terrain (지형)
    2. Source Q (배출량 - Gaussian Splatted)
    3. Source H (높이)
    4. Advected Source (바람에 밀린 오염원 힌트) - NEW
    """
    def __init__(self, in_channels=4, latent_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),      nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1),     nn.BatchNorm2d(128), nn.GELU(),
            # 필요 시 레이어 추가 가능
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)

# --- [3] Observation Encoder (Transformer) ---
class TransformerObsBranch(nn.Module):
    """
    과거 기상 데이터(u, v, L)의 시계열 패턴 학습
    """
    def __init__(self, input_dim=3, latent_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        # 최대 100시간까지 커버 가능한 Positional Embedding
        self.pos_emb = nn.Parameter(torch.randn(1, 100, latent_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, 
            dim_feedforward=latent_dim*4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len, 3)
        seq_len = x.shape[1]
        x = self.input_proj(x) + self.pos_emb[:, :seq_len, :]
        x = self.transformer(x)
        # 시퀀스 전체의 평균을 Latent Vector로 사용
        return self.fc_out(x.mean(dim=1))

# --- [4] Spatio-Temporal Trunk (Coordinates) ---
class SpatioTemporalTrunk(nn.Module):
    """
    예측하고 싶은 위치(x,y,z,t)를 임베딩
    """
    def __init__(self, input_dim=4, latent_dim=128, dropout=0.1, fourier_scale=10.0): 
        super().__init__()
        
        self.fourier_dim = 256 
        self.fourier_mapping = GaussianFourierFeatureTransform(
            input_dim, 
            mapping_size=self.fourier_dim // 2, 
            scale=fourier_scale 
        )
        
        self.net = nn.Sequential(
            nn.Linear(self.fourier_dim, 128),
            nn.Tanh(), 
            nn.Dropout(dropout),
            nn.Linear(128, 128), 
            nn.Tanh(), 
            nn.Dropout(dropout),
            nn.Linear(128, 128), 
            nn.Tanh(), 
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim), 
            nn.Tanh() # DeepONet의 Basis Function 역할
        )

    def forward(self, x):
        x_embed = self.fourier_mapping(x) 
        return self.net(x_embed)

# --- [5] Main Model (DeepONet with Cascade) ---
class ST_TransformerDeepONet(nn.Module):
    def __init__(self, latent_dim=128, dropout=0.1, num_heads=4, fourier_scale=10.0):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 1. Encoders
        # [수정] Map Encoder 입력 채널 4로 설정
        self.map_encoder = ConvBranch(in_channels=4, latent_dim=latent_dim)
        self.obs_encoder = TransformerObsBranch(input_dim=3, latent_dim=latent_dim, num_heads=num_heads, dropout=dropout)
        
        # 2. Trunk
        self.trunk = SpatioTemporalTrunk(input_dim=4, latent_dim=latent_dim, dropout=dropout, fourier_scale=fourier_scale)
        
        # 3. Fusion (Context Fusion)
        self.fusion = nn.Linear(latent_dim * 2, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 4. Heads (Cascade Structure)
        # (A) Wind Head: Latent -> (u, v, w)
        self.head_wind = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(),
            nn.Linear(64, 3) 
        )
        
        # (B) Conc Head: [Latent + Wind(3)] -> Conc(1)
        # [핵심 수정] 입력 차원이 latent_dim + 3으로 증가
        self.head_conc = nn.Sequential(
            nn.Linear(latent_dim + 3, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, ctx_map, obs_seq, query_coords):
        """
        ctx_map: (Batch, 4, 45, 45)
        obs_seq: (Batch, Seq, 3)
        query_coords: (Batch, N_points, 4)
        """
        # 1. Encode Context
        z_map = self.map_encoder(ctx_map)       # (B, Latent)
        z_obs = self.obs_encoder(obs_seq)       # (B, Latent)
        
        z_ctx = self.fusion(torch.cat([z_map, z_obs], dim=1)) # (B, Latent)
        z_ctx = self.dropout(z_ctx)
        
        # 2. Encode Query Coordinates (Trunk)
        z_trunk = self.trunk(query_coords)      # (B, N, Latent)
        
        # 3. DeepONet Interaction (Dot-product like)
        # (B, 1, Latent) * (B, N, Latent) -> (B, N, Latent)
        merged = z_ctx.unsqueeze(1) * z_trunk 
        
        # 4. Cascade Prediction
        # (A) 바람장 먼저 예측
        pred_wind = self.head_wind(merged) # (B, N, 3)
        
        # (B) 바람 정보를 농도 예측의 입력으로 활용 (Concatenation)
        # "이 위치의 잠재 특징" + "예측된 바람 벡터"
        conc_input = torch.cat([merged, pred_wind], dim=-1) # (B, N, Latent + 3)
        
        # (C) 농도 예측
        pred_conc = self.head_conc(conc_input) # (B, N, 1)
        
        return pred_wind, pred_conc