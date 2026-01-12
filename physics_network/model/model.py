import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. Sub-Modules (Encoders & Trunk)
# ==============================================================================

class Conv3dBranch(nn.Module):
    """
    지형 및 3D 물리 정보 인코더 (Terrain/Physics Encoder)
    Input: (B, 5, D, H, W) -> [Terrain, Source, U, V, W]
    Output: (B, Latent)
    """
    def __init__(self, in_channels=5, latent_dim=128):
        super().__init__()
        # in_channels=5 (Dataset과 일치시킴)
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16), # Batch가 작을 땐 BN보다 GroupNorm 추천
            nn.GELU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)) # Global Context로 압축 B x 64 x 1 x 1 x 1
        )
        self.fc = nn.Linear(64, latent_dim) # B 128 1 1 1

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten: (B, 64)
        return self.fc(x)

class TransformerObsBranch(nn.Module):
    """
    기상 시계열 정보 인코더 (Met Sequence Encoder)
    Input: (B, Seq, 3) -> [u_surf, v_surf, inv_L]
    Output: (B, Latent)
    """
    def __init__(self, input_dim=3, latent_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, latent_dim)
        # batch_first=True 필수
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dropout=dropout, 
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        # x: (B, Seq_len, 3)
        x = self.embedding(x)
        x = self.transformer(x)
        # 마지막 시점(t)의 정보만 사용하여 Context 벡터 생성
        return self.fc(x[:, -1, :])

class SpatioTemporalTrunk(nn.Module):
    """
    좌표(x,y,z,t) 인코더 (Trunk Network)
    Fourier Features를 사용하여 고주파 성분(급격한 농도 변화 등) 학습 능력 강화
    """
    def __init__(self, input_dim=4, latent_dim=128, hidden_dim=256, num_layers=4, dropout=0.1, fourier_scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.fourier_scale = fourier_scale
        
        # Fourier Mapping: 좌표를 고차원 주파수 영역으로 매핑
        self.fourier_mapping = nn.Linear(input_dim, latent_dim // 2) 
        
        # MLP Layers
        layers = []
        # Input dimension은 sin, cos 두 개씩 나오므로 latent_dim 크기가 됨
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            # Trunk에는 Dropout을 신중히 사용 (좌표 연속성을 위해 제외하거나 낮게 설정)
            # layers.append(nn.Dropout(dropout)) 
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        # coords: (B, N, 4)
        # Fourier Feature Embedding: [sin(2*pi*B*x), cos(2*pi*B*x)]
        proj = self.fourier_mapping(coords) * self.fourier_scale
        x = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1) # (B, N, Latent)
        return self.net(x)

# ==============================================================================
# 2. Main Model (ST-TransformerDeepONet)
# ==============================================================================

class ST_TransformerDeepONet(nn.Module):
    def __init__(self, latent_dim=128, dropout=0.1, num_heads=4, fourier_scale=10.0):
        super().__init__()
        
        # 1. Encoders (Branch Nets)
        # Dataset 채널 수 5개 반영
        self.map_encoder = Conv3dBranch(in_channels=5, latent_dim=latent_dim)
        self.obs_encoder = TransformerObsBranch(input_dim=3, latent_dim=latent_dim, num_heads=num_heads, dropout=dropout)
        
        # 2. Trunk Net
        self.trunk = SpatioTemporalTrunk(input_dim=4, latent_dim=latent_dim, dropout=dropout, fourier_scale=fourier_scale)
        
        # 3. Fusion (Context Generator)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 4. Heads
        # Wind Delta Head: (Context * Trunk) -> (du, dv, w)
        self.head_wind = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3) 
        )
        
        # Conc Head: (Context * Trunk + PredWind) -> Concentration
        self.head_conc = nn.Sequential(
            nn.Linear(latent_dim + 3, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, ctx_map, obs_seq, query_coords, global_wind):
        """
        [Inputs]
        ctx_map: (B, 5, D, H, W)
        obs_seq: (B, Seq, 3)
        query_coords: (B, N, 4) -> [x, y, z, t] (Normalized 0~1)
        global_wind: (B, 2) -> [u_top, v_top]
        """
        
        # -----------------------------------------------------------
        # 1. Encoding & Fusion (Branch)
        # -----------------------------------------------------------
        z_map = self.map_encoder(ctx_map) # (B, Latent)
        z_obs = self.obs_encoder(obs_seq) # (B, Latent)
        
        # Context Vector 생성
        z_ctx = self.fusion(torch.cat([z_map, z_obs], dim=1)) # (B, Latent)
        
        # -----------------------------------------------------------
        # 2. Trunk encoding
        # -----------------------------------------------------------
        # 좌표 임베딩
        z_trunk = self.trunk(query_coords) # (B, N, Latent)
        
        # -----------------------------------------------------------
        # 3. DeepONet Merging (FiLM-like Modulation)
        # -----------------------------------------------------------
        # Context(전역 정보)가 Trunk(지역 좌표 정보)를 모듈레이션
        # (B, 1, Latent) * (B, N, Latent) -> (B, N, Latent)
        merged = z_ctx.unsqueeze(1) * z_trunk 
        
        # -----------------------------------------------------------
        # 4. Wind Prediction (Physics-Informed)
        # -----------------------------------------------------------
        # (A) Global Base Wind 확장
        # global_wind: (B, 2) -> (B, N, 3) [u, v, 0]
        B, N, _ = query_coords.shape
        base_uv = global_wind.unsqueeze(1).expand(-1, N, -1) # (B, N, 2)
        base_w  = torch.zeros((B, N, 1), device=base_uv.device)
        base_wind = torch.cat([base_uv, base_w], dim=-1) # (B, N, 3)

        # (B) Residual (Delta) 예측
        pred_delta = self.head_wind(merged) # (B, N, 3)

        # (C) Combine & Physics Constraints (Log Profile / No-slip)
        # z 좌표 (0~1 범위라고 가정)
        z_vals = query_coords[..., 2:3] # (B, N, 1)
        
        # Power Law Profile 적용: z가 0일 때 0이 되도록 강제
        # (z + epsilon)^0.3
        height_factor = torch.pow(z_vals + 1e-6, 0.3)
        
        # Final Wind = (Base + Delta) * HeightFactor
        pred_wind = (base_wind + pred_delta) * height_factor
        
        # -----------------------------------------------------------
        # 5. Concentration Prediction
        # -----------------------------------------------------------
        # "좌표 특징 + 예측된 바람"을 이용하여 농도 예측
        conc_input = torch.cat([merged, pred_wind], dim=-1) # (B, N, Latent + 3)
        pred_conc = self.head_conc(conc_input) # (B, N, 1)
        
        # 농도는 음수가 될 수 없으므로 Softplus 또는 ReLU 적용 추천
        pred_conc = F.softplus(pred_conc) 
        
        return pred_wind, pred_conc