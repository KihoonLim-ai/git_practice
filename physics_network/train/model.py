import torch
import torch.nn as nn

# [1] 푸리에 임베딩 레이어 추가
class GaussianFourierFeatureTransform(nn.Module):
    def __init__(self, input_dim, mapping_size=256, scale=10.0):
        super().__init__()
        self._input_dim = input_dim
        self._mapping_size = mapping_size
        # 학습되지 않는 고정된 랜덤 가중치 (B)
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        # x: (Batch, input_dim)
        # Projection: (2 * pi * x) @ B
        x_proj = (2.0 * np.pi * x) @ self.B
        # sin, cos을 붙여서 차원을 2배로 뻥튀기
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
class ConvBranch(nn.Module):
    """ [Map Encoder] latent_dim 인자 추가 """
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)

class TransformerObsBranch(nn.Module):
    """ [Obs Encoder] Hyperparams 인자 추가 """
    def __init__(self, input_dim=3, latent_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, latent_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, 
            dim_feedforward=latent_dim*4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        seq_len = x.shape[1]
        x = self.input_proj(x) + self.pos_emb[:, :seq_len, :]
        x = self.transformer(x)
        return self.fc_out(x.mean(dim=1))

class SpatioTemporalTrunk(nn.Module):
    def __init__(self, input_dim=4, latent_dim=128, dropout=0.1): 
        super().__init__()
        
        # 푸리에 피처 설정
        # mapping_size가 128이면, sin/cos 합쳐서 출력은 256차원이 됨
        self.fourier_dim = 256 
        self.fourier_mapping = GaussianFourierFeatureTransform(
            input_dim, 
            mapping_size=self.fourier_dim // 2, 
            scale=10.0 # scale이 클수록 고주파(세밀한 변화)를 잘 잡음
        )
        
        # 첫 번째 레이어의 입력 차원이 '4'가 아니라 '256'으로 변경됨
        self.net = nn.Sequential(
            nn.Linear(self.fourier_dim, 128), # <--- 여기가 핵심 변경점
            nn.Tanh(), 
            nn.Dropout(dropout),
            
            nn.Linear(128, 128), 
            nn.Tanh(), 
            nn.Dropout(dropout),
            
            nn.Linear(128, 128), 
            nn.Tanh(), 
            nn.Dropout(dropout),
            
            nn.Linear(128, latent_dim), 
            nn.Tanh()
        )

    def forward(self, x):
        # 1. 좌표(x,y,z,t)를 먼저 푸리에 공간으로 매핑
        x_embed = self.fourier_mapping(x) 
        # 2. 매핑된 고차원 벡터를 MLP에 통과
        return self.net(x_embed)

class ST_TransformerDeepONet(nn.Module):
    """
    [Final Model] 
    수정사항: 
    1. __init__에 latent_dim, dropout 인자 추가 (Sweep용)
    2. head_conc에서 Softplus 제거 (Z-score 음수 예측 허용)
    """
    def __init__(self, latent_dim=128, dropout=0.1, num_heads=4):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.map_encoder = ConvBranch(3, latent_dim)
        self.obs_encoder = TransformerObsBranch(3, latent_dim, num_heads, dropout=dropout)
        self.trunk = SpatioTemporalTrunk(4, latent_dim, dropout)
        
        self.fusion = nn.Linear(latent_dim * 2, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.head_wind = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(),
            nn.Linear(64, 3)
        )
        
        # [핵심 수정] Softplus 제거! (Linear Output)
        self.head_conc = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(),
            nn.Linear(64, 1) # Identity Activation
        )

    def forward(self, ctx_map, obs_seq, query_coords):
        z_map = self.map_encoder(ctx_map)
        z_obs = self.obs_encoder(obs_seq)
        
        z_ctx = self.fusion(torch.cat([z_map, z_obs], dim=1))
        z_ctx = self.dropout(z_ctx)
        
        z_trunk = self.trunk(query_coords)
        merged = z_ctx.unsqueeze(1) * z_trunk
        
        return self.head_wind(merged), self.head_conc(merged)