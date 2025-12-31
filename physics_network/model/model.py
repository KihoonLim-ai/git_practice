import torch
import torch.nn as nn
import numpy as np

# --- [1] Fourier Feature Embedding (동일) ---
class GaussianFourierFeatureTransform(nn.Module):
    def __init__(self, input_dim, mapping_size=256, scale=10.0):
        super().__init__()
        self._input_dim = input_dim
        self._mapping_size = mapping_size
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        x_proj = (2.0 * np.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# --- [2] Map Encoder (3D CNN) ---
class Conv3dBranch(nn.Module):
    """
    [수정됨] 3D Volumetric Input을 처리하기 위한 3D CNN
    Input Shape: (Batch, 4, 21, 45, 45) -> (B, C, Depth, Height, Width)
    Channels: Terrain, Source, U_init, V_init
    """
    def __init__(self, in_channels=4, latent_dim=128):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Layer 1: (B, 4, 21, 45, 45) -> (B, 32, 21, 23, 23)
            # Z축은 유지(stride=1), XY축만 절반(stride=2)으로 줄임 (Z 해상도 보존)
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            
            # Layer 2: (B, 32, 21, 23, 23) -> (B, 64, 10, 12, 12)
            # 이제 Z축도 줄이기 시작 (stride=2)
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            
            # Layer 3: (B, 64, 10, 12, 12) -> (B, 128, 5, 6, 6)
            nn.Conv3d(64, 128, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(128),
            nn.GELU(),
            
            # Layer 4: (B, 128, 5, 6, 6) -> (B, 256, 3, 3, 3)
            # 깊이를 더 쌓아서 고차원 특징 추출
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.GELU()
        )
        
        # Global Average Pooling (3D)
        # (B, 256, 3, 3, 3) -> (B, 256, 1, 1, 1)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Latent Projection
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):
        # x: (Batch, 4, 21, 45, 45)
        x = self.conv_layers(x)
        x = self.gap(x).flatten(1) # (B, 256)
        return self.fc(x)          # (B, Latent)

# --- [3] Observation Encoder (Transformer) (동일) ---
class TransformerObsBranch(nn.Module):
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

# --- [4] Spatio-Temporal Trunk (Coordinates) (동일) ---
class SpatioTemporalTrunk(nn.Module):
    def __init__(self, input_dim=4, latent_dim=128, dropout=0.1, fourier_scale=10.0): 
        super().__init__()
        self.fourier_dim = 256 
        self.fourier_mapping = GaussianFourierFeatureTransform(input_dim, self.fourier_dim // 2, fourier_scale)
        self.net = nn.Sequential(
            nn.Linear(self.fourier_dim, 128), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(128, latent_dim), nn.Tanh()
        )
    def forward(self, x):
        x_embed = self.fourier_mapping(x) 
        return self.net(x_embed)

# --- [5] Main Model (DeepONet with 3D CNN) ---
class ST_TransformerDeepONet(nn.Module):
    def __init__(self, latent_dim=128, dropout=0.1, num_heads=4, fourier_scale=10.0):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 1. Encoders
        # [수정] 3D CNN Branch 사용
        self.map_encoder = Conv3dBranch(in_channels=4, latent_dim=latent_dim)
        self.obs_encoder = TransformerObsBranch(input_dim=3, latent_dim=latent_dim, num_heads=num_heads, dropout=dropout)
        
        # 2. Trunk
        self.trunk = SpatioTemporalTrunk(input_dim=4, latent_dim=latent_dim, dropout=dropout, fourier_scale=fourier_scale)
        
        # 3. Fusion
        self.fusion = nn.Linear(latent_dim * 2, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 4. Heads (Cascade)
        self.head_wind = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 3))
        self.head_conc = nn.Sequential(nn.Linear(latent_dim + 3, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, ctx_map, obs_seq, query_coords):
        # ctx_map: (Batch, 4, 21, 45, 45) -> 3D Volumetric Input
        z_map = self.map_encoder(ctx_map)
        z_obs = self.obs_encoder(obs_seq)
        
        z_ctx = self.fusion(torch.cat([z_map, z_obs], dim=1))
        z_ctx = self.dropout(z_ctx)
        
        z_trunk = self.trunk(query_coords) # (B, N, Latent)
        
        merged = z_ctx.unsqueeze(1) * z_trunk 
        
        pred_wind = self.head_wind(merged) # (u, v, w)
        
        conc_input = torch.cat([merged, pred_wind], dim=-1)
        pred_conc = self.head_conc(conc_input) # (c)
        
        return pred_wind, pred_conc