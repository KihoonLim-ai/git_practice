import torch
import torch.nn as nn
import numpy as np

# [1] 푸리에 임베딩 레이어 추가
class GaussianFourierFeatureTransform(nn.Module):
    # [수정] scale 인자 받도록 변경
    def __init__(self, input_dim, mapping_size=256, scale=10.0):
        super().__init__()
        self._input_dim = input_dim
        self._mapping_size = mapping_size
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        x_proj = (2.0 * np.pi * x) @ self.B
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
    # [수정] fourier_scale 인자 추가
    def __init__(self, input_dim=4, latent_dim=128, dropout=0.1, fourier_scale=10.0): 
        super().__init__()
        
        self.fourier_dim = 256 
        self.fourier_mapping = GaussianFourierFeatureTransform(
            input_dim, 
            mapping_size=self.fourier_dim // 2, 
            scale=fourier_scale # [핵심] 외부에서 받은 scale 적용
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
            nn.Tanh()
        )

    def forward(self, x):
        x_embed = self.fourier_mapping(x) 
        return self.net(x_embed)

class ST_TransformerDeepONet(nn.Module):
    # [수정] fourier_scale 인자 추가 및 전달
    def __init__(self, latent_dim=128, dropout=0.1, num_heads=4, fourier_scale=10.0):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.map_encoder = ConvBranch(3, latent_dim)
        self.obs_encoder = TransformerObsBranch(3, latent_dim, num_heads, dropout=dropout)
        
        # [수정] Trunk에 scale 전달
        self.trunk = SpatioTemporalTrunk(4, latent_dim, dropout, fourier_scale=fourier_scale)
        
        self.fusion = nn.Linear(latent_dim * 2, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.head_wind = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(),
            nn.Linear(64, 3)
        )
        self.head_conc = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, ctx_map, obs_seq, query_coords):
        z_map = self.map_encoder(ctx_map)
        z_obs = self.obs_encoder(obs_seq)
        
        z_ctx = self.fusion(torch.cat([z_map, z_obs], dim=1))
        z_ctx = self.dropout(z_ctx)
        
        z_trunk = self.trunk(query_coords)
        merged = z_ctx.unsqueeze(1) * z_trunk
        
        return self.head_wind(merged), self.head_conc(merged)