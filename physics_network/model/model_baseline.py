"""
Baseline Model: Current ST_TransformerDeepONet without PDE loss
No physics-informed modifications
"""
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
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

class SpatioTemporalTrunk(nn.Module):
    """
    좌표(x,y,z,t) 인코더 (Trunk Network)
    Fourier Features를 사용하여 고주파 성분 학습
    """
    def __init__(self, input_dim=4, latent_dim=128, hidden_dim=256, num_layers=4, dropout=0.1, fourier_scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.fourier_scale = fourier_scale

        self.fourier_mapping = nn.Linear(input_dim, latent_dim // 2)

        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.GELU())

        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        proj = self.fourier_mapping(coords) * self.fourier_scale
        x = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(x)

# ==============================================================================
# 2. Main Model (Baseline)
# ==============================================================================

class ST_TransformerDeepONet_Baseline(nn.Module):
    """
    Baseline: No PDE loss
    """
    def __init__(self, latent_dim=128, dropout=0.1, num_heads=4, fourier_scale=10.0, in_channels=5):
        super().__init__()

        self.map_encoder = Conv3dBranch(in_channels=in_channels, latent_dim=latent_dim)
        self.obs_encoder = TransformerObsBranch(input_dim=3, latent_dim=latent_dim, num_heads=num_heads, dropout=dropout)
        self.trunk = SpatioTemporalTrunk(input_dim=4, latent_dim=latent_dim, dropout=dropout, fourier_scale=fourier_scale)

        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.head_wind = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

        self.head_conc = nn.Sequential(
            nn.Linear(latent_dim + 3, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, ctx_map, obs_seq, query_coords, global_wind):
        """
        Standard forward pass without PDE computation
        """
        z_map = self.map_encoder(ctx_map)
        z_obs = self.obs_encoder(obs_seq)
        z_ctx = self.fusion(torch.cat([z_map, z_obs], dim=1))

        z_trunk = self.trunk(query_coords)
        merged = z_ctx.unsqueeze(1) * z_trunk

        # Wind prediction
        B, N, _ = query_coords.shape
        base_uv = global_wind.unsqueeze(1).expand(-1, N, -1)
        base_w = torch.zeros((B, N, 1), device=base_uv.device)
        base_wind = torch.cat([base_uv, base_w], dim=-1)

        pred_delta = self.head_wind(merged)
        z_vals = query_coords[..., 2:3]
        height_factor = torch.pow(z_vals + 1e-6, 0.3)
        pred_wind = (base_wind + pred_delta) * height_factor

        # Concentration prediction
        conc_input = torch.cat([merged, pred_wind], dim=-1)
        pred_conc = self.head_conc(conc_input)
        pred_conc = F.softplus(pred_conc)

        return pred_wind, pred_conc
