import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Simplified Model: No Wind/Met Data
# ==============================================================================

class Conv3dBranchSimple(nn.Module):
    """
    간소화된 3D 인코더: Terrain + Source만 사용
    Input: (B, 2, D, H, W) -> [Terrain, Source]
    Output: (B, Latent)
    """
    def __init__(self, in_channels=2, latent_dim=128):
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
        # x: (B, 2, D, H, W)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


class SpatioTemporalTrunk(nn.Module):
    """
    좌표(x,y,z,t) 인코더 (원본과 동일)
    Fourier Features를 사용하여 고주파 성분 학습
    """
    def __init__(self, input_dim=4, latent_dim=128, hidden_dim=256,
                 num_layers=4, fourier_scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.fourier_scale = fourier_scale

        # Fourier Mapping
        self.fourier_mapping = nn.Linear(input_dim, latent_dim // 2)

        # MLP Layers
        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.GELU())

        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        # coords: (B, N, 4)
        proj = self.fourier_mapping(coords) * self.fourier_scale
        x = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(x)


class SimplifiedDeepONet(nn.Module):
    """
    바람/기상 데이터 없이 농도만 예측하는 간소화 모델

    Architecture:
        Static Maps (Terrain + Source) → Conv3d → Context Vector
        Coordinates (x,y,z,t) → Fourier → Trunk Network
        Context * Trunk → MLP → Concentration

    입력:
        - ctx_map: (B, 2, D, H, W) 정적 지도 [Terrain, Source]
        - query_coords: (B, N, 4) 쿼리 좌표 [x, y, z, t]

    출력:
        - pred_conc: (B, N, 1) 예측 농도
    """
    def __init__(self, latent_dim=128, fourier_scale=10.0, dropout=0.1):
        super().__init__()

        # 1. Map Encoder (Static spatial features)
        self.map_encoder = Conv3dBranchSimple(
            in_channels=2,
            latent_dim=latent_dim
        )

        # 2. Trunk Network (Coordinate encoder)
        self.trunk = SpatioTemporalTrunk(
            input_dim=4,
            latent_dim=latent_dim,
            fourier_scale=fourier_scale
        )

        # 3. Concentration Prediction Head
        self.head_conc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, ctx_map, query_coords):
        """
        Forward pass without wind/met data

        Args:
            ctx_map: (B, 2, D, H, W) - Static maps [Terrain, Source]
            query_coords: (B, N, 4) - Query coordinates [x, y, z, t]

        Returns:
            pred_conc: (B, N, 1) - Predicted concentration
        """
        # -----------------------------------------------------------
        # 1. Encode Static Maps
        # -----------------------------------------------------------
        z_map = self.map_encoder(ctx_map)  # (B, latent_dim)

        # -----------------------------------------------------------
        # 2. Encode Coordinates
        # -----------------------------------------------------------
        z_trunk = self.trunk(query_coords)  # (B, N, latent_dim)

        # -----------------------------------------------------------
        # 3. DeepONet Fusion
        # -----------------------------------------------------------
        # Context modulates trunk features (FiLM-like)
        merged = z_map.unsqueeze(1) * z_trunk  # (B, N, latent_dim)

        # -----------------------------------------------------------
        # 4. Predict Concentration
        # -----------------------------------------------------------
        pred_conc = self.head_conc(merged)  # (B, N, 1)

        # Enforce non-negativity
        pred_conc = F.softplus(pred_conc)

        return pred_conc
