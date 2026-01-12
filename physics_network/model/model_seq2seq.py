"""
Sequence-to-Sequence Model for Concentration Prediction
과거 농도 시계열 → 미래 농도 예측 (바람 데이터 없이)

Architecture:
    - 3D CNN: 과거 농도 시퀀스 인코딩
    - Conv3D: 정적 맵 (Terrain + Source) 인코딩
    - Transformer: 시계열 패턴 학습
    - Decoder: 미래 농도 맵 생성
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcentrationEncoder3D(nn.Module):
    """
    3D CNN for encoding concentration sequence
    Input: (B, T, D, H, W) - Time series of 3D concentration maps
    Output: (B, T, latent_dim) - Temporal feature sequence
    """
    def __init__(self, latent_dim=128):
        super().__init__()

        # 3D Convolutions to extract spatial-temporal features
        self.conv_layers = nn.Sequential(
            # Input: (B, 1, T, D, H, W) - treat T as channel-like dimension
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.MaxPool3d((1, 2, 2)),  # Pool only spatial dimensions

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # Global pooling per timestep
        )

        self.fc = nn.Linear(64, latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, D, H, W) - Past concentration sequence
        Returns:
            (B, T, latent_dim) - Encoded features per timestep
        """
        B, T, D, H, W = x.shape

        # Process each timestep through 3D CNN
        features = []
        for t in range(T):
            x_t = x[:, t:t+1, :, :, :]  # (B, 1, D, H, W)
            feat = self.conv_layers(x_t)  # (B, 64, 1, 1, 1)
            feat = feat.view(B, -1)  # (B, 64)
            feat = self.fc(feat)  # (B, latent_dim)
            features.append(feat)

        # Stack along time dimension
        features = torch.stack(features, dim=1)  # (B, T, latent_dim)
        return features


class StaticMapEncoder(nn.Module):
    """
    3D CNN for encoding static maps (Terrain + Source)
    Input: (B, 2, D, H, W)
    Output: (B, latent_dim)
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
        """
        Args:
            x: (B, 2, D, H, W) - Static maps [Terrain, Source]
        Returns:
            (B, latent_dim)
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TemporalTransformer(nn.Module):
    """
    Transformer for temporal pattern learning
    """
    def __init__(self, latent_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, latent_dim)
        Returns:
            (B, T, latent_dim)
        """
        return self.transformer(x)


class ConcentrationDecoder(nn.Module):
    """
    Decoder to generate future concentration map
    Input: (B, latent_dim) - Context vector
    Output: (B, 1, D, H, W) - Future concentration map
    """
    def __init__(self, latent_dim=128, output_shape=(21, 45, 45)):
        super().__init__()
        self.output_shape = output_shape
        D, H, W = output_shape

        # Calculate initial spatial dimensions after upsampling
        # We'll start from (D//8, H//8, W//8) and upsample 3 times
        init_d, init_h, init_w = D // 8, H // 8, W // 8

        self.fc = nn.Linear(latent_dim, 128 * init_d * init_h * init_w)
        self.init_shape = (128, init_d, init_h, init_w)

        self.decoder = nn.Sequential(
            # (128, D//8, H//8, W//8)
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            # (64, D//4, H//4, W//4)

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            # (32, D//2, H//2, W//2)

            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            # (16, D, H, W)

            nn.Conv3d(16, 1, kernel_size=3, padding=1)
            # (1, D, H, W)
        )

    def forward(self, x, target_shape=None):
        """
        Args:
            x: (B, latent_dim)
            target_shape: Optional (D, H, W) for dynamic output size
        Returns:
            (B, 1, D, H, W)
        """
        B = x.size(0)
        x = self.fc(x)
        x = x.view(B, *self.init_shape)
        x = self.decoder(x)

        # Use target_shape if provided, otherwise use default output_shape
        output_size = target_shape if target_shape is not None else self.output_shape
        x = F.interpolate(x, size=output_size, mode='trilinear', align_corners=False)

        # Non-negativity
        x = F.softplus(x)
        return x


class ConcentrationSeq2Seq(nn.Module):
    """
    Complete Sequence-to-Sequence Model for Concentration Prediction

    Architecture:
        Past Conc (B, T, D, H, W) → 3D CNN → (B, T, latent)
        Static Maps (B, 2, D, H, W) → 3D CNN → (B, latent)
        → Combine → Transformer → Decoder → Future Conc (B, 1, D, H, W)
    """
    def __init__(self, latent_dim=128, num_heads=4, num_layers=2,
                 dropout=0.1, output_shape=(21, 45, 45)):
        super().__init__()

        # Encoders
        self.conc_encoder = ConcentrationEncoder3D(latent_dim)
        self.static_encoder = StaticMapEncoder(in_channels=2, latent_dim=latent_dim)

        # Temporal modeling
        self.transformer = TemporalTransformer(
            latent_dim, num_heads, num_layers, dropout
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Decoder
        self.decoder = ConcentrationDecoder(latent_dim, output_shape)

    def forward(self, past_conc, static_maps):
        """
        Args:
            past_conc: (B, T, D, H, W) - Past concentration sequence
            static_maps: (B, 2, D, H, W) - Static maps [Terrain, Source]

        Returns:
            pred_conc: (B, 1, D, H, W) - Predicted future concentration
        """
        # Get target spatial dimensions from input
        B, T, D, H, W = past_conc.shape
        target_shape = (D, H, W)

        # Encode past concentration sequence
        conc_features = self.conc_encoder(past_conc)  # (B, T, latent_dim)

        # Encode static maps
        static_features = self.static_encoder(static_maps)  # (B, latent_dim)

        # Apply transformer to temporal features
        conc_features = self.transformer(conc_features)  # (B, T, latent_dim)

        # Take last timestep features
        last_features = conc_features[:, -1, :]  # (B, latent_dim)

        # Fuse with static features
        combined = torch.cat([last_features, static_features], dim=-1)  # (B, 2*latent_dim)
        fused = self.fusion(combined)  # (B, latent_dim)

        # Decode to future concentration with dynamic output size
        pred_conc = self.decoder(fused, target_shape=target_shape)  # (B, 1, D, H, W)

        return pred_conc
