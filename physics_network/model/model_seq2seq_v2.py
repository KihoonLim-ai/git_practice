"""
Improved Sequence-to-Sequence Model for Concentration Prediction
과거 농도 시계열 → 미래 농도 예측 (바람 데이터 없이)

Key Improvements:
- Proper temporal modeling with ConvLSTM
- Preserves spatial structure throughout encoding
- U-Net style decoder with skip connections
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell for spatial-temporal modeling
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Combined convolution for input, forget, cell, output gates
        self.conv = nn.Conv3d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, hidden_state):
        """
        Args:
            x: (B, C_in, D, H, W)
            hidden_state: tuple of (h, c) each (B, C_hidden, D, H, W)
        Returns:
            h_next, c_next
        """
        h, c = hidden_state

        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)  # (B, C_in + C_hidden, D, H, W)
        gates = self.conv(combined)  # (B, 4*C_hidden, D, H, W)

        # Split into gates
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMEncoder(nn.Module):
    """
    Encoder with ConvLSTM for temporal sequence processing
    Maintains spatial structure while encoding temporal dynamics
    """
    def __init__(self, input_channels=1, hidden_channels=32, num_layers=2):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Initial conv to extract features
        self.input_conv = nn.Sequential(
            nn.Conv3d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # ConvLSTM layers
        self.lstm_cells = nn.ModuleList([
            ConvLSTMCell(hidden_channels, hidden_channels)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, T, D, H, W) - sequence of concentration maps
        Returns:
            h_final: (B, hidden_channels, D, H, W) - final hidden state
            all_h: list of (B, hidden_channels, D, H, W) for each layer
        """
        B, T, D, H, W = x.shape

        # Initialize hidden states
        hidden_states = []
        for _ in range(self.num_layers):
            h = torch.zeros(B, self.hidden_channels, D, H, W, device=x.device)
            c = torch.zeros(B, self.hidden_channels, D, H, W, device=x.device)
            hidden_states.append((h, c))

        # Process sequence
        for t in range(T):
            x_t = x[:, t:t+1, :, :, :]  # (B, 1, D, H, W)
            x_t = self.input_conv(x_t)  # (B, hidden_channels, D, H, W)

            # Pass through LSTM layers
            for layer_idx in range(self.num_layers):
                h, c = self.lstm_cells[layer_idx](x_t, hidden_states[layer_idx])
                hidden_states[layer_idx] = (h, c)
                x_t = h  # Use output as input to next layer

        # Extract final hidden states
        all_h = [h for h, c in hidden_states]
        h_final = all_h[-1]  # Last layer's hidden state

        return h_final, all_h


class StaticEncoder(nn.Module):
    """
    Encoder for static maps (Terrain + Source Height-Aware)
    """
    def __init__(self, in_channels=2, out_channels=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 2, D, H, W) - [Terrain, Source (height-aware)]
        Returns:
            (B, out_channels, D, H, W)
        """
        return self.encoder(x)


class UNetDecoder(nn.Module):
    """
    U-Net style decoder with skip connections
    Gradually upsamples from encoded features to full resolution
    """
    def __init__(self, hidden_channels=32, output_shape=(21, 45, 45)):
        super().__init__()
        self.output_shape = output_shape

        # Fusion of temporal + static features
        self.fusion = nn.Sequential(
            nn.Conv3d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv3d(hidden_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, temporal_features, static_features, target_shape):
        """
        Args:
            temporal_features: (B, hidden_channels, D, H, W)
            static_features: (B, hidden_channels, D, H, W)
            target_shape: (D, H, W)
        Returns:
            (B, 1, D, H, W)
        """
        # Concatenate temporal and static features
        combined = torch.cat([temporal_features, static_features], dim=1)

        # Fuse features
        fused = self.fusion(combined)

        # Decode
        out = self.decoder(fused)

        # Resize to target shape if needed
        if out.shape[2:] != target_shape:
            out = F.interpolate(out, size=target_shape, mode='trilinear', align_corners=False)

        # Non-negativity
        out = F.softplus(out)

        return out


class ConcentrationSeq2Seq_v2(nn.Module):
    """
    Improved Seq2Seq model with proper temporal modeling

    Architecture:
        Past Conc (B, T, D, H, W) → ConvLSTM Encoder → (B, C, D, H, W)
        Static Maps (B, 2, D, H, W) [Terrain, Source height-aware] → Static Encoder → (B, C, D, H, W)
        → Fusion + Decoder → Future Conc (B, 1, D, H, W)
    """
    def __init__(self, hidden_channels=32, num_lstm_layers=2, output_shape=(21, 45, 45)):
        super().__init__()
        self.output_shape = output_shape

        # Temporal encoder (ConvLSTM)
        self.temporal_encoder = ConvLSTMEncoder(
            input_channels=1,
            hidden_channels=hidden_channels,
            num_layers=num_lstm_layers
        )

        # Static encoder
        self.static_encoder = StaticEncoder(
            in_channels=2,
            out_channels=hidden_channels
        )

        # Decoder
        self.decoder = UNetDecoder(
            hidden_channels=hidden_channels,
            output_shape=output_shape
        )

    def forward(self, past_conc, static_maps):
        """
        Args:
            past_conc: (B, T, D, H, W) - Past concentration sequence
            static_maps: (B, 2, D, H, W) - Static maps [Terrain, Source (height-aware)]
        Returns:
            pred_conc: (B, 1, D, H, W) - Predicted future concentration
        """
        # Get target shape from input
        B, T, D, H, W = past_conc.shape
        target_shape = (D, H, W)

        # Encode temporal sequence
        temporal_features, _ = self.temporal_encoder(past_conc)  # (B, C, D, H, W)

        # Encode static maps
        static_features = self.static_encoder(static_maps)  # (B, C, D, H, W)

        # Decode to prediction
        pred_conc = self.decoder(temporal_features, static_features, target_shape)

        return pred_conc


# Backward compatibility: alias for new version
ConcentrationSeq2Seq = ConcentrationSeq2Seq_v2
