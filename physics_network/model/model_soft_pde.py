"""
Soft PDE Model: ST_TransformerDeepONet with PDE Residual Loss (Soft Constraints)
Adds advection-diffusion PDE residual computation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. Sub-Modules (same as baseline)
# ==============================================================================

class Conv3dBranch(nn.Module):
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
# 2. PDE Residual Computation (NEW!)
# ==============================================================================

class AdvectionDiffusionResidual(nn.Module):
    """
    Computes residual of advection-diffusion equation:
    ∂C/∂t + u·∇C = D∇²C + S

    Uses PyTorch automatic differentiation
    """
    def __init__(self, diffusion_coeff=0.1):
        super().__init__()
        self.D = diffusion_coeff

    def forward(self, pred_conc, coords, pred_wind, source=None):
        """
        Args:
            pred_conc: (B, N, 1) - predicted concentration
            coords: (B, N, 4) - [x, y, z, t] with requires_grad=True
            pred_wind: (B, N, 3) - [u, v, w]
            source: (B, N, 1) - emission source term (optional)

        Returns:
            pde_residual: scalar loss (mean squared PDE violation)
        """
        # Enable gradient computation
        if not coords.requires_grad:
            coords.requires_grad_(True)

        # Compute all derivatives at once for efficiency
        # First derivatives
        grad_outputs = torch.ones_like(pred_conc)
        grads = torch.autograd.grad(
            outputs=pred_conc,
            inputs=coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]  # (B, N, 4)

        dC_dx = grads[..., 0:1]  # X derivative
        dC_dy = grads[..., 1:2]  # Y derivative
        dC_dz = grads[..., 2:3]  # Z derivative
        dC_dt = grads[..., 3:4]  # Time derivative

        # Second derivatives (for diffusion term)
        d2C_dx2 = torch.autograd.grad(
            outputs=dC_dx,
            inputs=coords,
            grad_outputs=torch.ones_like(dC_dx),
            create_graph=True,
            retain_graph=True
        )[0][..., 0:1]

        d2C_dy2 = torch.autograd.grad(
            outputs=dC_dy,
            inputs=coords,
            grad_outputs=torch.ones_like(dC_dy),
            create_graph=True,
            retain_graph=True
        )[0][..., 1:2]

        d2C_dz2 = torch.autograd.grad(
            outputs=dC_dz,
            inputs=coords,
            grad_outputs=torch.ones_like(dC_dz),
            create_graph=True,
            retain_graph=True
        )[0][..., 2:3]

        # Extract wind components
        u, v, w = pred_wind[..., 0:1], pred_wind[..., 1:2], pred_wind[..., 2:3]

        # Advection term: u·∇C
        advection = u * dC_dx + v * dC_dy + w * dC_dz

        # Diffusion term: D∇²C
        diffusion = self.D * (d2C_dx2 + d2C_dy2 + d2C_dz2)

        # PDE residual: ∂C/∂t + u·∇C - D∇²C - S = 0
        if source is not None:
            residual = dC_dt + advection - diffusion - source
        else:
            residual = dC_dt + advection - diffusion

        # Return mean squared residual
        return torch.mean(residual ** 2)

# ==============================================================================
# 3. Main Model with Soft PDE Constraints
# ==============================================================================

class ST_TransformerDeepONet_SoftPDE(nn.Module):
    """
    Model with soft PDE constraints (residual added to loss)
    """
    def __init__(self, latent_dim=128, dropout=0.1, num_heads=4, fourier_scale=10.0,
                 diffusion_coeff=0.1, in_channels=5):
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

        # PDE residual calculator
        self.pde_residual = AdvectionDiffusionResidual(diffusion_coeff=diffusion_coeff)

    def forward(self, ctx_map, obs_seq, query_coords, global_wind, compute_pde_loss=False, source=None):
        """
        Forward pass with optional PDE residual computation

        Args:
            compute_pde_loss: If True, compute and return PDE residual
            source: Source term for PDE (B, N, 1)

        Returns:
            If compute_pde_loss=False: (pred_wind, pred_conc)
            If compute_pde_loss=True: (pred_wind, pred_conc, pde_loss)
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

        if compute_pde_loss:
            # Compute PDE residual
            pde_loss = self.pde_residual(pred_conc, query_coords, pred_wind, source)
            return pred_wind, pred_conc, pde_loss

        return pred_wind, pred_conc
