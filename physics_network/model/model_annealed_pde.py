"""
Annealed PDE Model: ST_TransformerDeepONet with Epoch-Based Loss Annealing
Gradually increases PDE weight during training (0.1 → 1.0)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import components from soft PDE model
from model.model_soft_pde import (
    Conv3dBranch,
    TransformerObsBranch,
    SpatioTemporalTrunk,
    AdvectionDiffusionResidual
)

# ==============================================================================
# 1. Annealing Scheduler
# ==============================================================================

def get_physics_weight(epoch, total_epochs=100):
    """
    Epoch-based annealing schedule for physics loss weight

    Phase 1 (0-30%): Data-driven (physics weight = 0.1)
    Phase 2 (30-70%): Integration (linear ramp 0.1 → 1.0)
    Phase 3 (70-100%): Physics-refined (physics weight = 1.0)

    Args:
        epoch: Current epoch number
        total_epochs: Total number of training epochs

    Returns:
        float: Physics loss weight (λ_pde)
    """
    if epoch < 0.3 * total_epochs:
        # Early training: Focus on data fitting
        return 0.1
    elif epoch < 0.7 * total_epochs:
        # Mid training: Gradually enforce physics
        progress = (epoch - 0.3 * total_epochs) / (0.4 * total_epochs)
        return 0.1 + 0.9 * progress  # Linear ramp 0.1 → 1.0
    else:
        # Late training: Full physics enforcement
        return 1.0

# ==============================================================================
# 2. Main Model with Annealing
# ==============================================================================

class ST_TransformerDeepONet_AnnealedPDE(nn.Module):
    """
    Model with epoch-based PDE loss annealing

    Usage in training loop:
        for epoch in range(total_epochs):
            lambda_pde = get_physics_weight(epoch, total_epochs)
            for batch in dataloader:
                pred_wind, pred_conc, pde_loss = model(..., compute_pde_loss=True)
                loss_total = loss_mse + loss_pcc + loss_phys + lambda_pde * pde_loss
    """
    def __init__(self, latent_dim=128, dropout=0.1, num_heads=4, fourier_scale=10.0,
                 diffusion_coeff=0.1, total_epochs=100, in_channels=5):
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

        # Store total epochs for annealing
        self.total_epochs = total_epochs
        self.current_epoch = 0  # Should be updated externally

    def set_epoch(self, epoch):
        """Update current epoch for annealing schedule"""
        self.current_epoch = epoch

    def get_pde_weight(self):
        """Get current PDE weight based on epoch"""
        return get_physics_weight(self.current_epoch, self.total_epochs)

    def forward(self, ctx_map, obs_seq, query_coords, global_wind, compute_pde_loss=False, source=None):
        """
        Forward pass with annealing-aware PDE loss

        Note: PDE loss is always computed when compute_pde_loss=True,
        but the weight should be applied externally using get_pde_weight()
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
            pde_loss = self.pde_residual(pred_conc, query_coords, pred_wind, source)
            return pred_wind, pred_conc, pde_loss

        return pred_wind, pred_conc
