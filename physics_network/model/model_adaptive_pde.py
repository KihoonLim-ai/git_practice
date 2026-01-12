"""
Adaptive PDE Model: ST_TransformerDeepONet with Gradient-Based Loss Weighting
Uses ReLoBRaLo (Relative Loss Balancing with Random Lookback) for automatic weight adjustment
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import components from soft PDE model
from model.model_soft_pde import (
    Conv3dBranch,
    TransformerObsBranch,
    SpatioTemporalTrunk,
    AdvectionDiffusionResidual
)

# ==============================================================================
# 1. Adaptive Loss Weighting (ReLoBRaLo)
# ==============================================================================

class ReLoBRaLo:
    """
    Relative Loss Balancing with Random Lookback

    State-of-the-art loss balancing for PINNs (arXiv:2110.09813)
    - No additional hyperparameters needed
    - Better than GradNorm and SoftAdapt
    - Adaptive weights based on relative loss changes

    Usage:
        balancer = ReLoBRaLo(lookback=10)
        for epoch in range(epochs):
            for batch in dataloader:
                pred_wind, pred_conc, pde_loss = model(...)

                loss_dict = {
                    'mse': F.mse_loss(pred_conc, target),
                    'pcc': compute_pcc_loss(...),
                    'phys': compute_physics_loss(...),
                    'pde': pde_loss
                }

                weights = balancer.compute_weights(loss_dict)
                loss_total = sum(weights[k] * loss_dict[k] for k in loss_dict)
    """
    def __init__(self, lookback=10):
        self.lookback = lookback
        self.loss_history = {}

    def compute_weights(self, current_losses):
        """
        Compute adaptive weights based on relative loss changes

        Args:
            current_losses: dict of {loss_name: loss_value}

        Returns:
            dict of {loss_name: weight}
        """
        weights = {}

        for name, current_loss in current_losses.items():
            # Initialize history if needed
            if name not in self.loss_history:
                self.loss_history[name] = []

            # Add to history
            self.loss_history[name].append(current_loss.item())

            # Keep only recent history
            if len(self.loss_history[name]) > self.lookback:
                self.loss_history[name] = self.loss_history[name][-self.lookback:]

            # Compute relative change (higher change → higher weight)
            if len(self.loss_history[name]) >= 2:
                recent_mean = np.mean(self.loss_history[name][-self.lookback:])
                older_mean = np.mean(self.loss_history[name][:-1])

                if older_mean > 0:
                    relative_change = abs(recent_mean - older_mean) / older_mean
                else:
                    relative_change = 1.0
            else:
                relative_change = 1.0

            # Inverse weighting: slower changing losses get higher weight
            weights[name] = 1.0 / (relative_change + 1e-8)

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Fallback to uniform weights
            n = len(weights)
            weights = {k: 1.0 / n for k in weights}

        return weights

# ==============================================================================
# 2. Main Model with Adaptive Weighting
# ==============================================================================

class ST_TransformerDeepONet_AdaptivePDE(nn.Module):
    """
    Model with adaptive PDE loss weighting (ReLoBRaLo)

    The model itself is the same, but includes a built-in ReLoBRaLo balancer
    for convenience. The balancer should be used in the training loop.
    """
    def __init__(self, latent_dim=128, dropout=0.1, num_heads=4, fourier_scale=10.0,
                 diffusion_coeff=0.1, lookback=10, in_channels=5):
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

        # Built-in adaptive loss balancer
        self.loss_balancer = ReLoBRaLo(lookback=lookback)

    def forward(self, ctx_map, obs_seq, query_coords, global_wind, compute_pde_loss=False, source=None):
        """
        Forward pass with PDE loss computation
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

    def compute_adaptive_weights(self, loss_dict):
        """
        Convenience method to compute adaptive weights

        Args:
            loss_dict: {'mse': loss_mse, 'pcc': loss_pcc, 'phys': loss_phys, 'pde': pde_loss}

        Returns:
            dict of adaptive weights
        """
        return self.loss_balancer.compute_weights(loss_dict)
