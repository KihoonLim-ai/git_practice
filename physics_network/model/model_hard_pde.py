"""
Hard PDE Model: ST_TransformerDeepONet with Hard Constraints (Output Transforms)
Automatically enforces physics constraints via output transformations
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
# 1. Hard Constraint Layers
# ==============================================================================

class HardConstraintLayer(nn.Module):
    """
    Applies hard constraints via output transformation

    Constraints enforced:
    1. Zero concentration inside terrain
    2. Non-negative concentration everywhere
    3. (Optional) Boundary conditions
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_conc, terrain_mask=None):
        """
        Enforce hard constraints on concentration

        Args:
            pred_conc: (B, N, 1) - raw network output
            terrain_mask: (B, N, 1) - 1 where terrain exists, 0 in air

        Returns:
            constrained_conc: (B, N, 1) - guaranteed to satisfy constraints
        """
        # Ensure non-negativity
        pred_conc = F.softplus(pred_conc)

        # Enforce zero concentration inside terrain
        if terrain_mask is not None:
            pred_conc = pred_conc * (1.0 - terrain_mask)

        return pred_conc

class TerrainMaskGenerator(nn.Module):
    """
    Generates terrain mask from coordinates and terrain map
    """
    def __init__(self):
        super().__init__()

    def forward(self, coords, terrain_height):
        """
        Args:
            coords: (B, N, 4) - [x, y, z, t] normalized [0, 1]
            terrain_height: (B, H, W) - terrain elevation map [0, 1]

        Returns:
            mask: (B, N, 1) - 1 where coords are inside terrain, 0 otherwise
        """
        B, N, _ = coords.shape
        _, H, W = terrain_height.shape

        # Extract spatial coordinates
        x_norm = coords[..., 0]  # (B, N)
        y_norm = coords[..., 1]  # (B, N)
        z_norm = coords[..., 2]  # (B, N)

        # Convert normalized coords to grid indices
        x_idx = (x_norm * (W - 1)).long().clamp(0, W - 1)  # (B, N)
        y_idx = (y_norm * (H - 1)).long().clamp(0, H - 1)  # (B, N)

        # Sample terrain height at query locations
        terrain_at_coords = []
        for b in range(B):
            heights = terrain_height[b, y_idx[b], x_idx[b]]  # (N,)
            terrain_at_coords.append(heights)
        terrain_at_coords = torch.stack(terrain_at_coords, dim=0)  # (B, N)

        # Mask: 1 if z < terrain_height, 0 otherwise
        mask = (z_norm <= terrain_at_coords).float().unsqueeze(-1)  # (B, N, 1)

        return mask

# ==============================================================================
# 2. Main Model with Hard Constraints
# ==============================================================================

class ST_TransformerDeepONet_HardPDE(nn.Module):
    """
    Model with hard physics constraints (guaranteed satisfaction)

    Constraints are enforced via output transformations:
    - Concentration is automatically zero inside terrain
    - Concentration is always non-negative

    Can optionally use soft PDE loss as well (hybrid approach)
    """
    def __init__(self, latent_dim=128, dropout=0.1, num_heads=4, fourier_scale=10.0,
                 diffusion_coeff=0.1, use_soft_pde=True, in_channels=5):
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

        # Hard constraint layer
        self.hard_constraint = HardConstraintLayer()
        self.terrain_mask_gen = TerrainMaskGenerator()

        # Optional soft PDE loss (hybrid approach)
        self.use_soft_pde = use_soft_pde
        if use_soft_pde:
            self.pde_residual = AdvectionDiffusionResidual(diffusion_coeff=diffusion_coeff)

    def forward(self, ctx_map, obs_seq, query_coords, global_wind,
                compute_pde_loss=False, source=None, apply_hard_constraints=True):
        """
        Forward pass with hard constraint application

        Args:
            apply_hard_constraints: If True, apply output transformations
            compute_pde_loss: If True, also compute soft PDE residual (hybrid)

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

        # Concentration prediction (raw)
        conc_input = torch.cat([merged, pred_wind], dim=-1)
        pred_conc = self.head_conc(conc_input)

        # Apply hard constraints if requested
        if apply_hard_constraints:
            # Extract terrain from ctx_map (channel 0)
            terrain_map = ctx_map[:, 0, 0, :, :]  # (B, H, W) - take first z-slice as terrain
            terrain_mask = self.terrain_mask_gen(query_coords, terrain_map)
            pred_conc = self.hard_constraint(pred_conc, terrain_mask)
        else:
            # Just ensure non-negativity
            pred_conc = F.softplus(pred_conc)

        # Optional soft PDE loss (hybrid approach)
        if compute_pde_loss and self.use_soft_pde:
            pde_loss = self.pde_residual(pred_conc, query_coords, pred_wind, source)
            return pred_wind, pred_conc, pde_loss

        return pred_wind, pred_conc
