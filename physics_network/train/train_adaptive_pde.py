"""
Adaptive PDE Training: ST_TransformerDeepONet with ReLoBRaLo loss balancing
Uses Relative Loss Balancing with Random Lookback for automatic weight adjustment.

Key Features:
- No additional hyperparameters needed
- Automatically balances data loss vs physics loss
- Slower-changing losses get higher weights
"""
import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
from tqdm import tqdm

# [경로 설정]
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.dataset import get_dataloaders
from dataset.physics_utils import make_batch_coords
from dataset.config_param import ConfigParam as Config
from model.model_adaptive_pde import ST_TransformerDeepONet_AdaptivePDE
from wind_loss import PhysicsInformedGridLoss

# ==========================================
# [설정] Adaptive PDE Training Configuration
# ==========================================
class TrainConfig:
    # 1. 경로 및 실험 설정
    PROJECT_NAME = "KARI_Ablation_Study"
    RUN_NAME = "4_AdaptivePDE_ReLoBRaLo"
    # Ablation study: 공정한 비교를 위해 pretrained weight 사용 안함
    WIND_CHECKPOINT = None
    SAVE_DIR = os.path.join(parent_dir, "..", "checkpoints_ablation")

    # 2. 하이퍼파라미터
    EPOCHS = 100
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    DROP_OUT = 0.1

    # 3. ReLoBRaLo 설정
    LOOKBACK = 10  # Loss history 길이

    # 4. PDE 설정
    DIFFUSION_COEFF = 0.1

    # 5. 데이터 설정
    SEQ_LEN = 30
    FUTURE_STEP = 1
    CROP_SIZE = 45


def compute_pcc_loss(pred, target):
    """Pattern Correlation Coefficient loss"""
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)

    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean

    cov = (pred_centered * target_centered).sum(dim=1)
    pred_std = pred_centered.pow(2).sum(dim=1).sqrt()
    target_std = target_centered.pow(2).sum(dim=1).sqrt()

    pcc = cov / (pred_std * target_std + 1e-8)
    return 1.0 - pcc.mean()


def train_adaptive_pde():
    cfg = TrainConfig()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    # WandB Init
    wandb.init(project=cfg.PROJECT_NAME, name=cfg.RUN_NAME, config=cfg.__dict__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Loaders
    print("Loading Data...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=cfg.BATCH_SIZE,
        seq_len=cfg.SEQ_LEN,
        future_step=cfg.FUTURE_STEP,
        crop_size=cfg.CROP_SIZE
    )

    # Model Init
    print("Initializing Adaptive PDE Model...")
    if cfg.WIND_CHECKPOINT and os.path.exists(cfg.WIND_CHECKPOINT):
        print(f"Loading Wind Weights from {cfg.WIND_CHECKPOINT}...")
        ckpt = torch.load(cfg.WIND_CHECKPOINT, map_location=device)
        saved_cfg = ckpt['config']

        loaded_latent_dim = int(saved_cfg.get('latent_dim', 256))
        loaded_fourier_scale = float(saved_cfg.get('fourier_scale', 20.0))
        loaded_in_channels = int(saved_cfg.get('in_channels', 5))

        model = ST_TransformerDeepONet_AdaptivePDE(
            latent_dim=loaded_latent_dim,
            dropout=cfg.DROP_OUT,
            fourier_scale=loaded_fourier_scale,
            diffusion_coeff=cfg.DIFFUSION_COEFF,
            lookback=cfg.LOOKBACK,
            in_channels=loaded_in_channels
        ).to(device)

        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  -> Loaded: latent_dim={loaded_latent_dim}, fourier_scale={loaded_fourier_scale}")
    else:
        print("🔧 Training from scratch (no pretrained checkpoint)")
        model = ST_TransformerDeepONet_AdaptivePDE(
            latent_dim=256,
            dropout=cfg.DROP_OUT,
            fourier_scale=20.0,
            diffusion_coeff=cfg.DIFFUSION_COEFF,
            lookback=cfg.LOOKBACK,
            in_channels=5
        ).to(device)

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # For validation (fixed weights)
    val_criterion = PhysicsInformedGridLoss(
        lambda_mse=1.0,
        lambda_pcc=0.5,
        lambda_phys=1.0
    )

    # Training Loop
    best_val_loss = float('inf')

    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0
        train_mse = 0
        train_pcc = 0
        train_phys = 0
        train_pde = 0

        # For tracking adaptive weights
        epoch_weights = {'mse': 0, 'pcc': 0, 'phys': 0, 'pde': 0}

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        for batch in loop:
            inp_vol, met_seq, target_vol, global_wind = [b.to(device) for b in batch]

            B, C, D, H, W = inp_vol.shape
            coords = make_batch_coords(B, D, H, W, device=device).requires_grad_(True)

            # Forward with PDE loss
            pred_w, pred_c, pde_loss = model(
                inp_vol, met_seq, coords, global_wind,
                compute_pde_loss=True
            )

            target_c = target_vol[:, 0, ...].reshape(B, -1, 1)
            target_w = inp_vol[:, 2:5, ...].permute(0, 2, 3, 4, 1).reshape(B, -1, 3)

            # Compute individual losses
            loss_mse = F.mse_loss(pred_c, target_c)
            loss_pcc = compute_pcc_loss(pred_c, target_c)

            # Physics loss (continuity)
            pred_w_reshaped = pred_w.reshape(B, D, H, W, 3)
            du_dx = (pred_w_reshaped[:, :, :, 1:, 0] - pred_w_reshaped[:, :, :, :-1, 0])
            dv_dy = (pred_w_reshaped[:, :, 1:, :, 1] - pred_w_reshaped[:, :, :-1, :, 1])
            dw_dz = (pred_w_reshaped[:, 1:, :, :, 2] - pred_w_reshaped[:, :-1, :, :, 2])

            # Trim to same size
            min_d = min(du_dx.shape[1], dv_dy.shape[1], dw_dz.shape[1])
            min_h = min(du_dx.shape[2], dv_dy.shape[2], dw_dz.shape[2])
            min_w = min(du_dx.shape[3], dv_dy.shape[3], dw_dz.shape[3])

            div = (du_dx[:, :min_d, :min_h, :min_w] +
                   dv_dy[:, :min_d, :min_h, :min_w] +
                   dw_dz[:, :min_d, :min_h, :min_w])
            loss_phys = torch.mean(div ** 2)

            # [핵심] Adaptive weight 계산
            loss_dict = {
                'mse': loss_mse,
                'pcc': loss_pcc,
                'phys': loss_phys,
                'pde': pde_loss
            }
            weights = model.compute_adaptive_weights(loss_dict)

            # Weighted total loss
            loss = sum(weights[k] * loss_dict[k] for k in loss_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mse += loss_mse.item()
            train_pcc += loss_pcc.item()
            train_phys += loss_phys.item()
            train_pde += pde_loss.item()

            # Track weights
            for k in epoch_weights:
                epoch_weights[k] += weights[k]

            loop.set_postfix(
                loss=loss.item(),
                w_mse=weights['mse'],
                w_pde=weights['pde']
            )

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inp_vol, met_seq, target_vol, global_wind = [b.to(device) for b in batch]
                B, C, D, H, W = inp_vol.shape
                coords = make_batch_coords(B, D, H, W, device=device)

                pred_w, pred_c = model(inp_vol, met_seq, coords, global_wind, compute_pde_loss=False)

                target_c = target_vol[:, 0, ...].reshape(B, -1, 1)
                target_w = inp_vol[:, 2:5, ...].permute(0, 2, 3, 4, 1).reshape(B, -1, 3)

                loss, _ = val_criterion(
                    pred_c, target_c, pred_w, coords,
                    target_w=target_w,
                    inp_vol=inp_vol
                )
                val_loss += loss.item()

        n_batches = len(train_loader)
        avg_train_loss = train_loss / n_batches
        avg_val_loss = val_loss / len(val_loader)

        # Average weights
        avg_weights = {k: v / n_batches for k, v in epoch_weights.items()}

        scheduler.step(avg_val_loss)

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_mse": train_mse / n_batches,
            "train_pcc": train_pcc / n_batches,
            "train_phys": train_phys / n_batches,
            "train_pde": train_pde / n_batches,
            "weight_mse": avg_weights['mse'],
            "weight_pcc": avg_weights['pcc'],
            "weight_phys": avg_weights['phys'],
            "weight_pde": avg_weights['pde'],
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        print(f"  Weights -> MSE:{avg_weights['mse']:.3f} PCC:{avg_weights['pcc']:.3f} PHYS:{avg_weights['phys']:.3f} PDE:{avg_weights['pde']:.3f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'latent_dim': model.map_encoder.fc.out_features,
                    'fourier_scale': model.trunk.fourier_scale,
                    'diffusion_coeff': cfg.DIFFUSION_COEFF,
                    'lookback': cfg.LOOKBACK,
                    'in_channels': 5
                }
            }, os.path.join(cfg.SAVE_DIR, "adaptive_pde_best.pth"))
            print(f"✅ Saved Best Adaptive PDE Model (Loss: {best_val_loss:.6f})")

    wandb.finish()

if __name__ == "__main__":
    train_adaptive_pde()
