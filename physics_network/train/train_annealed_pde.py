"""
Annealed PDE Training: ST_TransformerDeepONet with epoch-based PDE weight annealing
PDE weight gradually increases from 0.1 to 1.0 during training.

Schedule:
- Phase 1 (0-30%): λ_pde = 0.1 (Data-driven)
- Phase 2 (30-70%): λ_pde = 0.1 → 1.0 (Linear ramp)
- Phase 3 (70-100%): λ_pde = 1.0 (Physics-refined)
"""
import os
import sys
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

# [경로 설정]
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.dataset import get_dataloaders
from dataset.physics_utils import make_batch_coords
from dataset.config_param import ConfigParam as Config
from model.model_annealed_pde import ST_TransformerDeepONet_AnnealedPDE
from wind_loss import PhysicsInformedGridLoss

# ==========================================
# [설정] Annealed PDE Training Configuration
# ==========================================
class TrainConfig:
    # 1. 경로 및 실험 설정
    PROJECT_NAME = "KARI_Ablation_Study"
    RUN_NAME = "3_AnnealedPDE"
    # Ablation study: 공정한 비교를 위해 pretrained weight 사용 안함
    WIND_CHECKPOINT = None
    SAVE_DIR = os.path.join(parent_dir, "..", "checkpoints_ablation")

    # 2. 하이퍼파라미터
    EPOCHS = 100
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    DROP_OUT = 0.1

    # 3. Loss 가중치
    LAMBDA_MSE = 1.0
    LAMBDA_PCC = 0.5
    LAMBDA_PHYS = 1.0
    # LAMBDA_PDE는 annealing schedule에 의해 자동 조정됨

    # 4. PDE 설정
    DIFFUSION_COEFF = 0.1

    # 5. 데이터 설정
    SEQ_LEN = 30
    FUTURE_STEP = 1
    CROP_SIZE = 45

def train_annealed_pde():
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
    print("Initializing Annealed PDE Model...")
    if cfg.WIND_CHECKPOINT and os.path.exists(cfg.WIND_CHECKPOINT):
        print(f"Loading Wind Weights from {cfg.WIND_CHECKPOINT}...")
        ckpt = torch.load(cfg.WIND_CHECKPOINT, map_location=device)
        saved_cfg = ckpt['config']

        loaded_latent_dim = int(saved_cfg.get('latent_dim', 256))
        loaded_fourier_scale = float(saved_cfg.get('fourier_scale', 20.0))
        loaded_in_channels = int(saved_cfg.get('in_channels', 5))

        model = ST_TransformerDeepONet_AnnealedPDE(
            latent_dim=loaded_latent_dim,
            dropout=cfg.DROP_OUT,
            fourier_scale=loaded_fourier_scale,
            diffusion_coeff=cfg.DIFFUSION_COEFF,
            total_epochs=cfg.EPOCHS,
            in_channels=loaded_in_channels
        ).to(device)

        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  -> Loaded: latent_dim={loaded_latent_dim}, fourier_scale={loaded_fourier_scale}")
    else:
        print("🔧 Training from scratch (no pretrained checkpoint)")
        model = ST_TransformerDeepONet_AnnealedPDE(
            latent_dim=256,
            dropout=cfg.DROP_OUT,
            fourier_scale=20.0,
            diffusion_coeff=cfg.DIFFUSION_COEFF,
            total_epochs=cfg.EPOCHS,
            in_channels=5
        ).to(device)

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = PhysicsInformedGridLoss(
        lambda_mse=cfg.LAMBDA_MSE,
        lambda_pcc=cfg.LAMBDA_PCC,
        lambda_phys=cfg.LAMBDA_PHYS
    )

    # Training Loop
    best_val_loss = float('inf')

    for epoch in range(cfg.EPOCHS):
        model.train()

        # [핵심] 현재 epoch 업데이트 및 PDE weight 가져오기
        model.set_epoch(epoch)
        lambda_pde = model.get_pde_weight()

        train_loss = 0
        train_mse = 0
        train_pcc = 0
        train_phys = 0
        train_pde = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} (λ_pde={lambda_pde:.3f})")
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

            # Base loss
            base_loss, loss_dict = criterion(
                pred_c, target_c, pred_w, coords,
                target_w=target_w,
                inp_vol=inp_vol
            )

            # Total loss with annealed PDE weight
            loss = base_loss + lambda_pde * pde_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mse += loss_dict.get('mse', 0)
            train_pcc += loss_dict.get('pcc', 0)
            train_phys += loss_dict.get('phys', 0)
            train_pde += pde_loss.item()

            loop.set_postfix(
                loss=loss.item(),
                pde=pde_loss.item(),
                weight=lambda_pde
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

                loss, _ = criterion(
                    pred_c, target_c, pred_w, coords,
                    target_w=target_w,
                    inp_vol=inp_vol
                )
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_pde = train_pde / len(train_loader)

        scheduler.step(avg_val_loss)

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_mse": train_mse / len(train_loader),
            "train_pcc": train_pcc / len(train_loader),
            "train_phys": train_phys / len(train_loader),
            "train_pde": avg_train_pde,
            "lambda_pde": lambda_pde,  # Annealing 상태 로깅
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch+1} | λ_pde={lambda_pde:.3f} | Train: {avg_train_loss:.4f} (PDE: {avg_train_pde:.6f}) | Val: {avg_val_loss:.4f}")

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
                    'total_epochs': cfg.EPOCHS,
                    'in_channels': 5
                }
            }, os.path.join(cfg.SAVE_DIR, "annealed_pde_best.pth"))
            print(f"✅ Saved Best Annealed PDE Model (Loss: {best_val_loss:.6f})")

    wandb.finish()

if __name__ == "__main__":
    train_annealed_pde()
