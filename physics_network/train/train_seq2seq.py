"""
Training script for Sequence-to-Sequence Concentration Prediction Model
과거 30시간 농도 시계열 → 미래 1시간 농도 예측 (바람 데이터 없이)

Architecture:
    - Input: Past 30 timesteps of concentration + Static maps
    - Model: 3D CNN Encoder + Transformer + Decoder
    - Output: Future 1 timestep concentration map
"""
import os
import sys

# OpenMP 중복 로드 문제 해결
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.dataset_seq2seq import get_dataloaders_seq2seq
from model.model_seq2seq_v2 import ConcentrationSeq2Seq_v2 as ConcentrationSeq2Seq

# WandB (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ WandB not installed. Running without logging.")


# ========================= Configuration =========================
class TrainConfig:
    # Model Architecture
    HIDDEN_CHANNELS = 32      # ConvLSTM hidden channels
    NUM_LSTM_LAYERS = 2       # Number of ConvLSTM layers
    OUTPUT_SHAPE = (21, 45, 45)  # (D, H, W)

    # Data
    SEQ_LEN = 30          # Past 30 timesteps
    PRED_HORIZON = 1      # Predict 1 timestep ahead
    CROP_SIZE = 32        # Training crop size
    BATCH_SIZE = 8        # Reduce if GPU OOM
    NUM_WORKERS = 0       # Windows compatibility

    # Training
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Loss weights
    MSE_WEIGHT = 1.0
    PCC_WEIGHT = 0.5      # Spatial pattern preservation

    # LR Scheduler
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-6

    # Checkpointing
    CHECKPOINT_DIR = "checkpoints_seq2seq"
    SAVE_BEST_ONLY = True
    EARLY_STOP_PATIENCE = 30

    # WandB
    USE_WANDB = WANDB_AVAILABLE
    WANDB_PROJECT = "kari-seq2seq-concentration"
    WANDB_NAME = "seq2seq_nowind_v1"


# ========================= Loss Functions =========================
def pearson_correlation_loss(pred, target):
    """
    Pearson Correlation Coefficient Loss (1 - PCC)
    Measures spatial pattern similarity

    Args:
        pred: (B, 1, D, H, W)
        target: (B, 1, D, H, W)

    Returns:
        loss: scalar (1 - mean PCC across batch)
    """
    B = pred.size(0)
    pcc_sum = 0.0

    for i in range(B):
        pred_flat = pred[i].flatten()
        target_flat = target[i].flatten()

        # Normalize
        pred_mean = pred_flat.mean()
        target_mean = target_flat.mean()

        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean

        # Covariance / (std_pred * std_target)
        covariance = (pred_centered * target_centered).sum()
        pred_std = pred_centered.pow(2).sum().sqrt()
        target_std = target_centered.pow(2).sum().sqrt()

        pcc = covariance / (pred_std * target_std + 1e-8)
        pcc_sum += pcc

    mean_pcc = pcc_sum / B
    return 1.0 - mean_pcc  # Loss: minimize (1 - PCC)


def combined_loss(pred, target, mse_weight=1.0, pcc_weight=0.5):
    """
    Combined MSE + PCC Loss

    Args:
        pred: (B, 1, D, H, W)
        target: (B, 1, D, H, W)
        mse_weight: Weight for MSE loss
        pcc_weight: Weight for PCC loss

    Returns:
        total_loss, mse_loss, pcc_loss
    """
    mse_loss = nn.functional.mse_loss(pred, target)
    pcc_loss = pearson_correlation_loss(pred, target)

    total_loss = mse_weight * mse_loss + pcc_weight * pcc_loss

    return total_loss, mse_loss, pcc_loss


# ========================= Training Functions =========================
def train_one_epoch(model, train_loader, optimizer, device, cfg):
    """Single training epoch"""
    model.train()

    epoch_loss = 0.0
    epoch_mse = 0.0
    epoch_pcc_loss = 0.0

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, (past_conc, static_maps, future_conc) in enumerate(pbar):
        # Move to device
        past_conc = past_conc.to(device)      # (B, 30, 21, H, W)
        static_maps = static_maps.to(device)  # (B, 2, 21, H, W)
        future_conc = future_conc.to(device)  # (B, 1, 21, H, W)

        # Forward pass
        optimizer.zero_grad()
        pred_conc = model(past_conc, static_maps)  # (B, 1, 21, H, W)

        # Calculate loss
        loss, mse, pcc_l = combined_loss(
            pred_conc, future_conc,
            mse_weight=cfg.MSE_WEIGHT,
            pcc_weight=cfg.PCC_WEIGHT
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate metrics
        epoch_loss += loss.item()
        epoch_mse += mse.item()
        epoch_pcc_loss += pcc_l.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'mse': mse.item(),
            'pcc_loss': pcc_l.item()
        })

    num_batches = len(train_loader)
    return epoch_loss / num_batches, epoch_mse / num_batches, epoch_pcc_loss / num_batches


def validate(model, val_loader, device, cfg):
    """Validation loop"""
    model.eval()

    val_loss = 0.0
    val_mse = 0.0
    val_pcc_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)

        for past_conc, static_maps, future_conc in pbar:
            # Move to device
            past_conc = past_conc.to(device)
            static_maps = static_maps.to(device)
            future_conc = future_conc.to(device)

            # Forward pass
            pred_conc = model(past_conc, static_maps)

            # Calculate loss
            loss, mse, pcc_l = combined_loss(
                pred_conc, future_conc,
                mse_weight=cfg.MSE_WEIGHT,
                pcc_weight=cfg.PCC_WEIGHT
            )

            # Accumulate
            val_loss += loss.item()
            val_mse += mse.item()
            val_pcc_loss += pcc_l.item()

            pbar.set_postfix({'val_loss': loss.item()})

    num_batches = len(val_loader)
    return val_loss / num_batches, val_mse / num_batches, val_pcc_loss / num_batches


# ========================= Main Training Loop =========================
def main():
    cfg = TrainConfig()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    # WandB initialization
    if cfg.USE_WANDB:
        wandb.init(
            project=cfg.WANDB_PROJECT,
            name=cfg.WANDB_NAME,
            config={
                'hidden_channels': cfg.HIDDEN_CHANNELS,
                'num_lstm_layers': cfg.NUM_LSTM_LAYERS,
                'seq_len': cfg.SEQ_LEN,
                'pred_horizon': cfg.PRED_HORIZON,
                'batch_size': cfg.BATCH_SIZE,
                'learning_rate': cfg.LEARNING_RATE,
                'crop_size': cfg.CROP_SIZE,
                'mse_weight': cfg.MSE_WEIGHT,
                'pcc_weight': cfg.PCC_WEIGHT
            }
        )
        print("✅ WandB initialized")

    # Data loaders
    print("\n📦 Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders_seq2seq(
        batch_size=cfg.BATCH_SIZE,
        seq_len=cfg.SEQ_LEN,
        pred_horizon=cfg.PRED_HORIZON,
        crop_size=cfg.CROP_SIZE,
        num_workers=cfg.NUM_WORKERS
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Model
    print("\n🏗️ Building model...")
    model = ConcentrationSeq2Seq(
        hidden_channels=cfg.HIDDEN_CHANNELS,
        num_lstm_layers=cfg.NUM_LSTM_LAYERS,
        output_shape=cfg.OUTPUT_SHAPE
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {num_params:,}")

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE,
        min_lr=cfg.SCHEDULER_MIN_LR,
        verbose=True
    )

    # Checkpoint directory
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # Training loop
    print("\n🚀 Starting training...\n")
    print("=" * 70)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{cfg.NUM_EPOCHS}")
        print("-" * 70)

        # Train
        train_loss, train_mse, train_pcc_loss = train_one_epoch(
            model, train_loader, optimizer, device, cfg
        )

        # Validate
        val_loss, val_mse, val_pcc_loss = validate(
            model, val_loader, device, cfg
        )

        # LR Scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Print metrics
        print(f"  Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, PCC_loss: {train_pcc_loss:.6f})")
        print(f"  Val Loss:   {val_loss:.6f} (MSE: {val_mse:.6f}, PCC_loss: {val_pcc_loss:.6f})")
        print(f"  LR: {current_lr:.2e}")

        # WandB logging
        if cfg.USE_WANDB:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/mse': train_mse,
                'train/pcc_loss': train_pcc_loss,
                'val/loss': val_loss,
                'val/mse': val_mse,
                'val/pcc_loss': val_pcc_loss,
                'lr': current_lr
            })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "best_seq2seq.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': cfg.__dict__
            }, checkpoint_path)

            print(f"  ✅ Best model saved! (val_loss: {val_loss:.6f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= cfg.EARLY_STOP_PATIENCE:
            print(f"\n⚠️ Early stopping triggered (no improvement for {cfg.EARLY_STOP_PATIENCE} epochs)")
            break

    print("\n" + "=" * 70)
    print("🎉 Training completed!")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    print(f"   Model saved to: {cfg.CHECKPOINT_DIR}/best_seq2seq.pth")

    if cfg.USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
