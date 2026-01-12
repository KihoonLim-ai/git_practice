import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from tqdm import tqdm

# [경로 설정]
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.dataset import get_dataloaders
from dataset.physics_utils import make_batch_coords
from dataset.config_param import ConfigParam as Config
from model.model import ST_TransformerDeepONet
from wind_loss import PhysicsInformedGridLoss

# ==========================================
# [설정] Joint Training Configuration
# ==========================================
class TrainConfig:
    # 1. 경로 및 실험 설정
    PROJECT_NAME = "KARI_Joint_Diffusion"
    RUN_NAME = "Joint_Physics_Boundary_v1" # 이름 변경 (Boundary 추가됨)
    WIND_CHECKPOINT = "checkpoints/wind_pretrain_best.pth" # 검증된 바람 모델
    SAVE_DIR = "checkpoints"
    
    # 2. 하이퍼파라미터
    EPOCHS = 100
    BATCH_SIZE = 2 # 메모리에 맞춰 조절
    LEARNING_RATE = 1e-4
    DROP_OUT = 0.1
    
    # 3. Loss 가중치 (중요)
    # 농도(Concentration) 학습
    LAMBDA_MSE = 1.0     # 농도 값 정확도
    LAMBDA_PCC = 0.5     # 농도 분포 패턴(상관계수)
    
    # 바람(Wind) 물리 보정
    # 이미 train_wind.py에서 배웠지만, 농도와 함께 미세 조정되도록 유지
    LAMBDA_PHYS = 1.0    # 질량보존 + 지형경계조건(New!)
    
    # 4. 데이터 설정
    SEQ_LEN = 30
    FUTURE_STEP = 1
    CROP_SIZE = 45       # 전체 맵 사용 (안정성)

def train_joint():
    # 1. Init
    cfg = TrainConfig()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    
    # WandB Init
    wandb.init(project=cfg.PROJECT_NAME, name=cfg.RUN_NAME, config=cfg.__dict__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Data Loaders
    print("Loading Data...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=cfg.BATCH_SIZE,
        seq_len=cfg.SEQ_LEN,
        future_step=cfg.FUTURE_STEP,
        crop_size=cfg.CROP_SIZE
    )
    
    # 바람 모델 가중치 로드
    # 3. Model Init & Load Pretrained Wind
    print("Initializing Model...")
    
    if os.path.exists(cfg.WIND_CHECKPOINT):
        print(f"Loading Wind Config & Weights from {cfg.WIND_CHECKPOINT}...")
        ckpt = torch.load(cfg.WIND_CHECKPOINT, map_location=device)
        saved_cfg = ckpt['config']
        
        # 저장된 설정값 불러오기 (없으면 기본값 128, 10.0 사용)
        loaded_latent_dim = int(saved_cfg.get('latent_dim', 256))
        loaded_fourier_scale = float(saved_cfg.get('fourier_scale', 20.0))
        
        print(f"  -> Loaded Config: latent_dim={loaded_latent_dim}, fourier_scale={loaded_fourier_scale}")
        
        # 동적으로 모델 생성
        model = ST_TransformerDeepONet(
            latent_dim=loaded_latent_dim,
            dropout=cfg.DROP_OUT,       # Dropout은 현재 학습 설정(cfg)을 따름
            fourier_scale=loaded_fourier_scale
        ).to(device)
        
        # 가중치 로드 (strict=False: 농도 헤드는 초기화 유지)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        
    else:
        print("⚠️ WARNING: Pretrained Wind Checkpoint NOT FOUND. using Default (128, 10.0)")
        # 체크포인트가 없을 경우 기본값으로 생성 (혹은 에러 처리)
        model = ST_TransformerDeepONet(
            latent_dim=256,
            dropout=cfg.DROP_OUT,
            fourier_scale=20.0
        ).to(device)

    # [중요] Unfreeze All Parameters
    for param in model.parameters():
        param.requires_grad = True
    print("  -> All Parameters Unfrozen. Physics Loss will correct the Wind Field & Conc.")
    # 4. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Physics Loss (지형 경계 조건 포함됨)
    criterion = PhysicsInformedGridLoss(
        lambda_mse=cfg.LAMBDA_MSE, 
        lambda_pcc=cfg.LAMBDA_PCC, 
        lambda_phys=cfg.LAMBDA_PHYS
    )
    
    # 5. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0
        train_mse = 0
        train_pcc = 0
        train_phys = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        for batch in loop:
            # Unpack
            inp_vol, met_seq, target_vol, global_wind = [b.to(device) for b in batch]
            
            # Coords
            B, C, D, H, W = inp_vol.shape
            coords = make_batch_coords(B, D, H, W, device=device)
            
            # Forward
            # pred_w: (B, N, 3), pred_c: (B, N, 1)
            pred_w, pred_c = model(inp_vol, met_seq, coords, global_wind)
            
            # Target Setup
            # 1) Concentration Target (From target_vol Channel 0)
            target_c = target_vol[:, 0, ...].reshape(B, -1, 1) 
            
            # 2) Wind Target (From inp_vol Channel 2,3,4 -> U,V,W)
            # MSE Loss에서 U,V 학습용으로 사용됨
            target_w = inp_vol[:, 2:5, ...].permute(0, 2, 3, 4, 1).reshape(B, -1, 3)
            
            # Loss Calculation
            # [핵심] inp_vol을 전달해야 지형 경계 조건(Boundary Loss)이 계산됨!
            loss, loss_dict = criterion(
                pred_c, target_c, pred_w, coords, 
                target_w=target_w, 
                inp_vol=inp_vol  # 🔥 필수 인자
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            train_loss += loss.item()
            train_mse += loss_dict.get('mse', 0)
            train_pcc += loss_dict.get('pcc', 0)
            train_phys += loss_dict.get('phys', 0)
            
            loop.set_postfix(
                loss=loss.item(), 
                mse=loss_dict.get('mse', 0), 
                phys=loss_dict.get('phys', 0)
            )
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inp_vol, met_seq, target_vol, global_wind = [b.to(device) for b in batch]
                
                # B 갱신 (Val Loop 안전장치)
                B, C, D, H, W = inp_vol.shape
                coords = make_batch_coords(B, D, H, W, device=device)
                
                pred_w, pred_c = model(inp_vol, met_seq, coords, global_wind)
                
                target_c = target_vol[:, 0, ...].reshape(B, -1, 1)
                target_w = inp_vol[:, 2:5, ...].permute(0, 2, 3, 4, 1).reshape(B, -1, 3)
                
                loss, _ = criterion(
                    pred_c, target_c, pred_w, coords, 
                    target_w=target_w, 
                    inp_vol=inp_vol # 🔥 필수 인자
                )
                val_loss += loss.item()
        
        # Stat Calculation
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        # WandB Log
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_mse": train_mse / len(train_loader),
            "train_pcc": train_pcc / len(train_loader),
            "train_phys": train_phys / len(train_loader),
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]['lr']
        })
        
        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} (Phys: {train_phys/len(train_loader):.4f}) | Val: {avg_val_loss:.4f}")
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': ckpt['config'] if 'ckpt' in locals() else {'latent_dim': 128}
            }, os.path.join(cfg.SAVE_DIR, "joint_best.pth"))
            print(f"✅ Saved Best Joint Model (Loss: {best_val_loss:.6f})")

if __name__ == "__main__":
    train_joint()