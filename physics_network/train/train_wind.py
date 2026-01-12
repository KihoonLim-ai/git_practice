#train_wind.py

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
from train.wind_loss import PhysicsInformedGridLoss  # [핵심] 물리 Loss 가져오기

# ==========================================
# [설정]
# ==========================================
class WindTrainConfig:
    PROJECT_NAME = "KARI_Wind_Physics"
    RUN_NAME = "Wind_Pretrain_Physics_v1"
    SAVE_DIR = "checkpoints"
    
    EPOCHS = 100         # 충분히 학습
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    
    # Loss 가중치 조절
    LAMBDA_MSE = 1.0     # 데이터(U, V) 신뢰도
    LAMBDA_PHYS = 1.0    # [중요] 물리 법칙(W) 강제 주입
    
    SEQ_LEN = 30
    FUTURE_STEP = 1
    CROP_SIZE = 45       # 전체 맵 학습

def train_wind_physics():
    # 1. Init
    cfg = WindTrainConfig()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    
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
    
    # 3. Model Init
    print("Initializing Model...")
    model = ST_TransformerDeepONet(
        latent_dim=256,      # 기본 설정
        fourier_scale=20.0   # 고주파 성분(지형 디테일) 학습 강화
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # [핵심 변경] 단순 MSE가 아니라 Physics Loss 사용
    # PCC는 농도용이므로 여기선 0.0, Physics는 1.0으로 켭니다.
    criterion = PhysicsInformedGridLoss(
        lambda_mse=cfg.LAMBDA_MSE,
        lambda_pcc=0.0,       # 바람 학습엔 PCC 불필요
        lambda_phys=cfg.LAMBDA_PHYS
    )
    
    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0
        train_mse = 0
        train_phys = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        for batch in loop:
            # Unpack
            inp_vol, met_seq, target_vol, global_wind = [b.to(device) for b in batch]
            
            # Coords
            B, C, D, H, W = inp_vol.shape
            coords = make_batch_coords(B, D, H, W, device=device)
            
            # Forward
            # pred_w: (B, D*H*W, 3) -> (u, v, w)
            pred_w, _ = model(inp_vol, met_seq, coords, global_wind)
            
            # Target (Wind는 1,2,3번 채널: U, V, W)
            # W 채널이 0이어도 상관없음. MSE는 U, V를 맞추고, Physics Loss가 W를 만들어냄.
            target_w = inp_vol[:, 2:5, ...].permute(0, 2, 3, 4, 1).reshape(B, -1, 3)    
                   
            # Loss Calculation
            # pred_c는 None을 줘서 계산 생략
            loss, loss_dict = criterion(None, None, pred_w, coords, target_w=target_w, inp_vol=inp_vol)    
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += loss_dict['mse']
            train_phys += loss_dict['phys']
            
            loop.set_postfix(loss=loss.item(), mse=loss_dict['mse'], phys=loss_dict['phys'])
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inp_vol, met_seq, target_vol, global_wind = [b.to(device) for b in batch]
                
                # [수정됨] 여기서 B, C, D, H, W를 검증 데이터에 맞게 다시 가져와야 함!
                # (이전에는 Training Loop의 B=16이 그대로 쓰여서 오류 발생)
                B, C, D, H, W = inp_vol.shape  
                
                coords = make_batch_coords(B, D, H, W, device=device)
                
                pred_w, _ = model(inp_vol, met_seq, coords, global_wind)
                
                # 이제 갱신된 B를 사용하므로 안전함
                target_w = inp_vol[:, 2:5, ...].permute(0, 2, 3, 4, 1).reshape(B, -1, 3)    
                
                loss, loss_dict = criterion(None, None, pred_w, coords, target_w=target_w, inp_vol=inp_vol)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} (Phys: {train_phys/len(train_loader):.6f}) | Val Loss: {avg_val_loss:.6f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_phys": train_phys/len(train_loader),
            "val_loss": avg_val_loss
        })
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 이름 변경: wind_pretrain_best.pth
            save_path = os.path.join(cfg.SAVE_DIR, "wind_pretrain_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {'latent_dim': 256, 'dropout': 0.1, 'fourier_scale': 20.0}
            }, save_path)
            print(f"✅ Saved Best Physics-Wind Model to {save_path}")

if __name__ == "__main__":
    train_wind_physics()