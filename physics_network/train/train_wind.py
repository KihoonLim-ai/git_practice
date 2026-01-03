import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# [경로 설정]
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.dataset import get_time_split_datasets
from dataset.config_param import ConfigParam as Config
from model.model import ST_TransformerDeepONet

# ==============================================================================
# 1. Physics Loss (Residual Wind Learning용)
# ==============================================================================
class PhysicsInformedGridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_w, gt_w):
        """
        pred_w: (B, N, 3) -> Flattened Wind (u, v, w)
        gt_w:   (B, 3, D, H, W) -> Volumetric Wind
        """
        # Shape 맞추기 (Volumetric -> Flattened)
        # GT: (B, 3, D, H, W) -> (B, 3, N) -> (B, N, 3)
        b, c, d, h, w = gt_w.shape
        gt_flat = gt_w.permute(0, 2, 3, 4, 1).reshape(b, -1, 3)
        
        # MSE Loss 계산
        loss = self.mse(pred_w, gt_flat)
        return loss

# ==============================================================================
# 2. Utils
# ==============================================================================
def get_grid_coords(device):
    z = torch.linspace(0, 1, Config.NZ)
    y = torch.linspace(0, 1, Config.NY)
    x = torch.linspace(0, 1, Config.NX)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(device)
    t = torch.zeros((coords.shape[0], 1), device=device)
    return torch.cat([coords, t], dim=-1)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ==============================================================================
# 3. Main Training Loop
# ==============================================================================
def train_wind_only(config=None):
    # [중요] 프로젝트 이름 확인
    with wandb.init(config=config, project="kari_wind_pretrain_v2", entity="jhlee98") as run:
        cfg = wandb.config
        seed_everything(cfg.seed)
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"=== 🌪️ Phase 1: Wind Field Pre-training (Global Condition) ===")
        
        # 1. Data Setup
        train_ds, val_ds, _ = get_time_split_datasets(seq_len=30)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # 2. Model Setup
        model = ST_TransformerDeepONet(
            latent_dim=int(cfg.latent_dim), 
            dropout=cfg.dropout,
            fourier_scale=cfg.fourier_scale
        ).to(DEVICE)
        
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = PhysicsInformedGridLoss()
        
        grid_coords = get_grid_coords(DEVICE)
        
        best_loss = float('inf')
        
        for epoch in range(cfg.epochs):
            model.train()
            loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
            
            epoch_loss = 0
            
            for batch in loop:
                # [핵심] 4개 변수 받기 (global_wind 추가됨)
                inp_vol, met_seq, target_vol, global_wind = [b.to(DEVICE) for b in batch]
                
                # Target: Channel 1(U), 2(V), 3(W)
                gt_w = target_vol[:, 1:, ...] 
                
                # 좌표 생성
                batch_coords = grid_coords.unsqueeze(0).expand(inp_vol.shape[0], -1, -1)
                
                optimizer.zero_grad()
                
                # [핵심] 모델에 global_wind 전달 -> 상층풍 강제 주입
                pred_w, _ = model(inp_vol, met_seq, batch_coords, global_wind)
                
                loss = criterion(pred_w, gt_w)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                
                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                wandb.log({"train_loss": loss.item()})

            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    inp_vol, met_seq, target_vol, global_wind = [b.to(DEVICE) for b in batch]
                    gt_w = target_vol[:, 1:, ...]
                    
                    batch_coords = grid_coords.unsqueeze(0).expand(inp_vol.shape[0], -1, -1)
                    
                    # Validation에서도 global_wind 전달
                    pred_w, _ = model(inp_vol, met_seq, batch_coords, global_wind)
                    
                    loss = criterion(pred_w, gt_w)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss})
            print(f"   Ep {epoch+1} | Val Loss: {avg_val_loss:.6f}")
            
            # Save Best Model (Loss 기준)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                os.makedirs("checkpoints", exist_ok=True)
                # 파일명: wind_master1.pth (농도 학습 때 불러올 이름)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'config': dict(cfg),
                }, "checkpoints/wind_master2.pth")
                print(f"   ✅ Best Model Saved! (Loss: {best_loss:.6f})")

if __name__ == "__main__":
    config = {
        'epochs': 50,
        'batch_size': 16,
        'lr': 1e-3,
        'latent_dim': 256,
        'fourier_scale': 20.0,
        'dropout': 0.1,
        'grad_clip': 1.0,
        'weight_decay': 1e-4,
        'seed': 42
    }
    train_wind_only(config)