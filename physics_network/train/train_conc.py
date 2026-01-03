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
# 1. Advanced Physics-Informed Loss
# ==============================================================================
class PhysicsInformedGridLoss(nn.Module):
    def __init__(self, w_conc=1.0, w_wind=50.0, w_pcc=1.0, w_phys=0.1, topk_ratio=0.1, 
                 conc_weight_scale=10.0, dx=100.0, dy=100.0, dz=10.0):
        super().__init__()
        self.w_conc = w_conc
        self.w_wind = w_wind
        self.w_pcc = w_pcc
        self.w_phys = w_phys
        self.topk_ratio = topk_ratio
        self.conc_weight_scale = conc_weight_scale
        
        # 격자 간격 (미터)
        self.dx, self.dy, self.dz = dx, dy, dz
        self.mse_none = nn.MSELoss(reduction='none')

    def calc_pcc(self, pred, target):
        """Pearson Correlation Coefficient Loss (패턴 유사도)"""
        p_flat = pred.view(pred.size(0), -1)
        t_flat = target.view(target.size(0), -1)
        p_mean = p_flat - p_flat.mean(dim=1, keepdim=True)
        t_mean = t_flat - t_flat.mean(dim=1, keepdim=True)
        
        cos = (p_mean * t_mean).sum(dim=1) / (p_mean.norm(dim=1) * t_mean.norm(dim=1) + 1e-8)
        return 1.0 - cos.mean()

    def calc_advection_residual(self, pred_w_flat, pred_c_flat):
        """이류 방정식 잔차: u*dC/dx + v*dC/dy"""
        # (B, N, C) -> (B, C, D, H, W) 복원
        B = pred_w_flat.shape[0]
        w = pred_w_flat.view(B, Config.NZ, Config.NY, Config.NX, 3).permute(0, 4, 1, 2, 3)
        c = pred_c_flat.view(B, Config.NZ, Config.NY, Config.NX, 1).permute(0, 4, 1, 2, 3)
        
        u, v = w[:, 0:1, ...], w[:, 1:2, ...]
        
        # Central Difference with Padding (차원 유지)
        c_px = F.pad(c, (1, 1, 0, 0, 0, 0), mode='replicate')
        c_py = F.pad(c, (0, 0, 1, 1, 0, 0), mode='replicate')
        
        dc_dx = (c_px[..., 2:] - c_px[..., :-2]) / (2 * self.dx)
        dc_dy = (c_py[..., 2:, :] - c_py[..., :-2, :]) / (2 * self.dy)
        
        residual = (u * dc_dx) + (v * dc_dy)
        return torch.mean(residual**2)

    def forward(self, pred_wind, true_wind, pred_conc, true_conc):
        # 1. Weighted Top-K Concentration Loss
        pixel_loss = self.mse_none(pred_conc, true_conc)
        
        # 고농도 가중치 + 과소평가(Under-estimation) 페널티
        val_weights = 1.0 + (self.conc_weight_scale * F.softplus(true_conc))
        asym_weights = torch.where(true_conc > pred_conc, 3.0, 1.0) # 과소평가 시 3배 페널티
        
        weighted_loss = pixel_loss * val_weights * asym_weights
        
        # Top-K: 하위 90% 허공(0)은 무시하고 상위 10% 중요한 곳만 학습
        k = max(1, int(weighted_loss.numel() * self.topk_ratio))
        loss_c, _ = torch.topk(weighted_loss.view(-1), k)
        loss_c = loss_c.mean()

        # 2. Wind MSE
        loss_w = F.mse_loss(pred_wind, true_wind)

        # 3. Shape & Physics Loss
        loss_pcc = self.calc_pcc(pred_conc, true_conc)
        loss_phys = self.calc_advection_residual(pred_wind, pred_conc)

        total = (self.w_conc * loss_c) + (self.w_wind * loss_w) + \
                (self.w_pcc * loss_pcc) + (self.w_phys * loss_phys)
                
        return total, loss_c, loss_w, loss_pcc, loss_phys

# ==============================================================================
# 2. Training Helper Functions
# ==============================================================================
def get_grid_coords(device):
    z = torch.linspace(0, 1, Config.NZ); y = torch.linspace(0, 1, Config.NY); x = torch.linspace(0, 1, Config.NX)
    gz, gy, gx = torch.meshgrid(z, y, x, indexing='ij')
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).to(device)
    t = torch.zeros((coords.shape[0], 1), device=device)
    return torch.cat([coords, t], dim=-1)

def seed_everything(seed):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.backends.cudnn.deterministic = True

# ==============================================================================
# 3. Main Training Loop
# ==============================================================================
def train_joint_physics(config=None):
    with wandb.init(config=config, project="kari_physics_joint", entity="jhlee98") as run:
        cfg = wandb.config
        seed_everything(cfg.seed)
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data & Model Setup
        train_ds, val_ds, _ = get_time_split_datasets(seq_len=30)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
        
        model = ST_TransformerDeepONet(latent_dim=cfg.latent_dim, fourier_scale=cfg.fourier_scale).to(DEVICE)
        
        # Pre-trained Wind 로드
        if os.path.exists("checkpoints/wind_master_best.pth"):
            ckpt = torch.load("checkpoints/wind_master_best.pth", map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print("✅ Pre-trained Wind Model Loaded.")

        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = PhysicsInformedGridLoss(topk_ratio=cfg.topk_ratio).to(DEVICE)
        
        coords = get_grid_coords(DEVICE)
        best_val_loss = float('inf')

        for epoch in range(cfg.epochs):
            model.train()
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for batch in loop:
                inp, met, tgt, g_wind = [b.to(DEVICE) for b in batch]
                batch_coords = coords.unsqueeze(0).expand(inp.shape[0], -1, -1)
                
                # Target 준비 (B, N, C)
                gt_c = tgt[:, 0:1, ...].permute(0, 2, 3, 4, 1).reshape(inp.shape[0], -1, 1)
                gt_w = tgt[:, 1:4, ...].permute(0, 2, 3, 4, 1).reshape(inp.shape[0], -1, 3)
                
                optimizer.zero_grad()
                p_wind, p_conc = model(inp, met, batch_coords, g_wind)
                
                total_loss, lc, lw, lpcc, lphys = criterion(p_wind, gt_w, p_conc, gt_c)
                
                total_loss.backward()
                optimizer.step()
                
                loop.set_postfix(conc=lc.item(), phys=lphys.item())
                wandb.log({"train/total": total_loss.item(), "train/conc": lc.item(), 
                           "train/pcc": lpcc.item(), "train/phys": lphys.item()})

            # Validation
            model.eval()
            v_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    inp, met, tgt, g_wind = [b.to(DEVICE) for b in batch]
                    batch_coords = coords.unsqueeze(0).expand(inp.shape[0], -1, -1)
                    gt_c = tgt[:, 0:1, ...].permute(0, 2, 3, 4, 1).reshape(inp.shape[0], -1, 1)
                    gt_w = tgt[:, 1:4, ...].permute(0, 2, 3, 4, 1).reshape(inp.shape[0], -1, 3)
                    
                    p_w, p_c = model(inp, met, batch_coords, g_wind)
                    loss, _, _, _, _ = criterion(p_w, gt_w, p_c, gt_c)
                    v_total += loss.item()
            
            avg_val = v_total / len(val_loader)
            print(f"   Epoch {epoch+1} Val Loss: {avg_val:.6f}")
            wandb.log({"val/total": avg_val})
            
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save({'model_state_dict': model.state_dict(), 'config': dict(cfg)}, "checkpoints/joint_physics_best.pth")

if __name__ == "__main__":
    config = {
        'epochs': 50, 
        'batch_size': 16, 
        'lr': 5e-4, 
        'latent_dim': 256, 
        'fourier_scale': 20.0, 
        'topk_ratio': 0.1,
        'dropout': 0.1, 
        'weight_decay': 1e-4, 
        'seed': 42
    }
    train_joint_physics(config)