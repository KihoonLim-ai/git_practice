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
import gc

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

# [사용자 모듈]
from dataset.dataset import get_time_split_datasets
from dataset.config_param import ConfigParam as Config
from model import ST_TransformerDeepONet


# --- 1. Fully Parameterized Loss Function ---
class PhysicsInformedLoss(nn.Module):
    # [수정] conc_weight_scale 기본값을 4.0으로 변경 (너무 크면 배경이 망가짐)
    def __init__(self, w_conc=10.0, w_wind=1.0, topk_ratio=0.05, conc_weight_scale=4.0):
        super().__init__()
        self.w_conc = w_conc
        self.w_wind = w_wind
        self.topk_ratio = topk_ratio
        self.conc_weight_scale = conc_weight_scale
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_w, true_w, pred_c, true_c):
        # 1. Concentration Loss (Weighted + Sniper)
        
        # [다시 켜기] 가중치 적용 (Softplus로 안전하게)
        # true_c가 음수일 때도 softplus는 0에 가까운 양수를 반환하므로 안전함
        safe_conc = F.softplus(true_c) 
        weights = 1.0 + self.conc_weight_scale * safe_conc
        
        # 픽셀별 MSE에 가중치 곱하기
        pixel_loss = self.mse(pred_c, true_c) * weights
        
        # Sniper Loss (상위 k%만 학습)
        # 배경(0)이 너무 많으므로, 오차가 큰 상위 5%만 골라내서 집중 타격
        k = int(pixel_loss.numel() * self.topk_ratio)
        if k < 1: k = 1
        topk_loss, _ = torch.topk(pixel_loss.view(-1), k)
        loss_c = topk_loss.mean()

        # 2. Wind Loss (Vector + Direction)
        loss_w_vec = self.mse(pred_w, true_w).mean()
        cos_sim = F.cosine_similarity(pred_w, true_w, dim=-1, eps=1e-8)
        loss_w_dir = (1.0 - cos_sim).mean()
        
        loss_w = loss_w_vec + loss_w_dir

        # 최종 Loss 합산
        # w_conc를 10.0 정도로 높게 주어, 농도 학습을 최우선으로 함
        return (self.w_conc * loss_c) + (self.w_wind * loss_w), loss_c, loss_w
    
# --- 2. Seed Setting ---
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 3. Training Loop ---
def train(config=None):
    # WandB 초기화
    with wandb.init(config=config) as run:
        config = wandb.config
        
        # 시드 고정 (재현성)
        seed_everything(config.seed if hasattr(config, 'seed') else 42)
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data Setup (전체 데이터 사용)
        train_ds, val_ds, _ = get_time_split_datasets(seq_len=30, pred_step=5)
        
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Model Setup 
        # [주의] model.py의 ST_TransformerDeepONet이 인자를 받도록 수정되어 있어야 함
        # 수정되지 않았다면 기본값으로 동작하지만, 튜닝을 위해 인자 전달 권장
        try:
            model = ST_TransformerDeepONet(
                latent_dim=config.latent_dim,
                dropout=config.dropout
            ).to(DEVICE)
        except TypeError:
            print("⚠️ Warning: Model does not accept arguments. Using default architecture.")
            model = ST_TransformerDeepONet().to(DEVICE)

        # Loss Setup
        criterion = PhysicsInformedLoss(
            w_conc=config.w_conc,
            w_wind=config.w_wind,
            topk_ratio=config.topk_ratio,
            conc_weight_scale=config.conc_weight_scale
        )
        
        # Optimizer Setup
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
        
        # Scheduler Setup
        total_steps = len(train_loader) * config.epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=config.lr, 
            total_steps=total_steps, 
            pct_start=config.warmup_ratio, # Warm-up 비율 튜닝
            anneal_strategy='cos'
        )
        
        best_val_loss = float('inf') # 최고 성능 기록용 (초기값: 무한대)
        
        # 저장 경로 생성
        save_dir = os.path.join(current_dir, 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        
        # --- Loop ---
        for epoch in range(config.epochs):
            model.train()
            loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
            
            total_loss_avg = 0
            
            for batch in loop:
                ctx_map, met_seq, coords_4d, gt_w, gt_c = [b.to(DEVICE) for b in batch]
                
                optimizer.zero_grad()
                pred_w, pred_c = model(ctx_map, met_seq, coords_4d)
                
                loss, l_c, l_w = criterion(pred_w, gt_w, pred_c, gt_c)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss_avg += loss.item()
                loop.set_postfix(loss=loss.item())
                
                # Step-wise Logging
                wandb.log({
                    "train_loss": loss.item(), 
                    "train_conc_loss": l_c.item(),
                    "train_wind_loss": l_w.item(),
                    "lr": optimizer.param_groups[0]['lr']
                })
            
            # Validation
            model.eval()
            val_loss = 0
            val_c_loss = 0
            val_w_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    ctx_map, met_seq, coords_4d, gt_w, gt_c = [b.to(DEVICE) for b in batch]
                    pred_w, pred_c = model(ctx_map, met_seq, coords_4d)
                    
                    loss, l_c, l_w = criterion(pred_w, gt_w, pred_c, gt_c)
                    
                    val_loss += loss.item()
                    val_c_loss += l_c.item()
                    val_w_loss += l_w.item()
            
            avg_val = val_loss / len(val_loader)
            avg_val_c = val_c_loss / len(val_loader)
            avg_val_w = val_w_loss / len(val_loader)
            
            wandb.log({
                "epoch": epoch + 1,
                "val_loss": avg_val,
                "val_conc_loss": avg_val_c,
                "val_wind_loss": avg_val_w
            })
            
            # Checkpoint (Best Model Only)
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                
                # 파일명에 run 이름을 넣어서 덮어쓰기 방지 (예: model_sweep-1_best.pth)
                ckpt_name = f"model_{run.name}_best.pth"
                ckpt_path = os.path.join(save_dir, ckpt_name)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'config': dict(config), # 설정값도 같이 저장하면 나중에 편함
                }, ckpt_path)
                
                print(f"  -> New Best Model Saved! (Val Loss: {best_val_loss:.4f})")

# --- 4. Sweep Configuration ---
if __name__ == "__main__":
    sweep_config = {
        'method': 'bayes', # 베이지안 최적화 (성능 좋은 파라미터 탐색)
        'metric': {
            'name': 'val_loss', 
            'goal': 'minimize'
        },
        'parameters': {
            # [1] Training Dynamics
            'epochs': {'value': 100}, # 고정 (비교를 위해)
            'batch_size': {'values': [16, 32]},
            'lr': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-3},
            'weight_decay': {'values': [1e-4, 1e-3, 5e-3]},
            'warmup_ratio': {'values': [0.1, 0.2]}, # Warm-up 길이 조절
            'seed': {'value': 42},

            # [2] Loss Function Tuning (핵심)
            'w_conc': {'values': [1.0, 2.0]}, # 농도 Loss 중요도
            'w_wind': {'values': [1.0, 0.5]}, # 바람 Loss 중요도
            'topk_ratio': {'values': [0.5, 1.0]}, # Sniper 범위 (5%, 10%, 20%)
            'conc_weight_scale': {'values': [1.0, 2.0]}, # 고농도 가중치 (1 + alpha*y)

            # [3] Model Architecture Tuning
            # model.py가 이 인자들을 받아야 적용됨
            'latent_dim': {'values': [128, 256]}, 
            'dropout': {'values': [0.1, 0.2, 0.3]} 
        }
    }
    
    # 프로젝트 이름 수정 필요
    sweep_id = wandb.sweep(sweep_config, entity="jhlee98", project="UAS_Final_Paper1_Sweep")
    wandb.agent(sweep_id, function=train, count=20) # 20번의 실험 수행
