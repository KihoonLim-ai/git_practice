import os
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

# [사용자 모듈]
from dataset.dataset import get_time_split_datasets
from dataset.config_param import ConfigParam as Config
from model import ST_TransformerDeepONet

# --- 1. Fully Parameterized Loss Function ---
class PhysicsInformedLoss(nn.Module):
    def __init__(self, w_conc=1.0, w_wind=1.0, topk_ratio=0.05, conc_weight_scale=10.0):
        super().__init__()
        self.w_conc = w_conc
        self.w_wind = w_wind
        self.topk_ratio = topk_ratio
        self.conc_weight_scale = conc_weight_scale
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_w, true_w, pred_c, true_c):
        # 1. Concentration Loss (Weighted + Sniper)
        
        # [핵심 수정] 가중치 계산 안전장치
        # true_c는 Z-score라 음수가 될 수 있음.
        # F.softplus를 통과시켜 음수를 0 근처로, 양수를 그대로 양수로 만듦.
        # 예: -3 -> 0.05 (가중치 ~1), +3 -> 3.0 (가중치 높음)
        safe_conc = F.softplus(true_c) 
        weights = 1.0 + self.conc_weight_scale * safe_conc
        
        pixel_loss = self.mse(pred_c, true_c) * weights
        
        # Sniper Loss
        k = int(pixel_loss.numel() * self.topk_ratio)
        if k < 1: k = 1
        topk_loss, _ = torch.topk(pixel_loss.view(-1), k)
        loss_c = topk_loss.mean()

        # 2. Wind Loss (Vector + Direction)
        loss_w_vec = self.mse(pred_w, true_w).mean()
        cos_sim = F.cosine_similarity(pred_w, true_w, dim=-1, eps=1e-8)
        loss_w_dir = (1.0 - cos_sim).mean()
        
        loss_w = loss_w_vec + loss_w_dir

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
            # Sweep 중에는 디스크 용량 문제로 Best만 저장하거나 생략하기도 함
            # 여기서는 생략하거나 필요시 추가

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
            'batch_size': {'values': [32, 64]},
            'lr': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-3},
            'weight_decay': {'values': [1e-4, 1e-3, 5e-3]},
            'warmup_ratio': {'values': [0.1, 0.2]}, # Warm-up 길이 조절
            'seed': {'value': 42},

            # [2] Loss Function Tuning (핵심)
            'w_conc': {'values': [1.0, 2.0]}, # 농도 Loss 중요도
            'w_wind': {'values': [1.0, 0.5]}, # 바람 Loss 중요도
            'topk_ratio': {'values': [0.05, 0.1, 0.2]}, # Sniper 범위 (5%, 10%, 20%)
            'conc_weight_scale': {'values': [5.0, 10.0, 20.0]}, # 고농도 가중치 (1 + alpha*y)

            # [3] Model Architecture Tuning
            # model.py가 이 인자들을 받아야 적용됨
            'latent_dim': {'values': [128, 256]}, 
            'dropout': {'values': [0.1, 0.2, 0.3]} 
        }
    }
    
    # 프로젝트 이름 수정 필요
    sweep_id = wandb.sweep(sweep_config, entity="jhlee98", project="UAS_Final_Paper1_Sweep")
    wandb.agent(sweep_id, function=train, count=20) # 20번의 실험 수행
