import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import gc 

# [사용자 지정 경로 유지]
from dataset.dataset import AermodDataset
from model import RecurrentDeepONet

# ==========================================
# 0. Reproducibility
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. Train Function for Sweep
# ==========================================
def train(config=None):
    # API 객체를 통해 현재 프로젝트의 전체 실행 횟수를 조회하여 번호를 매깁니다.
    api = wandb.Api()
    try:
        runs = api.runs("jhlee98/kari-onestop-uas")
        run_idx = len(runs)
    except:
        run_idx = 0
        run_count = 1

    # 메모리 누수 방지 및 에러 핸들링을 위한 try-finally 구조
    try:
        # WandB 초기화
        with wandb.init(config=config) as run:
            config = wandb.config
            
            # [Run 이름 설정] 예: KARI_Sweep_15_8j2h
            run.name = f"kari_sweep_20251224_{run_idx}"
                        
            # 시드 설정
            seed_everything(config.seed)
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # --- Data Setup ---
            full_dataset = AermodDataset(mode='train', seq_len=config.seq_len)
            
            val_size = int(len(full_dataset) * config.val_split)
            train_size = len(full_dataset) - val_size
            
            generator = torch.Generator().manual_seed(config.seed)
            train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)
            
            train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            
            # --- Model Setup ---
            model = RecurrentDeepONet().to(DEVICE)
            
            # --- Optimizer & Scheduler ---
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            
            # 기본 MSE Criterion (Wind용)
            criterion = nn.MSELoss()
            
            # --- Training Loop ---
            best_val_loss = float('inf')
            
            for epoch in range(config.epochs):
                model.train()
                total_train_loss = 0
                train_wind_loss = 0
                train_conc_loss = 0
                
                # Tqdm 진행바에 Run Name 표시
                loop = tqdm(train_loader, desc=f"[{run.name}] Epoch {epoch+1}", leave=False)
                
                for batch in loop:
                    ctx_map, met_seq, coords, gt_wind, gt_conc = [b.to(DEVICE) for b in batch]
                    
                    pred_wind, pred_conc = model(ctx_map, met_seq, coords)
                    
                    # 1. Wind Loss (기본 MSE)
                    loss_w = criterion(pred_wind, gt_wind)
                    
                    # 2. Weighted MSE for Concentration
                    # 농도가 높은 'Plume' 영역에 가중치를 주어 0만 예측하는 현상을 방지
                    weights = torch.ones_like(gt_conc)
                    weights[gt_conc > 0.1] = 5.0
                    weights[gt_conc > 2.0] = 20.0
                    weights[gt_conc > 5.0] = 50.0
                    weights[gt_conc > 7.0] = 70.0
                    
                    loss_c = torch.mean(weights * (pred_conc - gt_conc)**2)
                    
                    # 최종 Loss 합산 (WandB Config 가중치 적용)
                    loss = (config.loss_w_weight * loss_w) + (config.loss_c_weight * loss_c)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    train_wind_loss += loss_w.item()
                    train_conc_loss += loss_c.item()
                    
                    loop.set_postfix(loss=loss.item())
                
                # --- Validation ---
                model.eval()
                total_val_loss = 0
                val_wind_loss = 0
                val_conc_loss = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        ctx_map, met_seq, coords, gt_wind, gt_conc = [b.to(DEVICE) for b in batch]
                        
                        pred_wind, pred_conc = model(ctx_map, met_seq, coords)
                        
                        loss_w = criterion(pred_wind, gt_wind)
                        
                        # Validation에서도 동일한 가중치 적용
                        weights = torch.ones_like(gt_conc)
                        weights[gt_conc > 0.1] = 5.0 
                        weights[gt_conc > 5.0] = 20.0
                        loss_c = torch.mean(weights * (pred_conc - gt_conc)**2)
                        
                        loss = (config.loss_w_weight * loss_w) + (config.loss_c_weight * loss_c)
                        
                        total_val_loss += loss.item()
                        val_wind_loss += loss_w.item()
                        val_conc_loss += loss_c.item()
                
                avg_val_loss = total_val_loss / len(val_loader)
                scheduler.step(avg_val_loss)
                
                # --- WandB Logging ---
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": total_train_loss / len(train_loader),
                    "val_loss": avg_val_loss,
                    "val_loss_wind": val_wind_loss / len(val_loader),
                    "val_loss_conc": val_conc_loss / len(val_loader),
                    "lr": optimizer.param_groups[0]['lr']
                })
                
                # [저장] Run Name을 파일명에 포함
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    os.makedirs("./checkpoints", exist_ok=True)
                    save_path = f"./checkpoints/best_model_{run.name}.pth"
                    torch.save(model.state_dict(), save_path)
            
            print(f"Run {run.name} finished. Best Val Loss: {best_val_loss}")

    except Exception as e:
        print(f"Error in run: {e}")
        if wandb.run is not None:
            wandb.run.finish(exit_code=1) 
            
    finally:
        # [중요] 매 실험 종료 시 GPU 메모리 초기화
        print(f"Cleaning up GPU memory...")
        if 'model' in locals(): del model
        if 'optimizer' in locals(): del optimizer
        if 'train_loader' in locals(): del train_loader
        if 'val_loader' in locals(): del val_loader
        
        gc.collect()
        torch.cuda.empty_cache()

# ==========================================
# 2. Sweep Setup & Run
# ==========================================
if __name__ == "__main__":
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'epochs': {'value': 50},
            'seed': {'value': 42},
            'val_split': {'value': 0.1},
            
            # 안전한 배치 사이즈 범위
            'batch_size': {'values': [8, 16, 32]}, 
            
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2},
            'weight_decay': {'values': [0.0, 1e-5, 1e-4]},
            'seq_len': {'values': [4, 6, 8]},
            
            'loss_w_weight': {'value': 1.0},
            # 농도 가중치 탐색 범위
            'loss_c_weight': {'distribution': 'log_uniform_values', 'min': 1.0, 'max': 5.0} 
        }
    }
    
    # Sweep ID 생성 (jhlee98/kari-onestop-uas)
    sweep_id = wandb.sweep(
        sweep_config, 
        entity="jhlee98", 
        project="kari-onestop-uas"
    )
    
    # 총 20회의 최적화 시도
    wandb.agent(sweep_id, function=train, count=20)