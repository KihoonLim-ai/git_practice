import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

# [경로 설정]
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.dataset import get_time_split_datasets
from dataset.config_param import ConfigParam as Config
from model.model import ST_TransformerDeepONet

# ==========================================
# [설정]
# ==========================================
CHECKPOINT_PATH = "checkpoints/best_model_winter-sweep-1.pth" # 학습된 모델 경로
BATCH_SIZE = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# [Metrics 함수]
# ==========================================
def calculate_metrics(pred, target):
    """기본적인 RMSE, MAE 계산"""
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - target))
    return rmse, mae

def calculate_fac2(pred, target):
    """
    FAC2: Fraction of predictions within a factor of 2
    0.5 * target <= pred <= 2.0 * target 인 비율
    (0으로 나누는 것 방지 위해 eps 추가)
    """
    eps = 1e-7
    ratio = (pred + eps) / (target + eps)
    valid_mask = (target > eps) # 실제 농도가 있는 곳만 평가 (배경 제외)
    
    if np.sum(valid_mask) == 0:
        return 0.0
    
    fac2_mask = (ratio >= 0.5) & (ratio <= 2.0)
    # Target이 0이 아닌 곳 중에서 맞춘 비율
    fac2 = np.sum(fac2_mask & valid_mask) / np.sum(valid_mask)
    return fac2

def get_grid_coords(device):
    """모델 입력용 좌표 생성"""
    z = torch.linspace(0, 1, Config.NZ)
    y = torch.linspace(0, 1, Config.NY)
    x = torch.linspace(0, 1, Config.NX)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(device)
    t = torch.zeros((coords.shape[0], 1), device=device)
    return torch.cat([coords, t], dim=-1)

# ==========================================
# [Main Evaluation]
# ==========================================
def evaluate():
    print(f"=== Starting Quantitative Evaluation ===")
    
    # 1. Load Data & Stats
    _, val_ds, stats = get_time_split_datasets(seq_len=30, pred_step=5)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Scaling Factors (역정규화용)
    scale_wind = stats['scale_wind']
    # Conc는 Log scaling 되어있을 수 있으므로 확인 필요 (현재 코드는 Log+Norm 가정)
    # dataset.py에 저장된 conc_mean, conc_std 필요. 
    # 만약 stats에 없다면 임시값 혹은 dataset 객체에서 가져옴
    conc_mean = getattr(val_ds, 'conc_mean', 0.0)
    conc_std = getattr(val_ds, 'conc_std', 1.0)
    
    print(f"   -> Scale Info: Wind Max={scale_wind:.2f}, Conc Mean={conc_mean:.2f}, Std={conc_std:.2f}")

    # 2. Load Model
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    cfg = checkpoint['config']
    
    model = ST_TransformerDeepONet(
        latent_dim=int(cfg['latent_dim']), 
        dropout=cfg['dropout'],
        fourier_scale=cfg['fourier_scale']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("   -> Model Loaded Successfully.")

    # 3. Inference Loop
    grid_coords = get_grid_coords(DEVICE)
    
    results = {
        'conc_rmse': [], 'conc_fac2': [],
        'wind_rmse': [], 'wind_cosine': [],
        'phys_div': []
    }
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inp_vol, met_seq, target_vol = [b.to(DEVICE) for b in batch]
            
            # (A) Inference
            B = inp_vol.shape[0]
            batch_coords = grid_coords.unsqueeze(0).expand(B, -1, -1)
            pred_w, pred_c = model(inp_vol, met_seq, batch_coords)
            
            # (B) Reshape & Denormalize
            # Prediction: (B, N, 3), (B, N, 1) -> (B, D, H, W)
            D, H, W = Config.NZ, Config.NY, Config.NX
            
            # --- Wind ---
            u_pred = pred_w[..., 0].view(B, D, H, W) * scale_wind
            v_pred = pred_w[..., 1].view(B, D, H, W) * scale_wind
            w_pred = pred_w[..., 2].view(B, D, H, W) * scale_wind # w도 생성됨
            
            # Target (Normalized) -> Real Scale
            u_true = target_vol[:, 1] * scale_wind
            v_true = target_vol[:, 2] * scale_wind
            
            # --- Conc ---
            # Log-Norm -> PPM 복원: exp(x * std + mean) - 1
            c_pred_norm = pred_c.view(B, D, H, W)
            c_true_norm = target_vol[:, 0]
            
            c_pred_ppm = torch.expm1(c_pred_norm * conc_std + conc_mean)
            c_true_ppm = torch.expm1(c_true_norm * conc_std + conc_mean)
            
            # 음수 제거 (ReLU)
            c_pred_ppm = torch.clamp(c_pred_ppm, min=0.0)
            c_true_ppm = torch.clamp(c_true_ppm, min=0.0)

            # (C) Calculate Metrics (Batch 단위)
            # 1. Concentration Metrics
            c_p_np = c_pred_ppm.cpu().numpy()
            c_t_np = c_true_ppm.cpu().numpy()
            
            rmse_c, _ = calculate_metrics(c_p_np, c_t_np)
            fac2_c = calculate_fac2(c_p_np, c_t_np)
            
            results['conc_rmse'].append(rmse_c)
            results['conc_fac2'].append(fac2_c)
            
            # 2. Wind Metrics (U, V Only)
            # 벡터 크기 비교
            speed_pred = torch.sqrt(u_pred**2 + v_pred**2)
            speed_true = torch.sqrt(u_true**2 + v_true**2)
            rmse_w, _ = calculate_metrics(speed_pred.cpu().numpy(), speed_true.cpu().numpy())
            
            # 방향 비교 (Cosine Sim)
            vec_p = torch.stack([u_pred, v_pred], dim=-1)
            vec_t = torch.stack([u_true, v_true], dim=-1)
            cos_sim = F.cosine_similarity(vec_p, vec_t, dim=-1).mean().item()
            
            results['wind_rmse'].append(rmse_w)
            results['wind_cosine'].append(cos_sim)
            
            # 3. Physics Check (Divergence)
            # dx=100m, dy=100m, dz=10m (실제 스케일로 계산)
            dx, dy, dz = 100.0, 100.0, 10.0
            du_dx = (u_pred[:, :, :, 2:] - u_pred[:, :, :, :-2]) / (2 * dx)
            dv_dy = (v_pred[:, :, 2:, :] - v_pred[:, :, :-2, :]) / (2 * dy)
            dw_dz = (w_pred[:, 2:, :, :] - w_pred[:, :-2, :, :]) / (2 * dz)
            
            div = torch.abs(du_dx[:, 1:-1, 1:-1, :] + dv_dy[:, 1:-1, :, 1:-1] + dw_dz[:, :, 1:-1, 1:-1])
            results['phys_div'].append(div.mean().item())

    # (D) Final Report
    print("\n" + "="*40)
    print("   📊 Final Evaluation Results   ")
    print("="*40)
    
    print(f"[Concentration]")
    print(f"  > RMSE : {np.mean(results['conc_rmse']):.4f} ppm")
    print(f"  > FAC2 : {np.mean(results['conc_fac2'])*100:.2f} % (Target: >50%)")
    
    print(f"\n[Wind Field]")
    print(f"  > Speed RMSE    : {np.mean(results['wind_rmse']):.4f} m/s")
    print(f"  > Direction Sim : {np.mean(results['wind_cosine']):.4f} (Max 1.0)")
    
    print(f"\n[Physics Consistency]")
    print(f"  > Mean Divergence: {np.mean(results['phys_div']):.6f} (Closer to 0 is better)")
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("eval_results.csv", index=False)
    print(f"\n✅ Results saved to 'eval_results.csv'")

if __name__ == "__main__":
    evaluate()