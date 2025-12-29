import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)


from dataset.config_param import ConfigParam as Config
from dataset.dataset import get_time_split_datasets
from model import ST_TransformerDeepONet

# ==========================================
# 설정
# ==========================================
CHECKPOINT_PATH = "./train/checkpoints/model_fearless-sweep-1_best.pth"
TARGET_TIME_IDX = 50   # 확인하고 싶은 시간대
TARGET_Z_IDX = 1       # 보고 싶은 고도 층 (예: 1=10m, 4=40m)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_wind():
    print("=== Wind Field Evaluation (Vector & Magnitude) ===")
    
    # 1. 데이터 & 모델 로드
    _, val_ds, _ = get_time_split_datasets(seq_len=30, pred_step=5)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print("❌ Checkpoint not found.")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # 설정 자동 감지
    loaded_dim = 128
    if 'config' in checkpoint:
        conf = checkpoint['config']
        loaded_dim = conf.get('latent_dim', 128) if isinstance(conf, dict) else getattr(conf, 'latent_dim', 128)
    
    model = ST_TransformerDeepONet(latent_dim=loaded_dim, dropout=0.0).to(DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 2. 추론
    idx = TARGET_TIME_IDX if TARGET_TIME_IDX < len(val_ds) else 0
    ctx, met, coords, gt_w, _ = val_ds[idx] # gt_w: (N_points, 3)
    
    # 배치 추가
    ctx_b = ctx.unsqueeze(0).to(DEVICE)
    met_b = met.unsqueeze(0).to(DEVICE)
    coords_b = coords.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred_w, _ = model(ctx_b, met_b, coords_b)
        
    # 3. 데이터 복원 (Scaling 역변환)
    # dataset.py에서 wind는 단순히 scale_wind 상수로 나눴었음 (보통 10.0 등)
    w_scale = val_ds.scale_wind
    
    pred_w = pred_w.squeeze().cpu().numpy() * w_scale
    gt_w = gt_w.numpy() * w_scale
    
    # 4. 정량적 평가 (Metrics)
    # 풍속(Speed) 오차
    gt_speed = np.linalg.norm(gt_w, axis=1)
    pred_speed = np.linalg.norm(pred_w, axis=1)
    
    rmse_speed = np.sqrt(np.mean((gt_speed - pred_speed)**2))
    mae_speed = np.mean(np.abs(gt_speed - pred_speed))
    
    # 풍향(Direction) 일치도 (Cosine Similarity)
    # 0벡터 제외
    mask = gt_speed > 0.1
    dot_prod = np.sum(gt_w[mask] * pred_w[mask], axis=1)
    mag_prod = gt_speed[mask] * pred_speed[mask]
    cos_sim = dot_prod / (mag_prod + 1e-8)
    avg_cos = np.mean(cos_sim) # 1.0에 가까울수록 좋음
    
    print(f"\n[Wind Metrics @ Total Volume]")
    print(f" > Speed RMSE : {rmse_speed:.4f} m/s")
    print(f" > Speed MAE  : {mae_speed:.4f} m/s")
    print(f" > Direction Cosine Similarity : {avg_cos:.4f} (Max 1.0)")
    
    # 5. 시각화를 위한 Reshape (NZ, NY, NX, 3)
    # dataset.py의 meshgrid 생성 순서 주의 (보통 Z, Y, X 순)
    pred_3d = pred_w.reshape(Config.NZ, Config.NY, Config.NX, 3)
    gt_3d = gt_w.reshape(Config.NZ, Config.NY, Config.NX, 3)
    
    # 특정 고도(Target Z) 슬라이싱
    z_idx = TARGET_Z_IDX
    pred_layer = pred_3d[z_idx] # (NY, NX, 3)
    gt_layer = gt_3d[z_idx]     # (NY, NX, 3)
    
    # 6. Plotting (Quiver & Heatmap)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 격자 생성 (시각화용)
    X, Y = np.meshgrid(np.arange(Config.NX), np.arange(Config.NY))
    
    # Downsampling for Quiver (너무 빽빽하면 안 보임)
    step = 3
    X_q, Y_q = X[::step, ::step], Y[::step, ::step]
    
    # GT Wind
    U_gt, V_gt = gt_layer[::step, ::step, 0], gt_layer[::step, ::step, 1]
    Speed_gt = np.linalg.norm(gt_layer[:, :, :2], axis=2)
    
    im1 = axes[0, 0].imshow(Speed_gt, origin='lower', cmap='viridis')
    axes[0, 0].quiver(X_q, Y_q, U_gt, V_gt, color='white', scale=50, width=0.005)
    axes[0, 0].set_title(f"GT Wind Field (Z={z_idx*Config.DZ}m)")
    fig.colorbar(im1, ax=axes[0, 0], label='Speed (m/s)')
    
    # Pred Wind
    U_pred, V_pred = pred_layer[::step, ::step, 0], pred_layer[::step, ::step, 1]
    Speed_pred = np.linalg.norm(pred_layer[:, :, :2], axis=2)
    
    im2 = axes[0, 1].imshow(Speed_pred, origin='lower', cmap='viridis', vmin=Speed_gt.min(), vmax=Speed_gt.max())
    axes[0, 1].quiver(X_q, Y_q, U_pred, V_pred, color='white', scale=50, width=0.005)
    axes[0, 1].set_title(f"Pred Wind Field (Z={z_idx*Config.DZ}m)")
    fig.colorbar(im2, ax=axes[0, 1], label='Speed (m/s)')
    
    # Error Map (Speed Diff)
    diff_speed = np.abs(Speed_gt - Speed_pred)
    im3 = axes[1, 0].imshow(diff_speed, origin='lower', cmap='inferno')
    axes[1, 0].set_title("Wind Speed Error (|GT - Pred|)")
    fig.colorbar(im3, ax=axes[1, 0], label='Error (m/s)')
    
    # Error Map (Direction Diff - Degree)
    # 코사인 유사도를 각도로 변환
    dot = (gt_layer[:,:,0]*pred_layer[:,:,0] + gt_layer[:,:,1]*pred_layer[:,:,1])
    mag = Speed_gt * Speed_pred
    cos_val = np.clip(dot / (mag + 1e-8), -1.0, 1.0)
    deg_diff = np.degrees(np.arccos(cos_val))
    # 바람이 거의 없는 곳(0.1m/s 이하)은 각도 의미 없으므로 마스킹
    mask_calm = Speed_gt < 0.1
    deg_diff[mask_calm] = 0
    
    im4 = axes[1, 1].imshow(deg_diff, origin='lower', cmap='Reds', vmin=0, vmax=180)
    axes[1, 1].set_title("Wind Direction Error (Degree)")
    fig.colorbar(im4, ax=axes[1, 1], label='Degree Diff')
    
    plt.tight_layout()
    plt.savefig("eval_wind.png")
    print("✅ Saved wind evaluation to 'eval_wind.png'")
    plt.show()

if __name__ == "__main__":
    evaluate_wind()