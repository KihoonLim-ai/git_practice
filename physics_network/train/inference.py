import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# [경로 유지] dataset.dataset
from dataset.dataset import AermodDataset
# from model import RecurrentDeepONet
from model.model import ST_TransformerDeepONet
from dataset.config_param import ConfigParam as Config

# ==========================================
# [설정] WandB에서 확인한 Best Model 정보 입력
# ==========================================
BEST_RUN_ID = "kari_sweep_20251224_5"  # 예: "KARI_Sweep_10"
BEST_SEQ_LEN = 6                        
CHECKPOINT_DIR = "./checkpoints"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def denormalize_wind(val, scale):
    """바람 정규화 복원 (-1~1 -> m/s)"""
    return val * scale

def plot_comparison(terrain, gt_wind, gt_conc, pred_wind, pred_conc, z_idx=4):
    """
    [수정] 맵 전체에 바람 화살표가 올바르게 배치되도록 좌표 그리드 적용
    """
    NX, NY, NZ = Config.NX, Config.NY, Config.NZ
    
    # 1. 1D Array -> 3D Grid 변환 (NY, NX, NZ, C)
    gt_w_3d = gt_wind.reshape(NY, NX, NZ, 3)
    gt_c_3d = gt_conc.reshape(NY, NX, NZ, 1)
    pd_w_3d = pred_wind.reshape(NY, NX, NZ, 3)
    pd_c_3d = pred_conc.reshape(NY, NX, NZ, 1)
    
    # 2. 특정 고도 슬라이싱
    gt_u, gt_v = gt_w_3d[:, :, z_idx, 0], gt_w_3d[:, :, z_idx, 1]
    pd_u, pd_v = pd_w_3d[:, :, z_idx, 0], pd_w_3d[:, :, z_idx, 1]
    
    gt_speed = np.sqrt(gt_u**2 + gt_v**2)
    pd_speed = np.sqrt(pd_u**2 + pd_v**2)
    
    gt_c = gt_c_3d[:, :, z_idx, 0]
    pd_c = pd_c_3d[:, :, z_idx, 0]
    
    # 3. [핵심 수정] 화살표가 위치할 X, Y 좌표 그리드 생성
    x = np.arange(NX)
    y = np.arange(NY)
    X, Y = np.meshgrid(x, y)
    
    # 화살표 밀도 조절 (3칸 간격)
    skip = (slice(None, None, 3), slice(None, None, 3))
    
    # --- 시각화 시작 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 컬러바 범위 설정
    c_max = max(gt_c.max(), pd_c.max())
    if c_max < 1e-3: c_max = 1.0
    w_max = max(gt_speed.max(), pd_speed.max())

    # [Row 1] Concentration Comparison
    ax1 = axes[0, 0]
    ax1.contour(terrain, levels=10, colors='k', alpha=0.3)
    im1 = ax1.imshow(gt_c, origin='lower', cmap='jet', vmin=0, vmax=c_max)
    ax1.set_title("[GT] Concentration (Label)", fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='ug/m3')

    ax2 = axes[0, 1]
    ax2.contour(terrain, levels=10, colors='k', alpha=0.3)
    im2 = ax2.imshow(pd_c, origin='lower', cmap='jet', vmin=0, vmax=c_max)
    ax2.set_title("[Pred] Concentration (Log-Scaled DeepONet)", fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax2)

    # [Row 2] Wind Field Comparison (좌표 지정 수정됨)
    # 2-1. Ground Truth Wind
    ax3 = axes[1, 0]
    im3 = ax3.imshow(gt_speed, origin='lower', cmap='viridis', vmin=0, vmax=w_max)
    # X[skip], Y[skip]을 통해 화살표 위치를 명시적으로 지정
    ax3.quiver(X[skip], Y[skip], gt_u[skip], gt_v[skip], 
               color='white', scale=30, width=0.005)
    ax3.set_title("[GT] Wind Speed & Direction", fontsize=14)
    plt.colorbar(im3, ax=ax3, label='m/s')

    # 2-2. Predicted Wind
    ax4 = axes[1, 1]
    im4 = ax4.imshow(pd_speed, origin='lower', cmap='viridis', vmin=0, vmax=w_max)
    # 예측 데이터도 동일하게 위치 지정
    ax4.quiver(X[skip], Y[skip], pd_u[skip], pd_v[skip], 
               color='white', scale=30, width=0.005)
    ax4.set_title("[Pred] Wind Speed & Direction", fontsize=14)
    plt.colorbar(im4, ax=ax4)
    
    plt.suptitle(f"Inference Analysis (Log-scaling) - Best Run: {BEST_RUN_ID}", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"inference_comparison_{BEST_RUN_ID}.png", dpi=300)
    plt.show()

def main():
    print(f"=== Inference with Best Sweep Model ({BEST_RUN_ID}) ===")
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model_{BEST_RUN_ID}.pth")
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint not found: {ckpt_path}")
        return

    # 1. 데이터셋 로드 (로그 변환된 dataset.py 사용)
    ds = AermodDataset(mode='test', seq_len=BEST_SEQ_LEN)
    model = ST_TransformerDeepONet().to(DEVICE)
    
    # 2. 모델 로드
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # 3. 샘플 추론 (랜덤 샘플링)
    idx = np.random.randint(0, len(ds))
    data = ds[idx]
    
    ctx_map = data[0].unsqueeze(0).to(DEVICE)
    met_seq = data[1].unsqueeze(0).to(DEVICE)
    coords  = data[2].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred_wind_norm, pred_conc_norm = model(ctx_map, met_seq, coords)
        
        # 4. 역정규화 (Denormalization)
        # 바람: Max-Abs 역변환
        pred_wind_real = pred_wind_norm.cpu().numpy().squeeze(0) * ds.scale_wind
        gt_wind_real = data[3].numpy() * ds.scale_wind
        
        # 농도: Log 역변환 (expm1)
        pred_conc_real = ds.denormalize_conc(pred_conc_norm.cpu().numpy().squeeze(0))
        gt_conc_real = ds.denormalize_conc(data[4].numpy())

    # 5. 수치 분석 출력
    print(f"\n[Comparison Summary for Sample {idx}]")
    print(f"   > Max Conc (Truth): {gt_conc_real.max():.2f} ug/m3")
    print(f"   > Max Conc (Pred) : {pred_conc_real.max():.2f} ug/m3")
    print(f"   > Max Wind Speed (Truth): {np.max(np.sqrt(gt_wind_real[:,0]**2 + gt_wind_real[:,1]**2)):.2f} m/s")
    
    # 6. 시각화 실행
    plot_comparison(ds.terrain, gt_wind_real, gt_conc_real, pred_wind_real, pred_conc_real)

if __name__ == "__main__":
    main()