import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.config_param import ConfigParam as Config
from dataset.dataset import get_time_split_datasets
from model import ST_TransformerDeepONet

# ==========================================
# 설정
# ==========================================
CHECKPOINT_PATH = "./train/checkpoints/model_confused-sweep-1_best.pth"
MIN_CONC_THRESHOLD = 100.0  # 이 값 이상인 농도가 있는 샘플만 찾음 (ppm)
VIS_W_SCALE = 15.0         # Side View에서 수직풍(W) 화살표 크기 증폭 배수
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_best_plume_sample(dataset):
    """
    단순 최대 농도가 아니라, '연기가 넓고 길게 퍼진(High Coverage)' 샘플을 찾습니다.
    바람에 의한 확산 패턴이 가장 잘 보이는 데이터를 골라냅니다.
    """
    print("🔍 Searching for the best plume sample (High Concentration + Wide Spread)...")
    
    best_idx = 0
    best_score = -1.0
    
    # 전체 데이터셋 순회 (시간이 걸릴 수 있으므로 10%만 샘플링하거나, 전체를 돌림)
    # 여기서는 전체를 빠르게 훑는 로직
    for i in range(len(dataset)):
        _, _, _, _, c_norm = dataset[i] # c_norm: (N_points, 1)
        
        # 1. 텐서 -> 넘파이 변환 (CPU 연산)
        c_val = c_norm.numpy().flatten()
        
        # 2. 복원 없이 Z-score 상태에서 빠르게 판단 (속도 최적화)
        # Z-score > 1.0 이면 대략 상위 16% (유의미한 농도)
        # Z-score > 3.0 이면 대략 상위 0.1% (고농도 피크)
        
        # 조건 A: 고농도 피크가 존재해야 함 (뚜렷함)
        max_val = c_val.max()
        if max_val < 3.0: # 약 평균+3표준편차 미만이면 패스 (너무 연함)
            continue
            
        # 조건 B: 유의미한 농도(Z > 0.5)를 가진 격자점의 개수 (넓이)
        spread_count = np.sum(c_val > 0.5)
        
        # 점수 산정: 피크 높이보다 '얼마나 넓게 퍼졌나'에 가중치
        # Score = (확산 면적) * (최대 농도 로그) 
        # -> 면적이 넓을수록 점수가 크게 오름
        score = spread_count * np.log1p(max_val)
        
        if score > best_score:
            best_score = score
            best_idx = i
            
            # 진행 상황 모니터링 (옵션)
            # print(f"  -> New Best Candidate: Idx {i} (Spread: {spread_count} pts, Max Z: {max_val:.2f})")

    print(f"✅ Best Sample Found: Index {best_idx} (Score: {best_score:.2f})")
    return best_idx

def find_high_concentration_sample(dataset):
    """
    Validation Set을 탐색하여 오염 농도가 기준치(MIN_CONC_THRESHOLD) 이상인 샘플을 찾습니다.
    """
    print(f"🔍 Searching for a sample with Max Conc > {MIN_CONC_THRESHOLD} ppm...")
    
    # 순차 탐색
    for i in range(len(dataset)):
        # dataset[i] -> (ctx, met, coords, wind_gt, conc_gt)
        # conc_gt는 (N_points, 1) 형태의 Tensor입니다.
        _, _, _, _, c_norm = dataset[i]
        
        # [수정] 텐서에서 최대값을 먼저 찾고(.max), 스칼라로 변환(.item)
        c_max_norm = c_norm.max().item()
        
        # 복원 (Z-score -> Log -> Exp)
        # dataset 객체의 통계치 사용
        c_mean = dataset.conc_mean
        c_std = dataset.conc_std
        
        # 물리적 농도(ppm)로 변환
        c_phys_max = np.expm1(c_max_norm * c_std + c_mean)
        
        if c_phys_max > MIN_CONC_THRESHOLD:
            print(f"✅ Found Sample Index: {i} (Max Conc: {c_phys_max:.2f} ppm)")
            return i
            
    print("⚠️ Could not find any sample exceeding threshold. Using Index 0.")
    return 0

def visualize_comparison():
    print("=== GT vs Prediction Visual Comparison (With Wind) ===")
    
    # 1. 데이터 로드
    _, val_ds, _ = get_time_split_datasets(seq_len=30, pred_step=5)
    
    # 2. 유의미한 샘플 찾기
    target_idx = find_best_plume_sample(val_ds)
    
    # 3. 모델 로드
    if not os.path.exists(CHECKPOINT_PATH):
        print("❌ Checkpoint not found.")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
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

    # 4. 데이터 준비
    ctx, met, coords, gt_w, gt_c = val_ds[target_idx]
    
    # 배치 차원 추가
    ctx_b = ctx.unsqueeze(0).to(DEVICE)
    met_b = met.unsqueeze(0).to(DEVICE)
    coords_b = coords.unsqueeze(0).to(DEVICE)
    
    # 추론 (Wind & Conc 동시 예측)
    with torch.no_grad():
        pred_w, pred_c = model(ctx_b, met_b, coords_b)
        
    # 오염원 위치 파악 (Top View 표시용)
    source_map = ctx[1].numpy()
    sy, sx = np.where(source_map > 0)

    # 5. 데이터 복원 (Denormalization)
    
    # [Concentration]
    pred_c_raw = pred_c.squeeze().cpu().numpy()
    gt_c_raw = gt_c.numpy()
    
    c_mean, c_std = val_ds.conc_mean, val_ds.conc_std
    pred_c_phys = np.maximum(np.expm1(pred_c_raw * c_std + c_mean), 0)
    gt_c_phys = np.maximum(np.expm1(gt_c_raw * c_std + c_mean), 0)
    
    # [Wind]
    # 모델 출력과 데이터셋 GT는 모두 Normalized (-1 ~ 1) 상태임
    # 실제 m/s로 보려면 scale_wind를 곱해야 함
    w_scale = val_ds.scale_wind
    
    pred_w_raw = pred_w.squeeze().cpu().numpy() # (N_points, 3)
    gt_w_raw = gt_w.numpy()                     # (N_points, 3)
    
    pred_w_phys = pred_w_raw * w_scale
    gt_w_phys = gt_w_raw * w_scale

    # 6. 3D Reshape (NZ, NY, NX)
    # (주의: dataset meshgrid 순서에 따름, 보통 z, y, x)
    def to_3d(arr_flat, channels=1):
        if channels == 1:
            return arr_flat.reshape(Config.NZ, Config.NY, Config.NX)
        else:
            return arr_flat.reshape(Config.NZ, Config.NY, Config.NX, channels)

    gt_c_3d = to_3d(gt_c_phys)
    pred_c_3d = to_3d(pred_c_phys)
    
    gt_w_3d = to_3d(gt_w_phys, channels=3)     # (NZ, NY, NX, 3) -> U, V, W
    pred_w_3d = to_3d(pred_w_phys, channels=3) # (NZ, NY, NX, 3)
    
    # 7. 슬라이싱 (가장 진한 농도 지점 기준)
    z_max, y_max, x_max = np.unravel_index(np.argmax(gt_c_3d), gt_c_3d.shape)
    
    print(f"🔍 Slicing at Max Concentration Point:")
    print(f"   > Z={z_max} ({z_max*Config.DZ}m)")
    print(f"   > Y={y_max} ({y_max*Config.DY}m)")
    
    # 지형 높이 (Side View 마스킹용)
    real_terrain = ctx[0].numpy() * Config.MAX_Z # (NY, NX)

    # 8. Plotting
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
    
    # 공통 Colorbar 설정
    c_max = max(gt_c_3d.max(), pred_c_3d.max())
    c_min = 0
    
    # Quiver Downsampling (너무 빽빽하지 않게)
    step = 3 
    
    # ==========================
    # Row 1: Ground Truth (GT)
    # ==========================
    # [1-1] Top View (XY)
    im1 = axes[0, 0].imshow(gt_c_3d[z_max, :, :], origin='lower', cmap='jet', vmin=c_min, vmax=c_max)
    axes[0, 0].set_title(f"GT: Top View (Conc + Wind) @ {z_max*Config.DZ}m")
    
    # GT Wind Overlay (U, V)
    U_gt = gt_w_3d[z_max, ::step, ::step, 0]
    V_gt = gt_w_3d[z_max, ::step, ::step, 1]
    X_q, Y_q = np.meshgrid(np.arange(0, Config.NX, step), np.arange(0, Config.NY, step))
    axes[0, 0].quiver(X_q, Y_q, U_gt, V_gt, color='white', scale=50, width=0.005, alpha=0.8)
    
    # Source 표시
    axes[0, 0].scatter(sx, sy, c='red', marker='*', s=200, edgecolors='black', label='Source')
    axes[0, 0].legend(loc='upper right')

    # [1-2] Side View (XZ)
    im2 = axes[0, 1].imshow(gt_c_3d[:, y_max, :], origin='lower', cmap='jet', aspect='auto', vmin=c_min, vmax=c_max)
    axes[0, 1].set_title(f"GT: Side View (Conc + Wind) @ Y={y_max*Config.DY}m")
    
    # 지형 마스킹 및 표시
    h_prof = real_terrain[y_max, :]
    axes[0, 1].plot(h_prof, color='white', linewidth=2)
    axes[0, 1].fill_between(np.arange(Config.NX), 0, h_prof, color='black', alpha=0.6)

    # GT Wind Overlay (U, W)
    # W는 작으므로 시각화를 위해 VIS_W_SCALE 배 증폭
    U_gt_side = gt_w_3d[:, y_max, :, 0]
    W_gt_side = gt_w_3d[:, y_max, :, 2] * VIS_W_SCALE
    
    # 지형 아래 바람은 지움 (시각적 깔끔함)
    XX, ZZ = np.meshgrid(np.arange(Config.NX), np.arange(Config.NZ)) # Grid for checking height
    mask_h = ZZ < h_prof[XX] # (NZ, NX) vs (NX,) broadcast ?? No, broadcasting issue.
    # Meshgrid shape matches array shape directly?
    # XX shape: (NZ, NX), h_prof shape: (NX,). correct.
    
    U_gt_side[ZZ < h_prof] = 0
    W_gt_side[ZZ < h_prof] = 0
    
    axes[0, 1].quiver(XX[::step, ::step], ZZ[::step, ::step], 
                      U_gt_side[::step, ::step], W_gt_side[::step, ::step], 
                      color='white', scale=80, width=0.005, alpha=0.8)

    # [1-3] Distribution
    axes[0, 2].hist(gt_c_phys.flatten(), bins=50, log=True, color='blue', alpha=0.7)
    axes[0, 2].set_title("GT Conc Distribution")

    # ==========================
    # Row 2: Prediction (Pred)
    # ==========================
    # [2-1] Top View
    im3 = axes[1, 0].imshow(pred_c_3d[z_max, :, :], origin='lower', cmap='jet', vmin=c_min, vmax=c_max)
    axes[1, 0].set_title(f"Pred: Top View")
    
    # Pred Wind Overlay (U, V)
    U_pred = pred_w_3d[z_max, ::step, ::step, 0]
    V_pred = pred_w_3d[z_max, ::step, ::step, 1]
    axes[1, 0].quiver(X_q, Y_q, U_pred, V_pred, color='white', scale=50, width=0.005, alpha=0.8)
    axes[1, 0].scatter(sx, sy, c='red', marker='*', s=200, edgecolors='black') # 오염원 위치

    # [2-2] Side View
    im4 = axes[1, 1].imshow(pred_c_3d[:, y_max, :], origin='lower', cmap='jet', aspect='auto', vmin=c_min, vmax=c_max)
    axes[1, 1].set_title(f"Pred: Side View")
    
    axes[1, 1].plot(h_prof, color='white', linewidth=2)
    axes[1, 1].fill_between(np.arange(Config.NX), 0, h_prof, color='black', alpha=0.6)
    
    # Pred Wind Overlay (U, W)
    U_pred_side = pred_w_3d[:, y_max, :, 0]
    W_pred_side = pred_w_3d[:, y_max, :, 2] * VIS_W_SCALE
    
    U_pred_side[ZZ < h_prof] = 0
    W_pred_side[ZZ < h_prof] = 0
    
    axes[1, 1].quiver(XX[::step, ::step], ZZ[::step, ::step], 
                      U_pred_side[::step, ::step], W_pred_side[::step, ::step], 
                      color='white', scale=80, width=0.005, alpha=0.8)

    # [2-3] Distribution
    axes[1, 2].hist(pred_c_phys.flatten(), bins=50, log=True, color='red', alpha=0.7)
    axes[1, 2].set_title("Pred Conc Distribution")

    # ==========================
    # Row 3: Error (Diff)
    # ==========================
    diff_c_3d = np.abs(gt_c_3d - pred_c_3d)
    diff_max = diff_c_3d.max()
    
    # [3-1] Top Error
    im5 = axes[2, 0].imshow(diff_c_3d[z_max, :, :], origin='lower', cmap='inferno', vmin=0, vmax=diff_max)
    axes[2, 0].set_title(f"Error: Top View (|GT-Pred|)")
    axes[2, 0].scatter(sx, sy, c='cyan', marker='x', s=100)

    # [3-2] Side Error
    im6 = axes[2, 1].imshow(diff_c_3d[:, y_max, :], origin='lower', cmap='inferno', aspect='auto', vmin=0, vmax=diff_max)
    axes[2, 1].set_title(f"Error: Side View")
    
    # Colorbars
    cbar = fig.colorbar(im1, ax=axes[0:2, :], location='right', shrink=0.6)
    cbar.set_label('Concentration (ppm)')
    
    cbar_err = fig.colorbar(im5, ax=axes[2, :], location='right', shrink=0.6)
    cbar_err.set_label('Absolute Error (ppm)')
    
    # 텍스트 정보 추가
    info_text = (
        f"Time Index: {target_idx}\n"
        f"Max GT Conc: {gt_c_phys.max():.2f} ppm\n"
        f"Max Pred Conc: {pred_c_phys.max():.2f} ppm\n"
        f"Wind Scale: {w_scale:.2f} m/s"
    )
    axes[2, 2].text(0.1, 0.5, info_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    axes[2, 2].axis('off')
    
    plt.savefig("comparison_gt_pred_wind.png")
    print("✅ Saved plot to 'comparison_gt_pred_wind.png'")
    plt.show()

if __name__ == "__main__":
    visualize_comparison()