import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3D Plot팅용

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from dataset.config_param import ConfigParam as Config
from dataset.dataset import get_time_split_datasets
from model import ST_TransformerDeepONet

# ==========================================
# 설정
# ==========================================
CHECKPOINT_PATH = "./train/checkpoints/model_soft-sweep-1_best.pth"
TARGET_TIME_IDX = 50   # 보고 싶은 시간대
THRESHOLD_CONC = 10.0  # 시각화할 최소 농도 (이 값 이상만 3D로 찍음)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def viz_3d():
    print("=== 3D Structure Visualization ===")
    
    # 1. 데이터 로드
    # (메모리 절약을 위해 validation set만 빠르게 로드)
    _, val_ds, _ = get_time_split_datasets(seq_len=30, pred_step=5)
    
    # 2. 체크포인트 로드 및 모델 설정 자동 감지
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint file not found: {CHECKPOINT_PATH}")
        return

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        # Latent Dim 자동 감지
        loaded_dim = 128 # 기본값
        if 'config' in checkpoint:
            conf = checkpoint['config']
            if isinstance(conf, dict):
                loaded_dim = conf.get('latent_dim', 128)
            else:
                loaded_dim = getattr(conf, 'latent_dim', 128)
        
        print(f"ℹ️ Detected latent_dim from checkpoint: {loaded_dim}")
        
        # 모델 초기화 (감지된 차원 사용)
        model = ST_TransformerDeepONet(latent_dim=loaded_dim, dropout=0.0).to(DEVICE)
        
        # 가중치 로드 (키 에러 방지)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print("✅ Model loaded successfully.")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model.eval()
    
    # 3. 특정 샘플 추론
    # 인덱스 에러 방지
    if TARGET_TIME_IDX >= len(val_ds):
        print(f"⚠️ Index {TARGET_TIME_IDX} is out of bounds. Using 0 instead.")
        idx = 0
    else:
        idx = TARGET_TIME_IDX
        
    ctx, met, coords, _, _ = val_ds[idx]
    
    # Batch 차원 추가 (1, ...) -> 메모리 사용량 최소화
    ctx = ctx.unsqueeze(0).to(DEVICE)
    met = met.unsqueeze(0).to(DEVICE)
    coords = coords.unsqueeze(0).to(DEVICE) # (1, N_points, 4)
    
    with torch.no_grad():
        _, pred_c = model(ctx, met, coords)
    
    # 4. 데이터 복원 및 구조화
    pred_c = pred_c.squeeze().cpu().numpy()
    
    c_mean = val_ds.conc_mean
    c_std = val_ds.conc_std
    pred_phys = np.maximum(np.expm1(pred_c * c_std + c_mean), 0)
    
    # (NZ, NY, NX) 형태로 Reshape
    conc_3d = pred_phys.reshape(Config.NZ, Config.NY, Config.NX)
    
    # 5. 시각화 (3D View + Side Cut)
    fig = plt.figure(figsize=(18, 8))
    
    # [왼쪽] 3D Scatter Plot (고농도 지점만)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Threshold 이상의 포인트만 추출
    z_idx, y_idx, x_idx = np.where(conc_3d > THRESHOLD_CONC)
    vals = conc_3d[conc_3d > THRESHOLD_CONC]
    
    # 인덱스 -> 실제 미터 좌표 변환
    xx = x_idx * Config.DX
    yy = y_idx * Config.DY
    zz = z_idx * Config.DZ
    
    if len(vals) == 0:
        print(f"⚠️ Warning: No concentration above {THRESHOLD_CONC} ppm found.")
        # 빈 3D 플롯이라도 범위는 설정
    else:
        # 3D Scatter
        img = ax1.scatter(xx, yy, zz, c=vals, cmap='jet', s=20, alpha=0.6, edgecolors='none')
        
        # 바닥에 지형 등고선 깔아주기
        X, Y = np.meshgrid(np.arange(0, Config.NX*Config.DX, Config.DX),
                           np.arange(0, Config.NY*Config.DY, Config.DY))
        # 지형 데이터 복원
        terrain = ctx[0, 0].cpu().numpy() * Config.MAX_Z 
        ax1.plot_surface(X, Y, terrain, color='gray', alpha=0.3)
        
        plt.colorbar(img, ax=ax1, label='Concentration (ppm)', shrink=0.7)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Height (m)')
    ax1.set_title(f'3D Plume Structure (Conc > {THRESHOLD_CONC} ppm)')
    ax1.set_zlim(0, Config.MAX_Z)

    # [오른쪽] XZ Side View (농도가 가장 짙은 Y 지점 절단)
    ax2 = fig.add_subplot(1, 2, 2)
    
    # 농도 합이 가장 높은 Y 인덱스 찾기 (Plume 중심)
    sum_conc_y = np.sum(conc_3d, axis=(0, 2)) # Z, X 축 합산 -> Y축만 남음
    target_y = np.argmax(sum_conc_y)
    
    # 단면 잘라내기 (Z, X)
    slice_xz = conc_3d[:, target_y, :]
    
    # 시각화
    im2 = ax2.imshow(slice_xz, origin='lower', cmap='jet', aspect='auto',
                     extent=[0, Config.NX*Config.DX, 0, Config.NZ*Config.DZ])
    
    # 해당 단면의 지형 표시
    terrain_prof = terrain[target_y, :]
    ax2.fill_between(np.arange(0, Config.NX*Config.DX, Config.DX), 
                     0, terrain_prof, color='black')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Height (m)')
    ax2.set_title(f'Vertical Cross-Section at Y={target_y*Config.DY}m')
    plt.colorbar(im2, ax=ax2, label='Concentration (ppm)')
    
    save_path = "vis_3d_structure.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✅ Saved 3D visualization to '{save_path}'")
    plt.show()

if __name__ == "__main__":
    viz_3d()