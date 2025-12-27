import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config_param import ConfigParam as Config

# [핵심] 공통 물리 함수 사용
from dataset.physics_utils import calc_wind_profile_power_law

# ==========================================
# 1. 설정
# ==========================================
TARGET_TIME_IDX = 10  # 확인하고 싶은 시간대 인덱스
Z_LAYER_IDX = 5       # 보고 싶은 고도 층 (0~20)
VIS_W_SCALE = 10.0    # 수직풍(W) 시각화 증폭 배수

def run_verification():
    print("=== Data Verification & Visualization (Denormalized) ===")
    
    # ---------------------------------------------------------
    # 1. 데이터 로드
    # ---------------------------------------------------------
    p_dir = Config.PROCESSED_DIR
    try:
        # A. Maps (Terrain, Source)
        d_maps = np.load(os.path.join(p_dir, Config.SAVE_MAPS))
        terrain_norm = d_maps['terrain'] # 0~1
        source_q_log = d_maps['source_q'] # Log scale
        
        # 실제 지형 높이 복원을 위한 max 값
        t_max = float(d_maps.get('terrain_max', 1.0))
        
        # B. Meteorology (Raw)
        d_met = np.load(os.path.join(p_dir, Config.SAVE_MET))
        met_data = d_met['met'] # (N, 4) -> u, v, L, wd
        
        # C. Labels (Conc) - Log + Z-score 정규화된 상태
        d_lbl = np.load(os.path.join(p_dir, Config.SAVE_LBL))
        conc_norm = d_lbl['conc']
        
        # 복원을 위한 통계량 로드
        c_mean = float(d_lbl['mean_stat'])
        c_std = float(d_lbl['std_stat'])
        
        print("✅ All data files loaded successfully.")

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return

    # ---------------------------------------------------------
    # 2. 데이터 복원 (Denormalization)
    # ---------------------------------------------------------
    print("   -> Restoring physical values...")
    
    # 지형 복원 (0~1 -> Meter)
    real_terrain = terrain_norm * t_max
    
    # 농도 복원 (Z-score -> Log -> Real)
    # 1) Z-score 역변환: x = z * std + mean
    conc_log = conc_norm[TARGET_TIME_IDX] * c_std + c_mean
    # 2) Log 역변환: C = exp(log_C) - 1
    conc_real = np.expm1(conc_log) # (NY, NX, NZ)
    
    # 음수 노이즈 제거 (물리적으로 0 이상이어야 함)
    conc_real = np.maximum(conc_real, 0)
    
    # 기울기 계산 (물리 좌표계 기준)
    grad_y, grad_x = np.gradient(real_terrain, Config.DY, Config.DX)

    # 타겟 기상 정보
    uref, vref, L, wd = met_data[TARGET_TIME_IDX]
    print(f"\n[Met Info] Index: {TARGET_TIME_IDX}")
    print(f"   > Wind: U={uref:.2f}, V={vref:.2f} (WD: {wd:.1f}°)")
    print(f"   > Stability (L): {L:.2f}")
    print(f"   > Max Conc (Real): {np.max(conc_real):.4f}")

    # ---------------------------------------------------------
    # 3. 시각화 (기존 스타일 유지)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(16, 9))
    plt.suptitle(f"Data Verification (Time: {TARGET_TIME_IDX})", fontsize=16)

    # === [Left] XY View (Top) ===
    ax1 = fig.add_subplot(1, 2, 1)
    
    # 실제 높이 (미터)
    z_height_m = Z_LAYER_IDX * Config.DZ
    
    # 배경: 지형 등고선
    ax1.contour(real_terrain, levels=10, colors='k', alpha=0.3)
    
    # 농도 맵 (복원된 값 사용)
    im1 = ax1.imshow(conc_real[:, :, Z_LAYER_IDX], origin='lower', cmap='jet', alpha=0.8,
                     extent=[0, Config.NX, 0, Config.NY], 
                     vmax=np.percentile(conc_real, 99.5)) # 이상치 제외하고 색상 범위 설정
    
    # 오염원 표시
    sy, sx = np.where(source_q_log > 0)
    ax1.scatter(sx, sy, c='red', marker='*', s=100, edgecolors='k', label='Source')

    # 바람장 계산 (physics_utils 활용)
    # z_height_m(스칼라)와 grad(2D 배열)를 전달 -> Broadcasting
    wind_vecs = calc_wind_profile_power_law(
        uref, vref, L, 
        z_points=np.array([z_height_m]), 
        slopes=(grad_x, grad_y)
    )
    # 결과 형상 주의: (45, 45, 3)으로 나올 것임 (numpy broadcasting 규칙에 따라)
    # 만약 함수가 (N, 3)을 리턴한다면 reshape 필요. 
    # physics_utils.py가 array input을 받으면 shape를 유지하도록 작성됨.
    
    if wind_vecs.ndim == 2: # (1, 3) Scalar case
         # Broadcasting이 안 된 경우 수동 확장 (안전장치)
         U_2d = np.full((Config.NY, Config.NX), wind_vecs[0,0])
         V_2d = np.full((Config.NY, Config.NX), wind_vecs[0,1])
    else:
         U_2d = wind_vecs[:, :, 0] # (NY, NX)
         V_2d = wind_vecs[:, :, 1]
    
    # Quiver Plot
    step = 3
    ax1.quiver(np.arange(0, Config.NX, step), np.arange(0, Config.NY, step), 
               U_2d[::step, ::step], V_2d[::step, ::step], 
               color='white', scale=50, width=0.005)
    
    ax1.set_title(f"Top View (XY) @ {z_height_m}m", fontsize=14)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1, label='Concentration (Real Value)')

    # === [Right] Side View (XZ) ===
    # 농도가 가장 높은 Y 지점을 잘라서 봄
    max_pos = np.unravel_index(np.argmax(conc_real), conc_real.shape)
    target_y = max_pos[0] # (Y, X, Z) 순서 주의 (numpy vs torch shape check)
    # dataset.py 로드 시 (Time, Y, X, Z) 였으므로 -> [Time][Y][X][Z]
    # 위에서 conc_real = conc_norm[TARGET_TIME_IDX] 했으므로 -> (Y, X, Z)
    
    ax2 = fig.add_subplot(1, 2, 2)
    
    # 단면 농도 (X, Z) -> Transpose -> (Z, X) for plotting
    slice_conc = conc_real[target_y, :, :].T 
    
    # 좌표 그리드 생성
    xx = np.arange(Config.NX)
    zz = np.linspace(0, Config.NZ * Config.DZ, Config.NZ)
    XX, ZZ = np.meshgrid(xx, zz)
    
    ax2.contourf(XX, ZZ, slice_conc, levels=50, cmap='jet')
    
    # 지형 단면
    h_prof = real_terrain[target_y, :]
    ax2.fill_between(xx, 0, h_prof, color='black', alpha=0.7)
    
    # 단면 바람장 계산
    # 1. 해당 Y 라인의 기울기 (NX,)
    gx_slice = grad_x[target_y, :]
    gy_slice = grad_y[target_y, :]
    
    # 2. 전체 단면 Grid에 대해 확장 (NZ, NX)
    slope_x_grid = np.tile(gx_slice, (Config.NZ, 1))
    slope_y_grid = np.tile(gy_slice, (Config.NZ, 1))
    
    # 3. 평탄화하여 함수 호출
    flat_z = ZZ.flatten() # (N_points,)
    flat_sx = slope_x_grid.flatten()
    flat_sy = slope_y_grid.flatten()
    
    vec_flat = calc_wind_profile_power_law(
        uref, vref, L, 
        z_points=flat_z, 
        slopes=(flat_sx, flat_sy)
    )
    
    # 4. Reshape
    U_grid = vec_flat[:, 0].reshape(Config.NZ, Config.NX)
    W_grid = vec_flat[:, 2].reshape(Config.NZ, Config.NX)
    
    # 시각화용 마스킹 & W 증폭
    mask = ZZ < h_prof # 지형 아래
    U_grid[mask] = 0
    W_grid[mask] = 0
    W_vis = W_grid * VIS_W_SCALE
    
    # Quiver
    q_step_x = 2
    q_step_z = 2
    ax2.quiver(XX[::q_step_z, ::q_step_x], ZZ[::q_step_z, ::q_step_x], 
               U_grid[::q_step_z, ::q_step_x], W_vis[::q_step_z, ::q_step_x],
               color='white', scale=100, width=0.005)
    
    ax2.set_title(f"Side View (XZ) @ Y={target_y}\n(Vertical Wind x{VIS_W_SCALE})", fontsize=14)
    ax2.set_ylim(0, Config.NZ * Config.DZ)
    
    plt.tight_layout()
    save_img = "data_verification.png"
    plt.savefig(save_img, dpi=300)
    print(f"\n✅ Verification image saved to: {save_img}")
    plt.show()

if __name__ == "__main__":
    run_verification()