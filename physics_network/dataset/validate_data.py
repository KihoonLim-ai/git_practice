import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dataset.config_param import ConfigParam as Config

# [핵심] 공통 물리 함수 사용
from physics_utils import calc_wind_profile_power_law

# ==========================================
# 1. 설정
# ==========================================
TARGET_TIME_IDX = 10  # 확인하고 싶은 시간대 인덱스
Z_LAYER_IDX = 4       # 보고 싶은 고도 층 (예: 4=40m)
VIS_W_SCALE = 15.0    # 수직풍(W) 시각화 증폭 배수

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
        t_max = float(d_maps.get('terrain_max', 1.0))
        
        # B. Meteorology (Normalized Data + Scales)
        d_met = np.load(os.path.join(p_dir, Config.SAVE_MET))
        met_data = d_met['met'] # (N, 4) -> Normalized (-1 ~ 1)
        
        # [수정 1] 스케일링 상수 로드
        scale_wind = float(d_met['max_uv']) 
        scale_L = float(d_met['max_L'])
        print(f"   -> Loaded Scales: Wind Max={scale_wind:.2f} m/s, L Max={scale_L:.2f}")
        
        # C. Labels (Conc)
        d_lbl = np.load(os.path.join(p_dir, Config.SAVE_LBL))
        conc_norm = d_lbl['conc']
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
    
    # 지형 복원
    real_terrain = terrain_norm * t_max
    
    # 농도 복원
    conc_log = conc_norm[TARGET_TIME_IDX] * c_std + c_mean
    conc_real = np.expm1(conc_log)
    conc_real = np.maximum(conc_real, 0)
    
    # 지형 기울기 계산
    grad_y, grad_x = np.gradient(real_terrain, Config.DY, Config.DX)

    # [수정 2] 기상 데이터 복원 (Normalized -> Physical Unit)
    uref_norm, vref_norm, L_norm, wd = met_data[TARGET_TIME_IDX]
    
    # 정규화 해제 (Denormalization)
    uref = uref_norm * scale_wind
    vref = vref_norm * scale_wind
    L = L_norm * scale_L
    
    print(f"\n[Met Info] Index: {TARGET_TIME_IDX}")
    print(f"   > Normalized Input: U={uref_norm:.2f}, V={vref_norm:.2f}")
    print(f"   > Restored Physical: U={uref:.2f} m/s, V={vref:.2f} m/s (WD: {wd:.1f}°)")
    print(f"   > Stability (L): {L:.2f}")
    print(f"   > Max Conc (Real): {np.max(conc_real):.4f}")

    # ---------------------------------------------------------
    # 3. 시각화 (물리 계산 수행)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(16, 9))
    plt.suptitle(f"Data Verification (Time: {TARGET_TIME_IDX})", fontsize=16)

    # === [Left] XY View (Top) ===
    ax1 = fig.add_subplot(1, 2, 1)
    z_height_m = Z_LAYER_IDX * Config.DZ
    
    # 배경: 지형
    ax1.contour(real_terrain, levels=10, colors='k', alpha=0.3)
    
    # 농도 맵
    im1 = ax1.imshow(conc_real[:, :, Z_LAYER_IDX], origin='lower', cmap='jet', alpha=0.8,
                     extent=[0, Config.NX, 0, Config.NY], 
                     vmax=np.percentile(conc_real, 99.5))
    
    # 오염원
    sy, sx = np.where(source_q_log > 0)
    ax1.scatter(sx, sy, c='red', marker='*', s=100, edgecolors='k')

    # 물리 함수 호출을 위한 Flatten
    sx_flat = grad_x.flatten()
    sy_flat = grad_y.flatten()
    z_flat = np.full(sx_flat.shape, z_height_m) 

    # [중요] 복원된 물리 값(uref, vref, L)을 사용해야 정상적인 벡터가 나옴
    wind_vecs_flat = calc_wind_profile_power_law(
        uref, vref, L, 
        z_points=z_flat, 
        slopes=(sx_flat, sy_flat)
    )
    wind_vecs = wind_vecs_flat.reshape(Config.NY, Config.NX, 3)

    # Quiver Plot
    U_2d = wind_vecs[:, :, 0]
    V_2d = wind_vecs[:, :, 1]
    
    step = 3
    ax1.quiver(np.arange(0, Config.NX, step), np.arange(0, Config.NY, step), 
               U_2d[::step, ::step], V_2d[::step, ::step], 
               color='white', scale=50, width=0.005)
    
    ax1.set_title(f"Top View (XY) @ {z_height_m}m\n(Wind Scaled Back to m/s)", fontsize=14)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1, label='Concentration (ppm)')

    # === [Right] Side View (XZ) ===
    max_pos = np.unravel_index(np.argmax(conc_real), conc_real.shape)
    target_y = max_pos[0]
    
    ax2 = fig.add_subplot(1, 2, 2)
    slice_conc = conc_real[target_y, :, :].T 
    
    xx = np.arange(Config.NX)
    zz = np.linspace(0, Config.NZ * Config.DZ, Config.NZ)
    XX, ZZ = np.meshgrid(xx, zz)
    
    ax2.contourf(XX, ZZ, slice_conc, levels=50, cmap='jet')
    
    h_prof = real_terrain[target_y, :]
    ax2.fill_between(xx, 0, h_prof, color='black', alpha=0.7)
    
    # 단면 바람장 계산
    gx_slice = grad_x[target_y, :]
    gy_slice = grad_y[target_y, :]
    
    slope_x_grid = np.tile(gx_slice, (Config.NZ, 1))
    slope_y_grid = np.tile(gy_slice, (Config.NZ, 1))
    
    vec_flat = calc_wind_profile_power_law(
        uref, vref, L, 
        z_points=ZZ.flatten(), 
        slopes=(slope_x_grid.flatten(), slope_y_grid.flatten())
    )
    
    U_grid = vec_flat[:, 0].reshape(Config.NZ, Config.NX)
    W_grid = vec_flat[:, 2].reshape(Config.NZ, Config.NX)
    
    mask = ZZ < h_prof
    U_grid[mask] = 0
    W_grid[mask] = 0
    W_vis = W_grid * VIS_W_SCALE
    
    ax2.quiver(XX[::2, ::2], ZZ[::2, ::2], 
               U_grid[::2, ::2], W_vis[::2, ::2],
               color='white', scale=100, width=0.005)
    
    ax2.set_title(f"Side View (XZ) @ Y={target_y}\n(Physically Restored Wind)", fontsize=14)
    ax2.set_ylim(0, Config.NZ * Config.DZ)
    
    plt.tight_layout()
    plt.savefig("data_verification.png", dpi=300)
    print("\n✅ Verification image saved.")
    plt.show()

if __name__ == "__main__":
    run_verification()