# verify_data.py
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config_param import ConfigParam as Config
# [핵심] 수식을 직접 쓰지 않고, 정의된 함수를 가져다 씁니다.
from physics_utils import calc_wind_profile_power_law

# ==========================================
# 1. 설정
# ==========================================
TARGET_TIME_IDX = 10 
Z_LAYER_IDX = 10

# 시각화에서 수직풍(W)을 얼마나 과장해서 보여줄지 (실제로는 작아서 잘 안 보임)
VIS_W_SCALE = 10.0 

def run_verification():
    print("=== Data Verification (Using Shared Physics Function) ===")
    
    # 1. 데이터 로드
    p_dir = Config.PROCESSED_DIR
    try:
        d_maps = np.load(os.path.join(p_dir, Config.SAVE_MAPS))
        terrain = d_maps['terrain']
        source_q = d_maps['source_q']
        # 저장된 최대 높이 로드
        t_max = float(d_maps.get('terrain_max', 1.0))
        
        d_met = np.load(os.path.join(p_dir, Config.SAVE_MET))
        met_data = d_met['met']
        
        d_lbl = np.load(os.path.join(p_dir, Config.SAVE_LBL))
        conc_data = d_lbl['conc']
        
    except FileNotFoundError:
        print("❌ 파일을 찾을 수 없습니다.")
        return

    # 2. 지형 기울기 계산 (실제 높이 스케일 반영)
    real_terrain = terrain * t_max
    grad_y, grad_x = np.gradient(real_terrain, Config.DY, Config.DX)
    
    # 3. 타겟 데이터
    uref, vref, L, wd = met_data[TARGET_TIME_IDX]
    conc_3d = conc_data[TARGET_TIME_IDX]
    
    print(f"\n[Met Info] Time: {TARGET_TIME_IDX}, WD: {wd:.1f}, Max H: {t_max:.1f}m")

    # 4. 시각화
    fig = plt.figure(figsize=(16, 9))
    plt.suptitle(f"Verification using 'calc_wind_profile_power_law' - Time: {TARGET_TIME_IDX}", fontsize=16)

    # ---------------------------------------------------------
    # [왼쪽] XY View (Top)
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1)
    z_height = Z_LAYER_IDX * Config.DZ
    
    # 배경: 지형 & 농도
    ax1.contour(terrain, levels=10, colors='k', alpha=0.4)
    im1 = ax1.imshow(conc_3d[:, :, Z_LAYER_IDX], origin='lower', cmap='jet', alpha=0.8,
                     extent=[0, Config.NX, 0, Config.NY], vmax=np.percentile(conc_3d, 99.8))
    
    # 오염원
    sy, sx = np.where(source_q > 0)
    ax1.scatter(sx, sy, c='red', marker='*', s=120, edgecolors='k')

    # [함수 사용] 2D 바람장 생성
    # z_height(스칼라)와 grad(2D 배열)를 넣으면 -> 함수가 Broadcasting 처리
    wind_vecs = calc_wind_profile_power_law(
        uref, vref, L, np.array([z_height]), 
        slopes=(grad_x, grad_y)
    )
    # wind_vecs shape: (45, 45, 3)
    U_2d = wind_vecs[:, :, 0]
    V_2d = wind_vecs[:, :, 1]
    
    # 퀴버 플롯
    step = 4
    ax1.quiver(np.arange(2,43,step), np.arange(2,43,step), 
               U_2d[2:43:step, 2:43:step], V_2d[2:43:step, 2:43:step], 
               color='white', scale=50, width=0.006)
    
    ax1.set_title(f"XY View @ {z_height}m", fontsize=14)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1, label='Concentration')

    # ---------------------------------------------------------
    # [오른쪽] XZ View (Side) - 여기가 핵심
    # ---------------------------------------------------------
    max_pos = np.unravel_index(np.argmax(conc_3d), conc_3d.shape)
    target_y = max_pos[1]
    
    ax2 = fig.add_subplot(1, 2, 2)
    
    # 배경
    slice_conc = conc_3d[target_y, :, :].T
    xx = np.arange(Config.NX)
    zz = np.linspace(0, 200, Config.NZ) # 0~200m
    XX, ZZ = np.meshgrid(xx, zz) # (NZ, NX) 형태
    
    ax2.contourf(XX, ZZ, slice_conc, levels=50, cmap='jet')
    ax2.fill_between(xx, 0, terrain[target_y, :] * t_max, color='k', alpha=0.8)

    # [함수 사용] 단면 벡터장 생성
    # 1. 해당 단면의 기울기 가져오기 (NX,)
    gx_slice = grad_x[target_y, :]
    gy_slice = grad_y[target_y, :]
    
    # 2. 기울기를 2D 메쉬 그리드(NZ, NX)에 맞게 확장
    # XX와 같은 모양으로 확장 (Z축으로는 기울기가 동일하므로)
    # broadcast_to를 쓰거나 np.tile 사용
    slope_x_grid = np.tile(gx_slice, (Config.NZ, 1)) # (NZ, NX)
    slope_y_grid = np.tile(gy_slice, (Config.NZ, 1)) # (NZ, NX)
    
    # 3. 함수 호출 (Flatten해서 전달 후 다시 Reshape)
    flat_z = ZZ.flatten()
    flat_sx = slope_x_grid.flatten()
    flat_sy = slope_y_grid.flatten()
    
    # 함수는 1D array 입력에 대해 잘 작동함
    vec_flat = calc_wind_profile_power_law(
        uref, vref, L, flat_z, 
        slopes=(flat_sx, flat_sy)
    )
    
    # 4. 결과 Reshape (N_points, 3) -> (NZ, NX)
    U_grid = vec_flat[:, 0].reshape(Config.NZ, Config.NX)
    W_grid = vec_flat[:, 2].reshape(Config.NZ, Config.NX)
    
    # 5. 시각화용 샘플링 & W 증폭
    vis_step_z = 2
    vis_step_x = 2
    
    # 지형 아래는 마스킹 (시각화 깔끔하게)
    h_prof = terrain[target_y, :] * t_max
    mask = ZZ < h_prof  # 지형보다 낮은 곳 True
    
    U_vis = U_grid.copy()
    W_vis = W_grid.copy()
    
    U_vis[mask] = 0
    W_vis[mask] = 0
    
    # [시각화용 W 증폭] : 물리적으로 계산된 값에 시각적 확인을 위해 상수배
    W_vis *= VIS_W_SCALE
    
    ax2.quiver(XX[::vis_step_z, ::vis_step_x], 
               ZZ[::vis_step_z, ::vis_step_x], 
               U_vis[::vis_step_z, ::vis_step_x], 
               W_vis[::vis_step_z, ::vis_step_x], 
               color='white', scale=100, width=0.006, headwidth=4)
    
    ax2.set_title(f"Side View @ Y={target_y}\n(W Amplified x{VIS_W_SCALE} for Vis)", fontsize=14)
    ax2.set_ylim(0, 200)

    plt.tight_layout()
    plt.savefig("data_verification.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_verification()