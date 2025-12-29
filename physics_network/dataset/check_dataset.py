import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.config_param import ConfigParam as Config

def check_normalization():
    print("=== Dataset Integrity & Normalization Check ===\n")
    p_dir = Config.PROCESSED_DIR
    
    # ---------------------------------------------------------
    # 1. Input Maps (Terrain, Source)
    # ---------------------------------------------------------
    print(f"1. Loading {Config.SAVE_MAPS} ...")
    try:
        d_maps = np.load(os.path.join(p_dir, Config.SAVE_MAPS))
        terrain = d_maps['terrain']
        source = d_maps['source_q']
        t_max = float(d_maps['terrain_max'])
        
        print(f"   - Terrain Shape: {terrain.shape}")
        
        # [Norm Check]
        print(f"   [Normalized Stats]")
        print(f"     > Range: {terrain.min():.4f} ~ {terrain.max():.4f}")
        is_valid = (terrain.min() >= 0.0) and (terrain.max() <= 1.0)
        print(f"     > 0~1 Check: {'✅ PASS' if is_valid else '❌ WARNING'}")
        
        # [Physical Check]
        print(f"   [Physical Stats]")
        print(f"     > Max Height: {t_max:.2f} m")
        print(f"     > Max Source (Log): {source.max():.4f}\n")
        
    except Exception as e:
        print(f"❌ Error loading maps: {e}")
        return

    # ---------------------------------------------------------
    # 2. Meteorology (수정됨)
    # ---------------------------------------------------------
    print(f"2. Loading {Config.SAVE_MET} ...")
    try:
        d_met = np.load(os.path.join(p_dir, Config.SAVE_MET))
        met_norm = d_met['met'] # 정규화된 데이터 (-1 ~ 1)
        
        # [수정] 스케일링 상수 로드
        max_uv = float(d_met['max_uv'])
        max_L = float(d_met['max_L'])
        
        print(f"   - Met Data Shape: {met_norm.shape}")
        
        # [Norm Check]
        print(f"   [Normalized Stats] (Expect -1.0 ~ 1.0)")
        u_min, u_max = met_norm[:, 0].min(), met_norm[:, 0].max()
        v_min, v_max = met_norm[:, 1].min(), met_norm[:, 1].max()
        print(f"     > U Range: {u_min:.4f} ~ {u_max:.4f}")
        print(f"     > V Range: {v_min:.4f} ~ {v_max:.4f}")
        
        if (abs(u_min) <= 1.0 and abs(u_max) <= 1.0):
            print("     -> ✅ Range Check Passed.")
        else:
            print("     -> ⚠️ Warning: Values exceed [-1, 1].")

        # [Physical Check] - 복원 수행
        print(f"   [Physical Stats] (Restored using max_uv={max_uv:.2f})")
        u_phys = met_norm[:, 0] * max_uv
        v_phys = met_norm[:, 1] * max_uv
        print(f"     > Real U Range: {u_phys.min():.2f} ~ {u_phys.max():.2f} m/s")
        print(f"     > Real V Range: {v_phys.min():.2f} ~ {v_phys.max():.2f} m/s")
        print("")

    except Exception as e:
        print(f"❌ Error loading met: {e}")
        return

    # ---------------------------------------------------------
    # 3. Labels (Concentration)
    # ---------------------------------------------------------
    print(f"3. Loading {Config.SAVE_LBL} ...")
    try:
        d_lbl = np.load(os.path.join(p_dir, Config.SAVE_LBL))
        conc_z = d_lbl['conc'] # Z-score Normalized
        mean_stat = float(d_lbl['mean_stat'])
        std_stat = float(d_lbl['std_stat'])
        
        print(f"   - Conc Shape: {conc_z.shape}")
        
        # [Norm Check]
        curr_mean = np.mean(conc_z)
        curr_std = np.std(conc_z)
        
        print(f"   [Normalized Stats] (Z-Score)")
        print(f"     > Mean: {curr_mean:.6f} (Target: 0.0)")
        print(f"     > Std : {curr_std:.6f}  (Target: 1.0)")
        
        if abs(curr_mean) < 0.1 and abs(curr_std - 1.0) < 0.1:
            print("     -> ✅ Normalization Distribution Correct.")
        else:
            print("     -> ⚠️ Warning: Distribution deviation detected.")

        # [Physical Check]
        print(f"   [Physical Stats]")
        # 샘플로 Max 값만 복원해서 확인
        max_z = np.max(conc_z)
        # 1) Z-score 역변환 -> 2) Log 역변환(expm1)
        max_phys = np.expm1(max_z * std_stat + mean_stat)
        print(f"     > Max Concentration: {max_phys:.4f} ppm (Restored)")

    except Exception as e:
        print(f"❌ Error loading labels: {e}")
        return

    # ---------------------------------------------------------
    # 4. 시각화 (히스토그램)
    # ---------------------------------------------------------
    print("\n4. Generating Distribution Plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1) Terrain (Normalized)
    axes[0].hist(terrain.flatten(), bins=50, color='green', alpha=0.7)
    axes[0].set_title('Terrain (Normalized 0~1)')
    axes[0].set_xlabel('Norm Height')
    
    # 2) Wind Speed (Physical Value로 표시!)
    # 사람이 보기엔 m/s가 편하므로 복원된 값으로 그림
    axes[1].hist(u_phys, bins=50, alpha=0.5, label='U (m/s)', color='blue')
    axes[1].hist(v_phys, bins=50, alpha=0.5, label='V (m/s)', color='orange')
    axes[1].set_title(f'Wind Speed (Physical m/s)\nScale Factor={max_uv:.2f}')
    axes[1].set_xlabel('Speed (m/s)')
    axes[1].legend()
    
    # 3) Concentration (Normalized Z-Score)
    # 학습 분포 확인용이므로 Z-score 그대로 그림
    axes[2].hist(conc_z.flatten(), bins=100, color='purple', alpha=0.7, range=(-3, 5))
    axes[2].set_title('Concentration (Z-Score Input)')
    axes[2].set_xlabel('Sigma ($\sigma$)')
    axes[2].axvline(0, color='red', linestyle='--', label='Mean')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('check_dataset_dist.png')
    print("✅ Distribution plot saved to: check_dataset_dist.png")
    plt.show()

if __name__ == "__main__":
    check_normalization()