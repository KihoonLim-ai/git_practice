import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset.config_param import ConfigParam as Config
from dataset.physics_utils import xy_to_grid

def run(met_df):
    if met_df is None or len(met_df) == 0: return
    print(f"\n[Step 3] Processing Labels (Reading Height-major PLT files)...")
    
    # 1. 타임스탬프 필터 준비
    target_dates = []
    valid_timestamps = set()
    for _, row in met_df.iterrows():
        # config에 12개월이 다 들어있다면, 여기서 1년치 타임스탬프가 생성됨
        yr, mo, dy, hr = int(row.get('year',0)), int(row.get('month',6)), int(row.get('day',1)), int(row.get('hour',1))
        yr_2d = yr % 100
        ts = int(f"{yr_2d:02d}{mo:02d}{dy:02d}{hr:02d}")
        valid_timestamps.add(ts)
        target_dates.append(ts)
        
    time_idx_map = {d: i for i, d in enumerate(target_dates)}
    
    # (Time, Y, X, Z) 형태의 빈 배열 생성
    conc_4d = np.zeros((len(met_df), Config.NY, Config.NX, Config.NZ), dtype=np.float32)
    plt_folder = os.path.join(Config.RAW_DIR, Config.PLT_DIR_NAME)
    
    # 2. 고도별 파일 읽기 및 데이터 채우기
    z_levels = range(0, 210, 10)
    
    # 파일이 존재하는지 먼저 확인하여 tqdm 바가 헛돌지 않게 함
    available_files = []
    for z_val in z_levels:
        fname = Config.PLT_FMT.format(z=z_val)
        fpath = os.path.join(plt_folder, fname)
        if os.path.exists(fpath):
            available_files.append((z_val, fpath))
            
    if not available_files:
        print("Error: No PLT files found.")
        return

    # 실제 읽기 루프
    for z_idx, (z_val, fpath) in enumerate(tqdm(available_files, desc="Processing Heights")):
        try:
            # X(0), Y(1), C(2), DATE(8) 컬럼만 읽기
            df_plt = pd.read_csv(fpath, sep=r'\s+', skiprows=Config.PLT_SKIP_ROWS, header=None,
                                 usecols=[0, 1, 2, 8], names=['x', 'y', 'c', 'date'])
            
            # 유효한 시간대만 필터링
            df_filtered = df_plt[df_plt['date'].isin(valid_timestamps)]
            
            for _, row in df_filtered.iterrows():
                t_idx = time_idx_map.get(int(row['date']))
                if t_idx is not None:
                    idx = xy_to_grid(row['x'], row['y'])
                    if idx: 
                        # Grid에 값 할당
                        conc_4d[t_idx, idx[0], idx[1], z_idx] = row['c']
        except Exception as e:
            print(f"Warning processing z={z_val}: {e}")
            pass

    # ---------------------------------------------------------
    # [핵심 수정] 3. 데이터 분포 변환 (Log1p -> StandardScaler)
    # ---------------------------------------------------------
    print("   >> Applying Log1p transformation...")
    # A. Log 변환: C' = log(C + 1)
    # 오염 농도는 0인 경우가 많으므로 log1p가 필수적임
    conc_log = np.log1p(conc_4d)
    
    print("   >> Applying StandardScaler (Mean=0, Std=1)...")
    # B. 통계량 계산 (전체 데이터 기준)
    global_mean = np.mean(conc_log)
    global_std = np.std(conc_log)
    
    if global_std == 0: global_std = 1.0 # 0 나누기 방지
    
    # C. 정규화 (Standardization): Z = (X - u) / s
    conc_normalized = (conc_log - global_mean) / global_std
    
    # 저장 경로 설정
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
    save_path = os.path.join(Config.PROCESSED_DIR, Config.SAVE_LBL)
    
    # 4. 저장 (데이터 + 복원용 통계량)
    # 나중에 모델 추론 값을 복원하려면: pred_real = exp(pred_norm * std + mean) - 1
    np.savez_compressed(
        save_path, 
        conc=conc_normalized, # 정규화된 데이터 저장
        mean_stat=global_mean, # 복원용 평균
        std_stat=global_std    # 복원용 표준편차
    )
    
    print(f"   >> Process Completed.")
    print(f"   >> Data Shape: {conc_normalized.shape}")
    print(f"   >> Log-Space Mean: {global_mean:.4f}, Std: {global_std:.4f}")
    print(f"   >> Saved to {save_path}")
