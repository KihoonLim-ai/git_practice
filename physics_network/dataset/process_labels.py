import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset.config_param import ConfigParam as Config
from dataset.physics_utils import xy_to_grid

def get_meteor_timestamps(file_path):
    """
    METEOR.DBG 파일을 읽어 process_met.py와 동일한 순서의 타임스탬프 리스트를 반환
    Format: YYMMDDHH (int)
    """
    timestamps = []
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return []

    print(f"   Parsing timestamps from {file_path}...")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        
        # 날짜 라인 감지 (process_met.py와 동일 로직)
        # 조건: 길이가 길고(>8), 첫 부분이 숫자
        if len(parts) > 8 and parts[0].isdigit():
            try:
                # YR MO DA HR 추출
                yr, mo, dy, hr = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                
                # 연도 보정 (2024 -> 24)
                yr_2d = yr % 100
                
                # 타임스탬프 생성 (YYMMDDHH)
                ts = int(f"{yr_2d:02d}{mo:02d}{dy:02d}{hr:02d}")
                timestamps.append(ts)
            except ValueError:
                continue
                
    # 중복 제거 없이 순서대로 반환 (process_met도 순서대로 쌓았으므로)
    # 단, METEOR.DBG는 고도별로 데이터가 반복되지 않고 시간별로 한 블록씩 나옴
    # 따라서 추출된 리스트가 곧 시간축임.
    return timestamps

def run():
    print("\n[Step 3] Processing Labels (Syncing with METEOR.DBG timestamps)...")
    
    # 1. 타임스탬프 로드 (Input 데이터와 동기화)
    met_path = os.path.join(Config.RAW_DIR, "met/METEOR.DBG")
    target_dates = get_meteor_timestamps(met_path)
    
    if not target_dates:
        print("Error: Could not extract timestamps from METEOR.DBG")
        return

    num_steps = len(target_dates)
    print(f"   -> Found {num_steps} time steps from Meteorology data.")
    
    # 빠른 조회를 위한 매핑 (Timestamp -> Index)
    time_idx_map = {ts: i for i, ts in enumerate(target_dates)}
    
    # 2. 빈 4D 배열 생성 (Time, Y, X, Z)
    # Config.NZ는 21이어야 함 (0~200m, 10m 간격)
    conc_4d = np.zeros((num_steps, Config.NY, Config.NX, Config.NZ), dtype=np.float32)
    
    plt_folder = os.path.join(Config.RAW_DIR, Config.PLT_DIR_NAME)
    
    # 3. 고도별 파일 읽기
    # process_met.py의 target_heights와 일치해야 함 (0, 10, ..., 200)
    z_levels = range(0, 210, 10) 
    
    available_files = []
    for z_val in z_levels:
        fname = Config.PLT_FMT.format(z=z_val) # 예: 'conc_z{z}.plt'
        fpath = os.path.join(plt_folder, fname)
        if os.path.exists(fpath):
            available_files.append((z_val, fpath))
    
    if not available_files:
        print("Error: No PLT files found. Check Config.PLT_DIR_NAME and PLT_FMT.")
        return
        
    print(f"   -> Processing {len(available_files)} vertical layers...")

    # 실제 파일 읽기 루프
    for z_idx, (z_val, fpath) in enumerate(tqdm(available_files, desc="Reading PLT Files")):
        try:
            # PLT 파일 포맷: X Y C ... DATE ...
            # 보통 AERMOD PLT는 공백 구분
            # usecols=[0, 1, 2, 8] -> X, Y, AverageConc, Date(YYMMDDHH)
            # 주의: 파일 포맷에 따라 컬럼 인덱스가 다를 수 있으니 확인 필요
            # (여기서는 기존 코드를 신뢰함)
            
            # chunksize를 사용하여 메모리 효율성 증대 (파일이 클 경우 대비)
            chunk_size = 50000
            for chunk in pd.read_csv(fpath, sep=r'\s+', skiprows=Config.PLT_SKIP_ROWS, header=None,
                                     usecols=[0, 1, 2, 8], names=['x', 'y', 'c', 'date'], 
                                     chunksize=chunk_size):
                
                # 유효한 시간대만 필터링 (Pandas Merge or Isin)
                # 속도를 위해 map 사용
                
                # 벡터화 연산을 위해 numpy로 변환
                dates = chunk['date'].values
                cs = chunk['c'].values
                xs = chunk['x'].values
                ys = chunk['y'].values
                
                for i in range(len(dates)):
                    ts = int(dates[i])
                    if ts in time_idx_map:
                        t_idx = time_idx_map[ts]
                        # 좌표 변환
                        idx = xy_to_grid(xs[i], ys[i])
                        if idx:
                            # 4D 배열에 할당
                            conc_4d[t_idx, idx[0], idx[1], z_idx] = cs[i]
                            
        except Exception as e:
            print(f"Warning processing z={z_val}: {e}")
            pass

    # ---------------------------------------------------------
    # 4. 데이터 전처리 (Log1p -> StandardScaler)
    # ---------------------------------------------------------
    print("   >> Applying Log1p transformation...")
    conc_log = np.log1p(conc_4d)
    
    print("   >> Applying Global StandardScaler...")
    global_mean = np.mean(conc_log)
    global_std = np.std(conc_log)
    
    if global_std == 0: global_std = 1.0
    
    conc_normalized = (conc_log - global_mean) / global_std
    
    # ---------------------------------------------------------
    # 5. 저장
    # ---------------------------------------------------------
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
    save_path = os.path.join(Config.PROCESSED_DIR, Config.SAVE_LBL)
    
    np.savez_compressed(
        save_path, 
        conc=conc_normalized, 
        mean_stat=global_mean, 
        std_stat=global_std
    )
    
    print(f"   >> Process Completed.")
    print(f"   >> Data Shape: {conc_normalized.shape} (Should be {num_steps}, 45, 45, 21)")
    print(f"   >> Saved to {save_path}")

if __name__ == "__main__":
    run()