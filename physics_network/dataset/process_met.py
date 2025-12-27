import os
import numpy as np
import pandas as pd
from dataset.config_param import ConfigParam as Config

def run():
    print("\n[Step 2] Processing Met Data...")
    path = os.path.join(Config.RAW_DIR, Config.FILE_SFC)
    try:
        df_raw = pd.read_csv(path, sep=r'\s+', skiprows=1, header=None)
    except: 
        print("Error: Could not read SFC file.")
        return pd.DataFrame()

    col_map = {
        Config.IDX_YEAR: 'year', Config.IDX_MONTH: 'month',
        Config.IDX_DAY: 'day',   Config.IDX_HOUR: 'hour',
        Config.IDX_L: 'L',       Config.IDX_WS: 'ws', Config.IDX_WD: 'wd'
    }
    valid = [c for c in col_map.keys() if c < df_raw.shape[1]]
    df = df_raw[valid].copy()
    df.columns = [col_map[c] for c in valid]
    
    # 1. 이상치 제거 (물리적으로 불가능한 값)
    df = df[(df['ws'] < 100.0) & (df['L'] > -90000.0)]
    
    # [수정] 월 필터링 제거 (전체 데이터 사용)
    # df = df[df['month'].isin(Config.TARGET_MONTHS)].reset_index(drop=True) 
    print(f"   -> Valid Data (All Seasons): {len(df)} timestamps")
    
    # 2. 벡터 변환 (Wind Speed/Dir -> U, V)
    met_data = []
    for _, row in df.iterrows():
        ws, wd, L = row['ws'], row['wd'], row['L']
        # 기상학적 벡터 변환 (바람이 불어오는 방향 기준 -> 수학적 벡터)
        uref = -ws * np.sin(np.deg2rad(wd))
        vref = -ws * np.cos(np.deg2rad(wd))
        met_data.append([uref, vref, L, wd])
        
    met_arr = np.array(met_data, dtype=np.float32)
    
    # 3. [중요] 정규화를 위한 통계 계산
    # u, v 성분의 절대값 중 가장 큰 값 찾기
    max_uv = np.max(np.abs(met_arr[:, :2])) 
    if max_uv == 0: max_uv = 1.0
    
    # L 성분의 절대값 중 가장 큰 값 찾기
    max_L = np.max(np.abs(met_arr[:, 2]))
    if max_L == 0: max_L = 1.0

    print(f"   -> Max Wind Speed (Component): {max_uv:.4f} m/s")
    print(f"   -> Max Monin-Obukhov Length: {max_L:.4f} m")

    # 4. 저장 (데이터 + 통계)
    save_path = os.path.join(Config.PROCESSED_DIR, Config.SAVE_MET)
    np.savez_compressed(
        save_path, 
        met=met_arr,       # Raw 데이터 (dataset.py에서 나눌 예정)
        max_uv=max_uv,     # 정규화 상수
        max_L=max_L        # 정규화 상수
    )
    print(f"   >> Saved met data to {save_path}")
    
    return df

if __name__ == "__main__":
    run()