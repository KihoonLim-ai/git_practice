import os
import numpy as np
import pandas as pd
from dataset.config_param import ConfigParam as Config

def run():
    print("\n[Step 2] Processing Met Data (With Interpolation)...")
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
    
    # -----------------------------------------------------------
    # [수정 1] 이상치 처리 방식 변경: 삭제(Drop) -> 보간(Interpolate)
    # -----------------------------------------------------------
    print(f"   Original Entries: {len(df)}")

    # 1. 이상치(결측치)를 NaN으로 마킹
    # AERMET에서 ws >= 100, L <= -90000 은 보통 결측을 의미함
    df.loc[df['ws'] >= 100.0, 'ws'] = np.nan
    df.loc[df['L'] <= -90000.0, 'L'] = np.nan
    
    # 2. 선형 보간 (Linear Interpolation) 적용
    # limit=4: 연속된 결측이 4시간 이하라면 채워줌. 그 이상은 너무 길어서 신뢰할 수 없으므로 둠.
    df['ws'] = df['ws'].interpolate(method='linear', limit=4)
    df['L']  = df['L'].interpolate(method='linear', limit=4)
    df['wd'] = df['wd'].interpolate(method='linear', limit=4) 
    
    # 3. 보간으로도 못 채운(너무 긴 결측 구간) 데이터만 삭제
    df = df.dropna().reset_index(drop=True)
    
    # [수정 2] 월 필터링 로직 삭제 (주석 처리)
    # 8,000시간 전체 데이터를 쓰기로 했으므로 필터링 없이 통과시킴
    # df = df[df['month'].isin(Config.TARGET_MONTHS)].reset_index(drop=True) 
    
    print(f"   -> Valid Data (After Interpolation): {len(df)} timestamps")
    
    # -----------------------------------------------------------
    
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

    # -----------------------------------------------------------
    # [수정] 정규화 적용 (Normalization)
    # 저장하기 전에 미리 나눠서 -1 ~ 1 사이로 만듭니다.
    # -----------------------------------------------------------
    met_arr[:, 0] = met_arr[:, 0] / max_uv  # U 성분 정규화
    met_arr[:, 1] = met_arr[:, 1] / max_uv  # V 성분 정규화
    met_arr[:, 2] = met_arr[:, 2] / max_L   # L 성분 정규화
    
    print("   -> Applied Normalization (Range: -1.0 ~ 1.0)")

    # 4. 저장 (데이터 + 통계)
    save_path = os.path.join(Config.PROCESSED_DIR, Config.SAVE_MET)
    np.savez_compressed(
        save_path, 
        met=met_arr,       # 이제 정규화된 데이터가 저장됨
        max_uv=max_uv,     # 복원을 위해 스케일 값은 계속 저장
        max_L=max_L        
    )
    print(f"   >> Saved met data to {save_path}")
    
    return df

if __name__ == "__main__":
    run()