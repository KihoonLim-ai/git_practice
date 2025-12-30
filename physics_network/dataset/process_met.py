import os
import numpy as np
import pandas as pd
from dataset.config_param import ConfigParam as Config

def run():
    print("\n[Step 2] Processing Met Data (FIXED for WD=999)...")
    path = os.path.join(Config.RAW_DIR, Config.FILE_SFC)
    try:
        df_raw = pd.read_csv(path, sep=r'\s+', skiprows=1, header=None)
    except: 
        print("Error: Could not read SFC file.")
        return None # Return None on failure

    col_map = {
        Config.IDX_YEAR: 'year', Config.IDX_MONTH: 'month',
        Config.IDX_DAY: 'day',   Config.IDX_HOUR: 'hour',
        Config.IDX_L: 'L',       Config.IDX_WS: 'ws', Config.IDX_WD: 'wd'
    }
    valid = [c for c in col_map.keys() if c < df_raw.shape[1]]
    df = df_raw[valid].copy()
    df.columns = [col_map[c] for c in valid]
    
    print(f"   Original Entries: {len(df)}")

    # -----------------------------------------------------------
    # [수정] 결측치(999, -9999 등) 전처리 강화
    # -----------------------------------------------------------
    # 1. 풍속(WS) 이상치 처리
    df.loc[df['ws'] >= 100.0, 'ws'] = np.nan
    
    # 2. [추가됨] 풍향(WD) 999.0 처리 (AERMET Missing Code)
    df.loc[df['wd'] >= 990.0, 'wd'] = np.nan
    
    # 3. Monin-Obukhov Length(L) 이상치 처리
    df.loc[df['L'] <= -90000.0, 'L'] = np.nan
    
    # -----------------------------------------------------------
    
    # 4. 선형 보간 (Linear Interpolate)
    # 풍향(wd)은 0~360도이므로 단순 선형 보간은 359->1도 갈 때 문제가 되지만,
    # 결측이 짧다면 큰 문제는 없습니다. (엄밀하게는 sin/cos 보간이 좋으나 여기선 패스)
    df['ws'] = df['ws'].interpolate(method='linear', limit=4)
    df['L']  = df['L'].interpolate(method='linear', limit=4)
    df['wd'] = df['wd'].interpolate(method='linear', limit=4) 
    
    # 5. 여전히 남은 결측 제거
    df = df.dropna().reset_index(drop=True)
    
    print(f"   -> Valid Data (After Cleaning 999): {len(df)} timestamps")
    
    # 6. 벡터 변환
    met_data = []
    for _, row in df.iterrows():
        ws, wd, L = row['ws'], row['wd'], row['L']
        uref = -ws * np.sin(np.deg2rad(wd))
        vref = -ws * np.cos(np.deg2rad(wd))
        met_data.append([uref, vref, L, wd])
        
    met_arr = np.array(met_data, dtype=np.float32)
    
    # 7. 정규화 통계 계산
    max_uv = np.max(np.abs(met_arr[:, :2])) 
    if max_uv == 0: max_uv = 1.0
    
    max_L = np.max(np.abs(met_arr[:, 2]))
    if max_L == 0: max_L = 1.0

    print(f"   -> Max Wind Speed: {max_uv:.4f} m/s")
    # -----------------------------------------------------------
    # 정규화 적용 (-1 ~ 1)
    # -----------------------------------------------------------
    met_arr[:, 0] /= max_uv
    met_arr[:, 1] /= max_uv
    met_arr[:, 2] /= max_L
    
    # 8. 저장
    save_path = os.path.join(Config.PROCESSED_DIR, Config.SAVE_MET)
    np.savez_compressed(
        save_path, 
        met=met_arr,       
        max_uv=max_uv,     
        max_L=max_L        
    )
    print(f"   >> Saved FIXED met data to {save_path}")
    
    return df

if __name__ == "__main__":
    run()