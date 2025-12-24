# process_met.py
import os
import numpy as np
import pandas as pd
from config_param import ConfigParam as Config

def run():
    print("\n[Step 2] Processing Met Data...")
    path = os.path.join(Config.RAW_DIR, Config.FILE_SFC)
    try:
        df_raw = pd.read_csv(path, sep=r'\s+', skiprows=1, header=None)
    except: return pd.DataFrame()

    col_map = {
        Config.IDX_YEAR: 'year', Config.IDX_MONTH: 'month',
        Config.IDX_DAY: 'day',   Config.IDX_HOUR: 'hour',
        Config.IDX_L: 'L',       Config.IDX_WS: 'ws', Config.IDX_WD: 'wd'
    }
    valid = [c for c in col_map.keys() if c < df_raw.shape[1]]
    df = df_raw[valid].copy()
    df.columns = [col_map[c] for c in valid]
    
    df = df[(df['ws'] < 100.0) & (df['L'] > -90000.0)]
    df = df[df['month'].isin(Config.TARGET_MONTHS)].reset_index(drop=True)
    print(f"   -> Valid Summer Data: {len(df)} timestamps")
    
    met_data = []
    for _, row in df.iterrows():
        ws, wd, L = row['ws'], row['wd'], row['L']
        uref = -ws * np.sin(np.deg2rad(wd))
        vref = -ws * np.cos(np.deg2rad(wd))
        met_data.append([uref, vref, L, wd])
        
    met_arr = np.array(met_data, dtype=np.float32)
    save_path = os.path.join(Config.PROCESSED_DIR, Config.SAVE_MET)
    np.savez_compressed(save_path, met=met_arr)
    
    return df

if __name__ == "__main__":
    run()