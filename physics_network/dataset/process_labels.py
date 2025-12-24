# process_labels.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from config_param import ConfigParam as Config
from physics_utils import xy_to_grid

def run(met_df):
    if met_df is None or len(met_df) == 0: return
    print(f"\n[Step 3] Processing Labels (Reading Height-major PLT files)...")
    
    # 타임스탬프 필터 준비
    target_dates = []
    valid_timestamps = set()
    for _, row in met_df.iterrows():
        yr, mo, dy, hr = int(row.get('year',0)), int(row.get('month',6)), int(row.get('day',1)), int(row.get('hour',1))
        yr_2d = yr % 100
        ts = int(f"{yr_2d:02d}{mo:02d}{dy:02d}{hr:02d}")
        valid_timestamps.add(ts)
        target_dates.append(ts)
        
    time_idx_map = {d: i for i, d in enumerate(target_dates)}
    conc_4d = np.zeros((len(met_df), Config.NY, Config.NX, Config.NZ), dtype=np.float32)
    plt_folder = os.path.join(Config.RAW_DIR, Config.PLT_DIR_NAME)
    
    # 고도별 처리
    z_levels = range(0, 210, 10)
    for z_idx, z_val in enumerate(tqdm(z_levels, desc="Processing Heights")):
        fname = Config.PLT_FMT.format(z=z_val)
        fpath = os.path.join(plt_folder, fname)
        
        if not os.path.exists(fpath): continue
        try:
            # X(0), Y(1), C(2), DATE(8)
            df_plt = pd.read_csv(fpath, sep=r'\s+', skiprows=Config.PLT_SKIP_ROWS, header=None,
                                 usecols=[0, 1, 2, 8], names=['x', 'y', 'c', 'date'])
            
            df_filtered = df_plt[df_plt['date'].isin(valid_timestamps)]
            for _, row in df_filtered.iterrows():
                t_idx = time_idx_map.get(int(row['date']))
                if t_idx is not None:
                    idx = xy_to_grid(row['x'], row['y'])
                    if idx: conc_4d[t_idx, idx[0], idx[1], z_idx] = row['c']
        except: pass

    save_path = os.path.join(Config.PROCESSED_DIR, Config.SAVE_LBL)
    np.savez_compressed(save_path, conc=conc_4d)
    print(f"   >> Saved labels to {save_path}")