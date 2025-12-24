# process_maps.py
import os
import numpy as np
import pandas as pd
from config_param import ConfigParam as Config
from physics_utils import xy_to_grid

def parse_locations(file_path, sources_dict):
    if not os.path.exists(file_path): return
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split()
            if 'LOCATION' in parts and 'POINT' in parts:
                try:
                    pt_idx = parts.index('POINT')
                    sid = parts[pt_idx - 1]
                    x = float(parts[pt_idx + 1])
                    y = float(parts[pt_idx + 2])
                    sources_dict.setdefault(sid, {})
                    sources_dict[sid]['x'] = x
                    sources_dict[sid]['y'] = y
                except: pass

def run():
    print("\n[Step 1] Processing Static Maps...")
    
    # 1. Terrain Map (GRIDCART Format)
    t_grid = np.zeros((Config.NY, Config.NX), dtype=np.float32)
    rou_path = os.path.join(Config.RAW_DIR, Config.FILE_ROU)
    
    points_filled = 0
    t_max_val = 1.0
    
    if os.path.exists(rou_path):
        try:
            row_data = {}
            with open(rou_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.split()
                    if 'ELEV' in parts: # ELEV 키워드만 찾음
                        try:
                            elev_idx = parts.index('ELEV')
                            row_id = int(parts[elev_idx + 1])
                            vals = []
                            for v in parts[elev_idx + 2:]:
                                try: vals.append(float(v))
                                except: pass
                            if row_id not in row_data: row_data[row_id] = []
                            row_data[row_id].extend(vals)
                        except: continue
            
            # 그리드 채우기
            for r_id, vals in row_data.items():
                py_row = r_id - 1
                if 0 <= py_row < Config.NY:
                    valid_vals = vals[:Config.NX]
                    t_grid[py_row, :len(valid_vals)] = valid_vals
                    points_filled += len(valid_vals)
            
            if points_filled > 0:
                t_max_val = t_grid.max() # [중요] 최대값 저장
                print(f"   > Max Terrain Height: {t_max_val:.2f} m")
                if t_max_val > 0: t_grid /= t_max_val # 0~1 정규화
                print(f"   -> Terrain processed ({points_filled} points).")
                
        except Exception as e:
            print(f"   ! Terrain Error: {e}")

    # 2. Source Maps
    q_grid = np.zeros((Config.NY, Config.NX), dtype=np.float32)
    h_grid = np.zeros((Config.NY, Config.NX), dtype=np.float32)
    
    inp_path = os.path.join(Config.RAW_DIR, Config.FILE_INP)
    loc_path = os.path.join(Config.RAW_DIR, Config.FILE_SRC_LOC)
    sources = {}
    
    parse_locations(loc_path, sources) # 위치 로드
    
    if os.path.exists(inp_path): # 파라미터 로드
        with open(inp_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.split()
                if 'SRCPARAM' in parts:
                    try:
                        p_idx = parts.index('SRCPARAM')
                        sid = parts[p_idx + 1]
                        q_val = float(parts[p_idx + 2])
                        h_val = float(parts[p_idx + 3])
                        sources.setdefault(sid, {})
                        sources[sid]['q'] = q_val
                        sources[sid]['h'] = h_val
                    except: pass

    cnt = 0
    for sid, data in sources.items():
        if 'x' in data and 'y' in data:
            idx = xy_to_grid(data['x'], data['y'])
            if idx:
                q_grid[idx] += data.get('q', 1.0)
                h_grid[idx] = max(h_grid[idx], data.get('h', 10.0))
                cnt += 1
    print(f"   -> Mapped {cnt} sources.")
    
    # [수정] terrain_max 함께 저장
    save_path = os.path.join(Config.PROCESSED_DIR, Config.SAVE_MAPS)
    np.savez_compressed(save_path, 
                        terrain=t_grid, 
                        source_q=q_grid, 
                        source_h=h_grid,
                        terrain_max=t_max_val)

if __name__ == "__main__":
    run()