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
                    if 'ELEV' in parts:
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
            
            for r_id, vals in row_data.items():
                py_row = r_id - 1
                if 0 <= py_row < Config.NY:
                    valid_vals = vals[:Config.NX]
                    t_grid[py_row, :len(valid_vals)] = valid_vals
                    points_filled += len(valid_vals)
            
            if points_filled > 0:
                t_max_val = t_grid.max()
                print(f"   > Max Terrain Height: {t_max_val:.2f} m")
                # [지형 정규화] 0~1
                if t_max_val > 0: t_grid /= t_max_val 
                print(f"   -> Terrain processed ({points_filled} points).")
                
        except Exception as e:
            print(f"   ! Terrain Error: {e}")

    # 2. Source Maps
    q_grid = np.zeros((Config.NY, Config.NX), dtype=np.float32)
    h_grid = np.zeros((Config.NY, Config.NX), dtype=np.float32)
    
    inp_path = os.path.join(Config.RAW_DIR, Config.FILE_INP)
    loc_path = os.path.join(Config.RAW_DIR, Config.FILE_SRC_LOC)
    sources = {}
    
    parse_locations(loc_path, sources)
    
    if os.path.exists(inp_path):
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
    
    # -----------------------------------------------------------
    # [추가됨] 오염원 데이터 정규화
    # -----------------------------------------------------------
    print(f"   -> Mapped {cnt} sources.")

    # 1) 배출량(Q): 편차가 크므로 Log Scale 적용 (농도 데이터와 전략 통일)
    # log1p를 적용하여 0인 값은 0으로 유지
    q_grid = np.log1p(q_grid) 
    print("   -> Applied Log1p scaling to Source Emission (Q)")

    # 2) 굴뚝 높이(H): 도메인 전체 높이(MAX_Z)로 나누어 0~1로 맞춤
    # Config에 MAX_Z가 없다면 대략적인 최대값(예: 200m) 혹은 t_max_val 사용
    # 여기서는 Config.NZ * Config.DZ 를 도메인 천장으로 가정
    domain_max_z = Config.MAX_Z
    if domain_max_z > 0:
        h_grid /= domain_max_z
    print(f"   -> Applied MinMax scaling to Source Height (H) using max_z={domain_max_z}m")
    # -----------------------------------------------------------

    save_path = os.path.join(Config.PROCESSED_DIR, Config.SAVE_MAPS)
    np.savez_compressed(save_path, 
                        terrain=t_grid, 
                        source_q=q_grid, 
                        source_h=h_grid,
                        terrain_max=t_max_val) # 복원용
    
    print(f"   >> Saved maps to {save_path}")

if __name__ == "__main__":
    run()
