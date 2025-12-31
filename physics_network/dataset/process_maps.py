import os
import numpy as np
import pandas as pd
from dataset.config_param import ConfigParam as Config
from dataset.physics_utils import xy_to_grid

def parse_locations(file_path, sources_dict):
    """외부 소스 위치 파일(.src) 파싱"""
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

def add_source_gaussian(q_grid, h_grid, center_x, center_y, q_val, h_val, sigma=1.0):
    """오염원 가우시안 스플래팅"""
    cx, cy = int(center_x), int(center_y)
    r = 2
    
    kernel = np.zeros((2*r+1, 2*r+1), dtype=np.float32)
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            dist_sq = dx**2 + dy**2
            weight = np.exp(-dist_sq / (2 * sigma**2))
            kernel[dy+r, dx+r] = weight
            
    if kernel.sum() > 0:
        kernel /= kernel.sum()
        
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < Config.NY and 0 <= nx < Config.NX:
                q_grid[ny, nx] += q_val * kernel[dy+r, dx+r]
                h_grid[ny, nx] = max(h_grid[ny, nx], h_val)

def run():
    print("\n[Step 1] Processing Static Maps (Saving RELATIVE coordinates)...")
    
    # -----------------------------------------------------------
    # 1. Terrain Map
    # -----------------------------------------------------------
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
                if t_max_val > 0: t_grid /= t_max_val
                t_grid = np.clip(t_grid, 0.0, 1.0)
                
        except Exception as e:
            print(f"   ! Terrain Error: {e}")

    # -----------------------------------------------------------
    # 2. Source Maps & Raw Metadata (Coordinate Fix)
    # -----------------------------------------------------------
    q_grid = np.zeros((Config.NY, Config.NX), dtype=np.float32)
    h_grid = np.zeros((Config.NY, Config.NX), dtype=np.float32)
    
    raw_source_metadata = [] 
    
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
            # xy_to_grid: 절대좌표 -> 그리드 인덱스 (0~44)
            idx = xy_to_grid(data['x'], data['y'])
            
            if idx:
                q_val = data.get('q', 1.0)
                h_val = data.get('h', 10.0)
                
                # 1. 모델용 Map 생성
                add_source_gaussian(q_grid, h_grid, idx[0], idx[1], q_val, h_val, sigma=1.0)
                
                # 2. [수정됨] 가시화용 좌표 저장 (Absolute -> Relative)
                # 절대 좌표를 저장하면 4,000,000 같이 큰 값이 저장되어 그래프가 망가짐
                # 그리드 인덱스를 다시 미터 단위(Relative)로 변환하여 저장
                
                rel_x = float(idx[0]) * Config.DX # 예: 10 * 100 = 1000.0m
                rel_y = float(idx[1]) * Config.DY
                
                # 이제 rel_x, rel_y는 0 ~ 4500 범위의 값이 됨
                raw_source_metadata.append([
                    rel_x,  
                    rel_y,  
                    h_val,      
                    q_val       
                ])
                cnt += 1
    
    print(f"   -> Mapped {cnt} sources (Fixed: Saved as relative coordinates).")

    # [데이터 정규화]
    q_grid = np.log1p(q_grid) 
    
    domain_max_z = Config.MAX_Z
    if domain_max_z > 0:
        h_grid /= domain_max_z 

    # 3. 저장
    save_path = os.path.join(Config.PROCESSED_DIR, Config.SAVE_MAPS)
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
    
    np.savez_compressed(save_path, 
                        terrain=t_grid, 
                        source_q=q_grid, 
                        source_h=h_grid,
                        terrain_max=t_max_val,
                        raw_sources=np.array(raw_source_metadata, dtype=object))
    
    print(f"   >> Saved integrated maps to {save_path}")

if __name__ == "__main__":
    run()