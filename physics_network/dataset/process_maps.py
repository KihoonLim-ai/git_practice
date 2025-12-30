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

def add_source_gaussian(grid, center_x, center_y, q_val, sigma=1.0):
    """
    [핵심 기능] 오염원 가우시안 스플래팅
    - 한 점(Pixel)에 꽂히는 Q값을 주변 5x5 영역으로 부드럽게 분산시킴
    - CNN이 "점"이 아닌 "영역"으로 인식하여 학습 효율 증대
    - 질량 보존 법칙 적용 (커널 전체 합 = q_val)
    """
    # 중심 좌표 (정수형 격자 인덱스)
    cx, cy = int(center_x), int(center_y)
    
    # 커널 반경 (radius=2 -> 5x5 영역)
    r = 2
    
    # 임시 커널 생성
    kernel = np.zeros((2*r+1, 2*r+1), dtype=np.float32)
    
    # 가우시안 가중치 계산
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            dist_sq = dx**2 + dy**2
            weight = np.exp(-dist_sq / (2 * sigma**2))
            kernel[dy+r, dx+r] = weight
            
    # 커널 정규화 (전체 합이 1이 되도록 -> 총 배출량 Q 보존)
    if kernel.sum() > 0:
        kernel /= kernel.sum()
        
    # 그리드에 더하기 (경계 처리 포함)
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            ny, nx = cy + dy, cx + dx
            # 유효한 격자 범위 내에만 값 누적
            if 0 <= ny < Config.NY and 0 <= nx < Config.NX:
                # 커널 값 * 배출량 Q
                grid[ny, nx] += q_val * kernel[dy+r, dx+r]

def run():
    print("\n[Step 1] Processing Static Maps (Integrated with Clipping & Gaussian Splatting)...")
    
    # -----------------------------------------------------------
    # 1. Terrain Map (GRIDCART Format)
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
                
                # [지형 정규화] 0~1 scaling
                if t_max_val > 0: 
                    t_grid /= t_max_val
                
                # [통합된 수정 로직] Clipping (0.0 ~ 1.0 강제)
                # 데이터 오류나 보간 문제로 인한 이상치 제거
                t_grid = np.clip(t_grid, 0.0, 1.0)
                
                print(f"   -> Terrain processed & Clipped (Min:{t_grid.min():.2f}, Max:{t_grid.max():.2f}).")
                
        except Exception as e:
            print(f"   ! Terrain Error: {e}")

    # -----------------------------------------------------------
    # 2. Source Maps (Gaussian Splatting 적용)
    # -----------------------------------------------------------
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
    skipped = 0
    for sid, data in sources.items():
        if 'x' in data and 'y' in data:
            # 좌표 변환 (UTM -> Grid Index)
            # xy_to_grid 함수는 정수형 튜플 혹은 None 반환
            idx = xy_to_grid(data['x'], data['y'])
            
            if idx:
                q_val = data.get('q', 1.0)
                h_val = data.get('h', 10.0)
                
                # [수정됨] 단순 대입 대신 가우시안 스플래팅 함수 호출
                # q_grid[idx] += q_val  <-- 기존 방식 (삭제됨)
                add_source_gaussian(q_grid, idx[1], idx[0], q_val, sigma=1.0)
                
                # 높이(h)는 물리적으로 점(Point)이므로 Max Pooling 방식 유지
                # (주변으로 퍼지는 성질의 데이터가 아님)
                h_grid[idx] = max(h_grid[idx], h_val)
                
                cnt += 1
            else:
                skipped += 1
    
    print(f"   -> Mapped {cnt} sources (Skipped {skipped} outside grid).")

    # -----------------------------------------------------------
    # [데이터 정규화 및 로그 변환]
    # -----------------------------------------------------------
    
    # 1) 배출량(Q): Log Scale 적용
    # 가우시안 스플래팅으로 값이 퍼져도, 총량 보존 후 Log를 취하므로 분포가 훨씬 부드러워짐
    q_grid = np.log1p(q_grid) 
    print("   -> Applied Log1p scaling to Source Emission (Q)")

    # 2) 굴뚝 높이(H): 0~1 MinMax Scaling
    domain_max_z = Config.MAX_Z
    if domain_max_z > 0:
        h_grid /= domain_max_z
    print(f"   -> Applied MinMax scaling to Source Height (H) using max_z={domain_max_z}m")

    # -----------------------------------------------------------
    # 3. 저장 (Saving)
    # -----------------------------------------------------------
    save_path = os.path.join(Config.PROCESSED_DIR, Config.SAVE_MAPS)
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
    
    np.savez_compressed(save_path, 
                        terrain=t_grid, 
                        source_q=q_grid, 
                        source_h=h_grid,
                        terrain_max=t_max_val) # 복원용 메타데이터
    
    print(f"   >> Saved integrated maps to {save_path}")

if __name__ == "__main__":
    run()