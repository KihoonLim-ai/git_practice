import os
import numpy as np
from dataset.config_param import ConfigParam as Config

# =========================================================
# 1. 파싱 및 보간 함수 (핵심 로직)
# =========================================================
def process_meteor_dbg_interpolated(file_path, target_heights):
    """
    METEOR.DBG를 읽어서 target_heights에 맞는 u, v 프로파일과 1/L을 추출
    
    [주요 기능]
    1. RECEPT 섹션 무시 (GRID HEIGHT 섹션만 사용)
    2. 선형 보간 (Linear Interpolation)으로 21개 고도 데이터 생성
    3. Monin-Obukhov Length (L) -> Inverse L (1/L) 변환 수행
    """
    print(f"   Parsing {file_path}...")
    
    raw_data = [] 
    current_time_data = {'heights': [], 'us': [], 'vs': []}
    current_L = 0.0
    
    is_profile_section = False # 프로파일 섹션 진입 여부 플래그
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        
        # [섹션 감지] GRID나 WIND가 보이면 프로파일 섹션으로 간주
        if "GRID" in line or ("WIND" in line and "HEIGHT" in line):
            is_profile_section = True
            continue
            
        # [섹션 종료] RECEPT(수용체)나 SURFACE 데이터는 무시
        if "RECEPT" in line or "SURFACE" in line:
            is_profile_section = False
            continue
            
        # (A) 날짜 라인 및 L값 파싱
        # 특징: 숫자로 시작 + 컬럼이 많음(>8) + 첫 컬럼이 연도
        # 예: 24 1 1 9 ... 또는 2024 1 1 9 ...
        if len(parts) > 8 and parts[0].isdigit():
            try:
                # 날짜 라인인지 확인 (정수 변환 테스트)
                _ = int(parts[0])
                
                # 날짜 라인이 맞다면, 이 줄에 L값(Monin-Obukhov Length)이 있음
                # METEOR.DBG 표준 헤더 기준 9번째 컬럼 (인덱스 8)
                l_cand = float(parts[8])
                
                # 이전 시간대 데이터가 모였으면 저장
                if current_time_data['heights']:
                    raw_data.append((current_time_data, current_L))
                    # 초기화
                    current_time_data = {'heights': [], 'us': [], 'vs': []}
                
                current_L = l_cand
                # 날짜 라인은 프로파일 데이터가 아니므로 다음 줄로
                continue
            except ValueError:
                pass # 날짜 라인이 아니면 아래 프로파일 파싱 로직으로 넘어감

        # (B) 프로파일 데이터 파싱 (GRID HEIGHT 섹션일 때만)
        # 포맷: 층인덱스(int) 고도(float) 풍향(float) 풍속(float) ...
        # 예: 1 0.0 247.0 0.01 ...
        if is_profile_section and len(parts) >= 7 and parts[0].isdigit():
            try:
                # 첫 번째가 인덱스, 두 번째가 고도인지 확인
                h = float(parts[1])  # 고도
                wd = float(parts[2]) # 풍향 (Degree)
                ws = float(parts[3]) # 풍속 (m/s)
                
                # 벡터 변환 (Meteorological -> Mathematical)
                # u = -ws * sin(theta), v = -ws * cos(theta)
                u = -ws * np.sin(np.deg2rad(wd))
                v = -ws * np.cos(np.deg2rad(wd))
                
                current_time_data['heights'].append(h)
                current_time_data['us'].append(u)
                current_time_data['vs'].append(v)
            except ValueError:
                continue

    # 파일 끝: 마지막 타임스텝 저장
    if current_time_data['heights']:
        raw_data.append((current_time_data, current_L))

    print(f"   -> Extracted {len(raw_data)} time steps.")
    if len(raw_data) == 0:
        raise ValueError("No data extracted! Check METEOR.DBG format.")

    # ---------------------------------------------------------
    # 보간 (Interpolation) 및 1/L 변환 수행
    # ---------------------------------------------------------
    final_met_seq = [] 
    final_inv_L_seq = []
    
    epsilon = 1e-1 # 0으로 나누기 방지용 작은 수
    
    for data, L_val in raw_data:
        # 1. 바람 프로파일 보간
        src_h = np.array(data['heights'])
        src_u = np.array(data['us'])
        src_v = np.array(data['vs'])
        
        # 고도순 정렬 (np.interp 필수 조건)
        sort_idx = np.argsort(src_h)
        src_h = src_h[sort_idx]
        src_u = src_u[sort_idx]
        src_v = src_v[sort_idx]
        
        # 선형 보간 수행
        interp_u = np.interp(target_heights, src_h, src_u)
        interp_v = np.interp(target_heights, src_h, src_v)
        
        layer_vectors = np.stack([interp_u, interp_v], axis=-1)
        final_met_seq.append(layer_vectors)
        
        # 2. L값 역수 변환 (Physics-informed Preprocessing)
        # 목적: 중립(Neutral, L->Inf)일 때의 특이점(Singularity) 제거
        # 결과: 중립 -> 0, 안정 -> 양수, 불안정 -> 음수
        
        # AERMOD 결측/코드 처리
        # -9000 이하: 결측, 5000 이상: 중립(8888 등) -> 모두 중립(0)으로 처리
        if abs(L_val) > 5000.0 or L_val <= -9000.0:
            inv_L = 0.0
        elif abs(L_val) < epsilon:
            # L이 0에 너무 가까우면(극단적 불안정/안정), 수치 안정성을 위해 제한
            inv_L = np.sign(L_val) * (1.0 / epsilon)
        else:
            inv_L = 1.0 / L_val
            
        final_inv_L_seq.append(inv_L)

    return np.array(final_met_seq, dtype=np.float32), np.array(final_inv_L_seq, dtype=np.float32)

# =========================================================
# 2. 메인 실행 함수
# =========================================================
def run():
    print("\n[Step 2] Processing Met Data (From METEOR.DBG Profile)...")
    
    # 1. 파일 경로 설정
    met_file_path = os.path.join(Config.RAW_DIR, "met", "METEOR.DBG") 
    
    if not os.path.exists(met_file_path):
        print(f"Error: {met_file_path} not found.")
        return

    # 2. 목표 고도 설정 (21개 층: 0m ~ 200m, 10m 간격)
    target_heights = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 
                      100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0]
    
    # 3. 데이터 로드 및 처리
    try:
        winds, inv_Ls = process_meteor_dbg_interpolated(met_file_path, target_heights)
    except ValueError as e:
        print(f"Error during parsing: {e}")
        return
    
    # 4. 정규화 (Normalization)
    # 바람: 전체 데이터 중 최대 풍속으로 나눔
    max_uv = np.max(np.abs(winds))
    if max_uv == 0: max_uv = 1.0
    
    # 1/L: 전체 데이터 중 최대 절대값으로 나눔
    max_inv_L = np.max(np.abs(inv_Ls))
    if max_inv_L == 0: max_inv_L = 1.0
    
    print(f"   -> Max Wind Speed (All Heights): {max_uv:.4f} m/s")
    print(f"   -> Max |1/L| (Stability Param): {max_inv_L:.4f} m^-1")
    
    winds_norm = winds / max_uv
    inv_Ls_norm = inv_Ls / max_inv_L
    
    # 5. 차원 변환 (Flattening)
    # winds: (Time, 21, 2) -> (Time, 42)
    # 10m 단위로 펼쳐서 모델에 시간축을 따라 입력할 준비
    winds_flat = winds_norm.reshape(winds_norm.shape[0], -1)
    
    # inv_L: (Time,) -> (Time, 1)
    inv_Ls_flat = inv_Ls_norm.reshape(-1, 1)
    
    # 병합: (Time, 43)
    # [u0, v0, u10, v10, ..., u200, v200, inv_L]
    met_final = np.hstack([winds_flat, inv_Ls_flat])
    
    # 6. 저장
    save_path = os.path.join(Config.PROCESSED_DIR, Config.SAVE_MET)
    np.savez_compressed(
        save_path, 
        met=met_final,    
        max_uv=max_uv,     
        max_L=max_inv_L   # 주의: 이제 max_L 키에는 '1/L의 최대값'이 저장됨
    )
    print(f"   >> Saved FULL-PROFILE met data to {save_path}")
    print(f"   >> Final Shape: {met_final.shape}")
    print("      (Note: Includes 21 wind layers + 1 inverse MO length)")

if __name__ == "__main__":
    run()