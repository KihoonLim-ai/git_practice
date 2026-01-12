# config_param.py
import os


class ConfigParam:
    # ==========================================
    # 1. 그리드 및 물리 설정 (울산 산단 로그 기반)
    # [사용자 설정 유지: 변경 없음]
    # ==========================================
    NX, NY, NZ = 45, 45, 21
    DX, DY, DZ = 100.0, 100.0, 10.0

    # 전체 도메인 물리 크기
    MAX_X = NX * DX
    MAX_Y = NY * DY
    MAX_Z = (NZ - 1) * DZ 

    # [🚨 핵심 수정] AERMAP 로그의 DOMAINXY 값으로 정확히 일치시킴
    X_ORIGIN = 529800.0
    Y_ORIGIN = 3919200.0
    
    # ==========================================
    # 2. 경로 및 파일 관리 (Windows 호환 수정)
    # ==========================================
    
    # [수정 1] 현재 파일 위치를 기준으로 프로젝트 루트(kari-onestop-uas) 자동 찾기
    # 구조: kari-onestop-uas / physics_network / dataset / config_param.py
    _CURRENT_FILE = os.path.abspath(__file__)
    _DATASET_DIR  = os.path.dirname(_CURRENT_FILE)        # .../dataset
    _PHYSNET_DIR  = os.path.dirname(_DATASET_DIR)         # .../physics_network
    BASE_DIR      = os.path.dirname(_PHYSNET_DIR)         # .../kari-onestop-uas (루트)

    # [수정 2] 절대 경로 생성 (os.path.join 사용)
    RAW_DIR       = os.path.join(BASE_DIR, 'epa_sim', 'data')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'physics_network', 'processed_data')
    
    # 결과 파일(.plt)이 있는 폴더 (폴더가 없다면 미리 생성 필요)
    PLT_DIR_NAME  = os.path.join(BASE_DIR, 'epa_sim', 'data', 'conc')
    
    # 파일명 (운영체제에 맞게 경로 구분자 자동 처리)
    FILE_ROU     = os.path.join('ter', 'ulsan_terrain.rou')
    FILE_INP     = os.path.join('mod', 'aermod.inp')
    FILE_SRC_LOC = os.path.join('ter', 'ulsan_source_elev.src')
    FILE_SFC     = os.path.join('met', 'ulsan_2024.sfc')

    # [추가 제안] 만약 코드에서 PFL 파일을 찾는다면 아래 변수 사용 가능
    # (이전 에러 로그에서 기상 데이터를 못 찾는 문제가 있었음)
    FILE_PFL     = os.path.join('met', 'ulsan_2024.pfl')
    
    SAVE_MAPS = 'input_maps.npz'
    SAVE_MET  = 'input_met.npz'
    SAVE_LBL  = 'labels_conc.npz'
    
    PLT_FMT = 'ulsan_conc_1hr_{z:06d}m.plt'
    PLT_SKIP_ROWS = 8
    
    # ==========================================
    # 3. 데이터 필터링 및 파싱 옵션
    # [사용자 설정 유지: 변경 없음]
    # ==========================================
    TARGET_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    IDX_YEAR  = 0
    IDX_MONTH = 1
    IDX_DAY   = 2
    IDX_HOUR  = 4
    IDX_L     = 11
    IDX_WS    = 15
    IDX_WD    = 16

    @classmethod
    def make_dirs(cls):
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)
        print(f"[Config] Processed Data Directory: {cls.PROCESSED_DIR}")

# class ConfigParam:
#     # ==========================================
#     # 1. 그리드 및 물리 설정 (울산 산단 로그 기반)
#     # ==========================================
#     NX, NY, NZ = 45, 45, 21
#     DX, DY, DZ = 100.0, 100.0, 10.0

#     # 전체 도메인 물리 크기
#     MAX_X = NX * DX
#     MAX_Y = NY * DY
#     MAX_Z = (NZ - 1) * DZ 

#     # [🚨 핵심 수정] AERMAP 로그의 DOMAINXY 값으로 정확히 일치시킴
#     # 기존: 529839.8, 3919252.5 (X) -> 오차 발생 원인
#     # 수정: 529800.0, 3919200.0 (O) -> AERMOD 격자 원점
#     X_ORIGIN = 529800.0
#     Y_ORIGIN = 3919200.0
    
#     # ==========================================
#     # 2. 경로 및 파일 관리
#     # ==========================================
#     # (나머지 경로는 그대로 유지)
#     RAW_DIR = '/home/jhlee/kari-onestop-uas/epa_sim/data'
#     PROCESSED_DIR = '/home/jhlee/kari-onestop-uas/physics_network/processed_data'
#     PLT_DIR_NAME = '/home/jhlee/epa_sim/data/conc'
    
#     FILE_ROU = 'ter/ulsan_terrain.rou'
#     FILE_INP = 'mod/aermod.inp'
#     FILE_SRC_LOC = 'ter/ulsan_source_elev.src'
#     FILE_SFC = 'met/ulsan_2024.sfc'
    
#     SAVE_MAPS = 'input_maps.npz'
#     SAVE_MET  = 'input_met.npz'
#     SAVE_LBL  = 'labels_conc.npz'
    
#     PLT_FMT = 'ulsan_conc_1hr_{z:06d}m.plt'
#     PLT_SKIP_ROWS = 8
    
#     # ==========================================
#     # 3. 데이터 필터링 및 파싱 옵션
#     # ==========================================
#     TARGET_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
#     IDX_YEAR  = 0
#     IDX_MONTH = 1
#     IDX_DAY   = 2
#     IDX_HOUR  = 4
#     IDX_L     = 11
#     IDX_WS    = 15
#     IDX_WD    = 16

#     @classmethod
#     def make_dirs(cls):
#         os.makedirs(cls.PROCESSED_DIR, exist_ok=True)