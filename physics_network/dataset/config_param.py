# config_param.py
import os

class ConfigParam:
    # ==========================================
    # 1. 그리드 및 물리 설정 (울산 산단 로그 기반)
    # ==========================================
    NX, NY, NZ = 45, 45, 21
    DX, DY, DZ = 100.0, 100.0, 10.0

    # [중요] 전체 도메인 물리 크기
    MAX_X = NX * DX
    MAX_Y = NY * DY
    MAX_Z = (NZ - 1) * DZ # 200m

    # [확인된 원점 좌표: 좌하단]
    X_ORIGIN = 529839.8
    Y_ORIGIN = 3919252.5
    
    # ==========================================
    # 2. 경로 및 파일 관리
    # ==========================================
    RAW_DIR = '/home/jhlee/kari-onestop-uas/epa_sim/data'           # 원본 데이터 폴더
    PROCESSED_DIR = '/home/jhlee/kari-onestop-uas/physics_network/processed_data' # 결과 저장 폴더
    PLT_DIR_NAME = '/home/jhlee/epa_sim/data/conc'             # raw_data/plt (PLT 파일들 모아둔 곳)
    
    # 원본 파일명 (사용자 파일명과 일치해야 함)
    FILE_ROU = 'ter/ulsan_terrain.rou'         # 지형
    FILE_INP = 'mod/aermod.inp'                # 오염원 파라미터(SRCPARAM)
    FILE_SRC_LOC = 'ter/ulsan_source_elev.src' # 오염원 위치(LOCATION) - 외부 파일
    FILE_SFC = 'met/ulsan_2024.sfc'                # 기상
    
    # 결과 저장 파일명
    SAVE_MAPS = 'input_maps.npz'
    SAVE_MET  = 'input_met.npz'
    SAVE_LBL  = 'labels_conc.npz'
    
    # [중요] PLT 파일 포맷 설정
    # 파일명 패턴: (예: conc_24060101_z00.plt) -> 본인이 생성한 규칙에 맞게 수정 필수!
    PLT_FMT = 'ulsan_conc_1hr_{z:06d}m.plt'
    
    # [중요] PLT 헤더 스킵 줄 수 (제공해주신 파일 기준 8줄)
    PLT_SKIP_ROWS = 8
    
    # ==========================================
    # 3. 데이터 필터링 및 파싱 옵션
    # ==========================================
    TARGET_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 여름 데이터만 사용
    
    # AERMET SFC 파일 컬럼 인덱스 (0부터 시작)
    # Y(0), M(1), D(2), H(4), L(11), Ws(15), Wd(16)
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
