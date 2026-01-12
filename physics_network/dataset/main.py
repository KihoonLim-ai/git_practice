import os
import sys

# 현재 디렉토리 경로 설정 (모듈 import 문제 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config_param import ConfigParam as Config
from dataset.process_maps import run as run_maps
from dataset.process_met import run as run_met
from dataset.process_labels import run as run_labels

def main():
    # 0. 폴더 생성 및 초기화
    Config.make_dirs()
    print("=== DeepONet Preprocessing Pipeline Started ===\n")
    
    # -----------------------------------------------------
    # 1. Maps 생성 (지형 + 오염원)
    # -----------------------------------------------------
    # - 지형 데이터(ROU) 로드 및 정규화
    # - 오염원 데이터(INP/SRC) 로드
    # - 가우시안 스플래팅(Gaussian Splatting) 적용
    # - 가시화용 메타데이터(raw_sources) 저장
    # -----------------------------------------------------
    run_maps()
    print("-" * 60)
    
    # -----------------------------------------------------
    # 2. Meteorology 생성 (기상 입력)
    # -----------------------------------------------------
    # - METEOR.DBG 파싱 (Full Profile)
    # - 21개 고도(0~200m) 선형 보간
    # - 안정도(L) -> 역수(1/L) 변환 (물리적 연속성 확보)
    # - 결과 저장: input_met.npz
    # -----------------------------------------------------
    run_met() # [변경] 반환값(met_df) 없음
    print("-" * 60)
    
    # -----------------------------------------------------
    # 3. Labels 생성 (농도 정답)
    # -----------------------------------------------------
    # - METEOR.DBG에서 타임스탬프 직접 추출 (Sync)
    # - PLT 파일(고도별 농도) 파싱
    # - Log1p 변환 및 Standard Scaling 적용
    # - 결과 저장: labels_conc.npz
    # -----------------------------------------------------
    run_labels() # [변경] 인자(met_df) 필요 없음
    
    print("\n=== All Done! Data ready in './processed_data' ===")

if __name__ == "__main__":
    main()