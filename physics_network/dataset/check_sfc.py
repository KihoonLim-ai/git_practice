import pandas as pd
import os
import sys

# [설정] 확인하고 싶은 SFC 파일 경로
# 실제 경로로 수정해서 사용하세요
SFC_FILE_PATH = '/home/jhlee/kari-onestop-uas/epa_sim/data/met/ulsan_2024.sfc' 

def check_sfc_data():
    if not os.path.exists(SFC_FILE_PATH):
        print(f"❌ 파일을 찾을 수 없습니다: {SFC_FILE_PATH}")
        return

    print(f"📂 Reading SFC file: {SFC_FILE_PATH} ...")

    try:
        # 1. 데이터 로드 (첫 번째 줄은 헤더이므로 skip)
        # 구분자는 공백(regex='\s+'), 헤더 없음(header=None)
        df = pd.read_csv(SFC_FILE_PATH, sep=r'\s+', skiprows=1, header=None)
        
        # 2. 컬럼 매핑 (일반적인 AERMET SFC 포맷 기준)
        # 0: Year, 1: Month, 2: Day, 3: JulianDay, 4: Hour
        # 5: Sensible Heat Flux (H) -> 결측치 판단용으로 사용
        # 10: Monin-Obukhov Length (L) -> 결측치 판단용으로 사용
        
        # 전체 데이터 개수 (단순 줄 수)
        total_rows = len(df)
        
        # 3. 유효 데이터 필터링
        # 제공해주신 스니펫을 보면 결측치는 -999.0, -9.0, -99999.0 등으로 표시됨
        # 가장 확실한 건 'Sensible Heat Flux (Col 5)'나 'L (Col 10)'이 정상 범위인지 보는 것입니다.
        
        # 조건: 5번 컬럼(H)이 -900보다 크고, 10번 컬럼(L)이 -90000보다 큰 경우
        valid_df = df[ (df[5] > -900.0) & (df[10] > -90000.0) ]
        valid_count = len(valid_df)

        # 4. 결과 출력
        print("\n" + "="*40)
        print("   📊 SFC Data Check Result")
        print("="*40)
        
        print(f"1. Total Entries (Lines): {total_rows:,} hours")
        print(f"   (단순히 파일에 기록된 시간의 수)")
        
        print(f"2. Valid Physics Data   : {valid_count:,} hours")
        print(f"   (결측치 -999 등을 제외한 실제 학습 가능 데이터)")
        
        if valid_count > 0:
            print("-" * 40)
            # 날짜 범위 확인
            start_row = valid_df.iloc[0]
            end_row = valid_df.iloc[-1]
            
            s_yr, s_mo, s_dy, s_hr = int(start_row[0]), int(start_row[1]), int(start_row[2]), int(start_row[4])
            e_yr, e_mo, e_dy, e_hr = int(end_row[0]), int(end_row[1]), int(end_row[2]), int(end_row[4])
            
            print(f"📅 Valid Range:")
            print(f"   Start: 20{s_yr:02d}-{s_mo:02d}-{s_dy:02d} {s_hr:02d}h")
            print(f"   End  : 20{e_yr:02d}-{e_mo:02d}-{e_dy:02d} {e_hr:02d}h")
            
            # 유효 비율
            ratio = (valid_count / total_rows) * 100
            print(f"📈 Usable Ratio: {ratio:.2f}%")
            
            if valid_count < 2000:
                print("\n⚠️  Warning: 학습 데이터가 2,000시간 미만입니다. 과적합 위험이 있습니다.")
            else:
                print("\n✅  Info: 데이터 양은 충분해 보입니다.")
        else:
            print("\n❌ Error: 유효한 데이터가 하나도 없습니다. 결측치 기준을 확인하세요.")

    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    check_sfc_data()