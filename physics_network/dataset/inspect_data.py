import numpy as np
import os
import matplotlib.pyplot as plt

from dataset.config_param import ConfigParam as Config

# [설정] 확인하고 싶은 npz 파일 경로 (가장 첫 번째 파일 추천)
# 경로가 맞는지 꼭 확인해주세요!
FILE_PATH = os.path.join(Config.PROCESSED_DIR, Config.SAVE_MET)

def inspect_npz(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        # 폴더 내의 아무 npz 파일이나 하나 찾아서 대시함
        dir_path = os.path.dirname(file_path)
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.npz')]
            if files:
                file_path = os.path.join(dir_path, files[0])
                print(f"🔄 대체 파일 로드: {files[0]}")
            else:
                return
        else:
            return

    print(f"\n🔍 Inspecting: {os.path.basename(file_path)}")
    print("=" * 60)
    
    try:
        data = np.load(file_path)
        keys = data.files
        print(f"📂 포함된 키(Keys): {keys}")
        print("-" * 60)

        for key in keys:
            arr = data[key]
            print(f"🔑 Key: [{key}]")
            print(f"   > Shape : {arr.shape}")
            print(f"   > Dtype : {arr.dtype}")
            
            # 수치형 데이터인 경우 통계 출력
            if np.issubdtype(arr.dtype, np.number):
                print(f"   > Min   : {arr.min():.4f}")
                print(f"   > Max   : {arr.max():.4f}")
                print(f"   > Mean  : {arr.mean():.4f}")
                
                # [중요] 좌표 데이터인지 확인 (이름에 coord, pos, points 등이 포함되면)
                if 'coord' in key or 'pos' in key or 'points' in key:
                    print(f"   🚨 [좌표 점검] Max 값이 1.0을 넘나요? -> {'YES (정규화 필요)' if arr.max() > 1.5 else 'NO (정규화 된듯)'}")
                    # 샘플 출력
                    print(f"   > Sample[0]: {arr[0]}")
            
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ 읽기 오류 발생: {e}")

if __name__ == "__main__":
    # 1. 파일 경로가 정확한지 확인하고 실행하세요
    # 보통 processed_data 폴더 안에 train_x.npz 형태로 있을 겁니다.
    target_file = os.path.join(Config.PROCESSED_DIR, Config.SAVE_MET) # 실제 파일명으로 수정 필요
    
 
    inspect_npz(target_file)