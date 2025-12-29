import os
import sys
import numpy as np

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.config_param import ConfigParam as Config

def fix_terrain_range():
    print("=== Fixing Terrain Range (Clipping 0~1) ===")
    
    p_dir = Config.PROCESSED_DIR
    map_path = os.path.join(p_dir, Config.SAVE_MAPS)
    
    # 1. 로드
    if not os.path.exists(map_path):
        print(f"❌ File not found: {map_path}")
        return

    data = np.load(map_path)
    terrain = data['terrain']
    # 다른 키들은 그대로 유지하기 위해 dict로 복사
    save_dict = {key: data[key] for key in data.files}
    
    print(f"Original Range: Min={terrain.min():.4f}, Max={terrain.max():.4f}")
    
    # 2. 보정 (Clipping)
    # 0보다 작은 값은 0으로, 1보다 큰 값은 1로 만듭니다.
    terrain_fixed = np.clip(terrain, 0.0, 1.0)
    
    # 수정된 데이터 덮어쓰기
    save_dict['terrain'] = terrain_fixed
    
    print(f"Fixed Range   : Min={terrain_fixed.min():.4f}, Max={terrain_fixed.max():.4f}")
    
    # 3. 저장
    np.savez(map_path, **save_dict)
    print(f"✅ Saved fixed maps to: {map_path}")

if __name__ == "__main__":
    fix_terrain_range()