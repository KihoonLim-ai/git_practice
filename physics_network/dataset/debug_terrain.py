# debug_terrain.py
import os
import pandas as pd
from config_param import ConfigParam as Config

def check_coordinates():
    rou_path = os.path.join(Config.RAW_DIR, Config.FILE_ROU)
    print(f"Checking file: {rou_path}")
    
    if not os.path.exists(rou_path):
        print("File not found!")
        return

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    valid_count = 0
    
    # 파일 내용을 직접 조금만 찍어보기
    print("\n[Head of File (First 5 valid lines)]")
    with open(rou_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split()
            # 숫자로 변환 가능한 줄만 체크
            if len(parts) >= 3:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                    
                    if valid_count < 5:
                        print(f"Line: {line.strip()} -> X={x}, Y={y}, Z={z}")
                    valid_count += 1
                except:
                    continue

    print(f"\n[Coordinate Range Analysis]")
    print(f"File Min X: {min_x} ~ Max X: {max_x}")
    print(f"File Min Y: {min_y} ~ Max Y: {max_y}")
    
    print("\n[Config Settings]")
    print(f"Config Origin X: {Config.X_ORIGIN}")
    print(f"Config Origin Y: {Config.Y_ORIGIN}")
    print(f"Grid Width : {Config.NX * Config.DX} meters")
    print(f"Grid Height: {Config.NY * Config.DY} meters")
    
    # 범위 체크
    limit_x = Config.X_ORIGIN + (Config.NX * Config.DX)
    limit_y = Config.Y_ORIGIN + (Config.NY * Config.DY)
    
    print(f"\n[Validity Check]")
    if min_x > limit_x or max_x < Config.X_ORIGIN:
        print("❌ CRITICAL: X coordinates do not overlap with Config Grid!")
    else:
        print("✅ X coordinates look overlapping.")
        
    if min_y > limit_y or max_y < Config.Y_ORIGIN:
        print("❌ CRITICAL: Y coordinates do not overlap with Config Grid!")
    else:
        print("✅ Y coordinates look overlapping.")

if __name__ == "__main__":
    check_coordinates()