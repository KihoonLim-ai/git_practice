# check_dataset.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import AermodDataset
from config_param import ConfigParam as Config

def check_dataset():
    print("=== Dataset Integrity Check Started ===\n")

    # 1. 데이터셋 로드 테스트
    try:
        ds = AermodDataset(mode='train')
        print(f"✅ Dataset initialized. Total samples: {len(ds)}")
    except Exception as e:
        print(f"❌ Dataset initialization failed: {e}")
        return

    # 2. 단일 샘플 형상(Shape) 검증
    print("\n--- [1] Single Sample Shape Check ---")
    
    # 0번 인덱스 데이터 가져오기
    try:
        # ctx, met, coords, wind, conc
        data = ds[0]
        ctx, met, coords, wind, conc = data
    except Exception as e:
        print(f"❌ __getitem__ failed: {e}")
        return

    # 예상되는 차원 계산
    # Grid Point 수 = NX * NY * NZ
    n_points = Config.NX * Config.NY * Config.NZ 

    print(f"Expected Grid Points: {n_points} ({Config.NX}x{Config.NY}x{Config.NZ})")
    
    # Shape 출력 및 검증
    # Context (Branch Input 1): 지형, Source Q, Source H -> (3, 45, 45)
    print(f"1. Context Map (Terrain/Src) : {tuple(ctx.shape)} \t-> Expected: (3, {Config.NY}, {Config.NX})")
    
    # Global Met (Branch Input 2): u_ref, v_ref, L -> (3,)
    print(f"2. Global Met (u, v, L)      : {tuple(met.shape)} \t\t-> Expected: (3,)")
    
    # Coords (Trunk Input): x, y, z -> (N_points, 3)
    print(f"3. Coordinates (Trunk)       : {tuple(coords.shape)} \t-> Expected: ({n_points}, 3)")
    
    # Wind Label (Output 1): u, v, w -> (N_points, 3)
    print(f"4. Wind Label (u, v, w)      : {tuple(wind.shape)} \t-> Expected: ({n_points}, 3)")
    
    # Conc Label (Output 2): concentration -> (N_points, 1)
    print(f"5. Conc Label (C)            : {tuple(conc.shape)} \t-> Expected: ({n_points}, 1)")

    # 3. 배치 로딩 및 값 유효성(Sanity) 검증
    print("\n--- [2] Batch Loading & Value Sanity Check ---")
    batch_size = 4
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    # 첫 번째 배치만 뽑아서 확인
    try:
        batch = next(iter(dl))
        b_ctx, b_met, b_coords, b_wind, b_conc = batch
        
        print(f"✅ Batch loading successful (Batch size: {batch_size})")
        
        # NaN / Inf 체크
        tensors = {
            "Context": b_ctx, "Met": b_met, 
            "Coords": b_coords, "Wind": b_wind, "Conc": b_conc
        }
        
        all_good = True
        for name, t in tensors.items():
            if torch.isnan(t).any():
                print(f"❌ [FAIL] {name} contains NaN values!")
                all_good = False
            if torch.isinf(t).any():
                print(f"❌ [FAIL] {name} contains Inf values!")
                all_good = False
                
        if all_good:
            print("✅ No NaN or Inf values found.")
            
        # 4. W 값 존재 여부 확인 (매우 중요!)
        # Wind tensor shape: (Batch, N_points, 3) -> 3번째 채널이 W
        w_values = b_wind[:, :, 2] # W channel
        
        max_w = torch.max(torch.abs(w_values)).item()
        mean_w = torch.mean(torch.abs(w_values)).item()
        
        print(f"\n--- [3] Vertical Wind (W) Check ---")
        print(f"   > Max |W| in batch: {max_w:.6f}")
        print(f"   > Mean |W| in batch: {mean_w:.6f}")
        
        if max_w < 1e-6:
            print("⚠️ WARNING: W values are all near zero. Check 'dataset.py' scaling logic!")
        else:
            print("✅ W values look active (Non-zero).")

    except Exception as e:
        print(f"❌ Batch check failed: {e}")

if __name__ == "__main__":
    check_dataset()