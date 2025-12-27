import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset.dataset import get_time_split_datasets
from dataset.config_param import ConfigParam as Config

def check_dataset():
    print("=== Dataset Integrity Check Started (Updated for ST-Transformer) ===\n")

    # 1. 데이터셋 로드 테스트 (Helper Function 사용)
    print("--- [1] Loading Dataset ---")
    try:
        # seq_len=30, pred_step=5 설정으로 로드
        train_ds, val_ds, _ = get_time_split_datasets(seq_len=30, pred_step=5)
        print(f"✅ Train Dataset loaded. Total samples: {len(train_ds)}")
        print(f"✅ Val Dataset loaded.   Total samples: {len(val_ds)}")
        
        # 테스트는 Train Dataset으로 진행
        ds = train_ds
    except Exception as e:
        print(f"❌ Dataset initialization failed: {e}")
        return

    # 2. 단일 샘플 형상(Shape) 검증
    print("\n--- [2] Single Sample Shape Check ---")
    
    try:
        # ctx, met_seq, coords, wind, conc
        data = ds[0]
        ctx, met_seq, coords, wind, conc = data
    except Exception as e:
        print(f"❌ __getitem__ failed: {e}")
        return

    # 예상되는 차원 계산
    n_points = Config.NX * Config.NY * Config.NZ 
    print(f"Expected Grid Points: {n_points} ({Config.NX}x{Config.NY}x{Config.NZ})")
    
    # --- Shape 출력 및 검증 ---
    
    # 1. Context (Map): (3, 45, 45) -> Terrain, Q, H
    print(f"1. Context Map       : {tuple(ctx.shape)} \t-> Expected: (3, {Config.NY}, {Config.NX})")
    if ctx.shape[0] != 3: print("   ⚠️ Warning: Context channels mismatch!")

    # 2. Met Sequence (Transformer Input): (Seq, 3) -> (30, 3)
    print(f"2. Met Sequence      : {tuple(met_seq.shape)} \t-> Expected: (30, 3)")
    if met_seq.shape[0] != 30: print("   ⚠️ Warning: Sequence length mismatch!")

    # 3. Coordinates (Trunk Input): (N, 4) -> x, y, z, t
    print(f"3. Coordinates (4D)  : {tuple(coords.shape)} \t-> Expected: ({n_points}, 4)")
    if coords.shape[1] != 4: print("   ❌ Error: Coordinates must be 4D (x,y,z,t)!")

    # 4. Wind Label: (N, 3) -> u, v, w
    print(f"4. Wind GT (u,v,w)   : {tuple(wind.shape)} \t-> Expected: ({n_points}, 3)")

    # 5. Conc Label: (N, 1) -> concentration
    print(f"5. Conc GT           : {tuple(conc.shape)} \t-> Expected: ({n_points}, 1)")


    # 3. 배치 로딩 및 값 유효성(Sanity) 검증
    print("\n--- [3] Batch Loading & Value Sanity Check ---")
    batch_size = 4
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    try:
        batch = next(iter(dl))
        b_ctx, b_met, b_coords, b_wind, b_conc = batch
        
        print(f"✅ Batch loading successful (Batch size: {batch_size})")
        
        # NaN / Inf 체크
        tensors = {
            "Context": b_ctx, "Met_Seq": b_met, 
            "Coords": b_coords, "Wind_GT": b_wind, "Conc_GT": b_conc
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
            
        # --- 핵심 검증 포인트 ---
        
        # A. 농도 데이터 정규화 확인 (Z-score)
        # 음수 값이 존재해야 정상입니다.
        c_min = b_conc.min().item()
        c_max = b_conc.max().item()
        c_mean = b_conc.mean().item()
        print(f"\n[Check A] Concentration Normalization (Log + Z-score)")
        print(f"   > Min: {c_min:.4f}, Max: {c_max:.4f}, Mean: {c_mean:.4f}")
        if c_min < 0:
            print("   ✅ Valid: Negative values found (Standardization applied).")
        else:
            print("   ⚠️ Warning: All values are positive. Check process_labels.py!")

        # B. 기상 시계열 정규화 확인
        # 0~1 사이 혹은 그 근처값이어야 함
        m_max = b_met.max().item()
        print(f"\n[Check B] Met Sequence Normalization")
        print(f"   > Max value in Met Seq: {m_max:.4f}")
        if m_max > 10.0:
            print("   ⚠️ Warning: Met values seem too large (>10). Check process_met.py!")
        else:
            print("   ✅ Met values look normalized.")

        # C. 4D 좌표 확인 (Time channel)
        # 현재 t=0으로 설정했으므로 4번째 채널은 모두 0이어야 함
        t_vals = b_coords[:, :, 3] # (B, N)
        if torch.all(t_vals == 0):
             print("\n[Check C] Time Channel (4th dim)")
             print("   ✅ Valid: All time coordinates are 0.0 (Current prediction).")
        else:
             print("\n[Check C] Time Channel (4th dim)")
             print("   ⚠️ Warning: Non-zero time values found (Did you intend future prediction?).")

        # D. 수직풍(W) 존재 여부
        w_values = b_wind[:, :, 2]
        max_w = torch.max(torch.abs(w_values)).item()
        print(f"\n[Check D] Vertical Wind (W)")
        print(f"   > Max |W|: {max_w:.6f}")
        if max_w > 1e-6:
            print("   ✅ W component is active.")
        else:
            print("   ⚠️ Warning: W component is zero (Check Slope calculation).")

    except Exception as e:
        print(f"❌ Batch check failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_dataset()