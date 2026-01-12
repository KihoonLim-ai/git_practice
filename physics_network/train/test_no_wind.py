"""
간단한 테스트 스크립트: train_no_wind.py의 데이터 로딩 및 모델 초기화 테스트
"""
import os
import sys
import torch

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

print("=" * 70)
print("🧪 Testing No-Wind Training Setup")
print("=" * 70)

# Step 1: Import 테스트
print("\n[1/5] Testing imports...")
try:
    from dataset.dataset_no_wind import get_dataloaders_no_wind
    from dataset.physics_utils import make_batch_coords
    from dataset.config_param import ConfigParam as Config
    from model.model_no_wind import SimplifiedDeepONet
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Step 2: 데이터 로더 테스트
print("\n[2/5] Testing data loaders...")
try:
    train_loader, val_loader, test_loader = get_dataloaders_no_wind(
        batch_size=2,
        crop_size=32,
        num_workers=0
    )
    print(f"✅ Data loaders created")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
except Exception as e:
    print(f"❌ Data loader failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: 단일 배치 로딩 테스트
print("\n[3/5] Testing single batch loading...")
try:
    inp_vol, target_vol = next(iter(train_loader))
    print(f"✅ Batch loaded successfully")
    print(f"   Input shape: {inp_vol.shape}")
    print(f"   Target shape: {target_vol.shape}")
    print(f"   Input dtype: {inp_vol.dtype}")
    print(f"   Target dtype: {target_vol.dtype}")
except Exception as e:
    print(f"❌ Batch loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: 모델 초기화 테스트
print("\n[4/5] Testing model initialization...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")

    model = SimplifiedDeepONet(
        latent_dim=128,
        fourier_scale=10.0,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model initialized")
    print(f"   Total parameters: {total_params:,}")
except Exception as e:
    print(f"❌ Model initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Forward pass 테스트
print("\n[5/5] Testing forward pass...")
try:
    inp_vol = inp_vol.to(device)
    target_vol = target_vol.to(device)

    B, C, D, H, W = inp_vol.shape
    coords = make_batch_coords(B, D, H, W, device=device)

    print(f"   Coordinates shape: {coords.shape}")

    with torch.no_grad():
        pred_conc = model(inp_vol, coords)

    print(f"✅ Forward pass successful")
    print(f"   Prediction shape: {pred_conc.shape}")
    print(f"   Prediction range: [{pred_conc.min():.4f}, {pred_conc.max():.4f}]")

except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("🎉 All tests passed! Ready to train.")
print("=" * 70)
print("\nTo start training, run:")
print("  python train_no_wind.py")
