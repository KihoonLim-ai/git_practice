"""
Test visualization with dummy data
더미 데이터를 생성하여 시각화 기능이 제대로 작동하는지 테스트
"""
import os
import numpy as np

print("=" * 70)
print("🧪 Creating dummy inference results for testing visualization")
print("=" * 70)

# Create output directory
output_dir = "inference_results_no_wind"
os.makedirs(output_dir, exist_ok=True)

# Generate dummy data
np.random.seed(42)

num_samples = 5
nz, ny, nx = 21, 45, 45

print(f"\n📦 Generating {num_samples} dummy samples...")

# Create realistic-looking dummy data
predictions = []
targets = []
inputs = []
mse_list = []
mae_list = []
pcc_list = []

for i in range(num_samples):
    # Generate spatial pattern (Gaussian-like)
    y, x = np.ogrid[0:ny, 0:nx]
    cy, cx = np.random.randint(10, 35, 2)

    # Ground truth: Gaussian plume
    target_2d = np.exp(-((x - cx)**2 + (y - cy)**2) / 200.0)

    # Add height decay
    z_decay = np.linspace(1.0, 0.3, nz).reshape(nz, 1, 1)
    target_3d = target_2d[np.newaxis, :, :] * z_decay
    target_3d = target_3d * np.random.uniform(0.5, 2.0)  # Random scaling

    # Prediction: Similar but with noise
    pred_3d = target_3d + np.random.normal(0, 0.1 * target_3d.max(), target_3d.shape)
    pred_3d = np.maximum(pred_3d, 0)  # Non-negative

    # Input: Terrain and Source
    terrain = np.random.rand(nz, ny, nx) * 0.3  # Terrain mask
    source = np.zeros((nz, ny, nx))
    source[:, cy-2:cy+2, cx-2:cx+2] = np.random.uniform(0.5, 1.0)  # Source location

    inp_3d = np.stack([terrain, source], axis=0)  # (2, 21, 45, 45)

    # Calculate metrics
    mse = np.mean((pred_3d - target_3d) ** 2)
    mae = np.mean(np.abs(pred_3d - target_3d))

    pred_flat = pred_3d.flatten()
    target_flat = target_3d.flatten()
    pcc = np.corrcoef(pred_flat, target_flat)[0, 1]

    # Store
    predictions.append(pred_3d)
    targets.append(target_3d)
    inputs.append(inp_3d)
    mse_list.append(mse)
    mae_list.append(mae)
    pcc_list.append(pcc)

    print(f"   Sample {i}: MSE={mse:.6f}, MAE={mae:.6f}, PCC={pcc:.4f}")

# Convert to arrays
predictions = np.array(predictions)
targets = np.array(targets)
inputs = np.array(inputs)
mse_list = np.array(mse_list)
mae_list = np.array(mae_list)
pcc_list = np.array(pcc_list)

# Save
save_path = os.path.join(output_dir, "predictions.npz")
np.savez_compressed(
    save_path,
    predictions=predictions,
    targets=targets,
    inputs=inputs,
    mse=mse_list,
    mae=mae_list,
    pcc=pcc_list
)

print(f"\n✅ Saved dummy data to: {save_path}")
print(f"   Shapes:")
print(f"     predictions: {predictions.shape}")
print(f"     targets: {targets.shape}")
print(f"     inputs: {inputs.shape}")

# Save summary
import json
summary = {
    'num_samples': num_samples,
    'mean_mse': float(mse_list.mean()),
    'std_mse': float(mse_list.std()),
    'mean_mae': float(mae_list.mean()),
    'std_mae': float(mae_list.std()),
    'mean_pcc': float(pcc_list.mean()),
    'std_pcc': float(pcc_list.std()),
}

summary_path = os.path.join(output_dir, "summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Saved summary to: {summary_path}")

print("\n" + "=" * 70)
print("🎉 Dummy data created! Now you can run:")
print("   python visualize_no_wind.py")
print("=" * 70)
