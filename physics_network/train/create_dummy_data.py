import os
import numpy as np
import json

# Create directory
os.makedirs("inference_results_no_wind", exist_ok=True)

# Generate dummy data
np.random.seed(42)
num_samples = 5

predictions = np.random.rand(num_samples, 21, 45, 45) * 0.5
targets = predictions + np.random.randn(num_samples, 21, 45, 45) * 0.1
targets = np.maximum(targets, 0)

# Inputs: terrain and source
terrain = np.random.rand(num_samples, 1, 21, 45, 45) * 0.3
source = np.zeros((num_samples, 1, 21, 45, 45))
source[:, :, :, 20:25, 20:25] = 1.0
inputs = np.concatenate([terrain, source], axis=1)

# Metrics
mse = np.random.rand(num_samples) * 0.1
mae = np.random.rand(num_samples) * 0.05
pcc = 0.6 + np.random.rand(num_samples) * 0.2

# Save
np.savez_compressed(
    "inference_results_no_wind/predictions.npz",
    predictions=predictions,
    targets=targets,
    inputs=inputs,
    mse=mse,
    mae=mae,
    pcc=pcc
)

# Summary
summary = {
    "num_samples": num_samples,
    "mean_mse": float(mse.mean()),
    "std_mse": float(mse.std()),
    "mean_mae": float(mae.mean()),
    "std_mae": float(mae.std()),
    "mean_pcc": float(pcc.mean()),
    "std_pcc": float(pcc.std())
}

with open("inference_results_no_wind/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Dummy data created successfully!")
