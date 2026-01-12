# Self-Supervised Learning without Wind Data

## Overview

이 실험은 **바람/기상 데이터 없이** 정적 지도(Terrain + Source)만으로 농도를 예측할 수 있는지 테스트합니다.

### 가설
- ✅ **학습 가능**: 오염원 위치와 지형 정보만으로도 공간적 패턴 학습 가능
- ❌ **한계점**: 바람에 의한 이류(advection) 효과는 모델링 불가능

---

## Architecture Comparison

### Original Model (Full)
```
Input: (B, 5, 21, H, W) [Terrain, Source, U, V, W]
       + Met Sequence (B, 30, 3) [U_surf, V_surf, 1/L]
       + Global Wind (B, 2)

Output: Wind (B, N, 3) + Concentration (B, N, 1)
```

### Simplified Model (No Wind)
```
Input: (B, 2, 21, H, W) [Terrain, Source]

Output: Concentration (B, N, 1)
```

**Key Differences:**
- ❌ No TransformerObsBranch (met encoder)
- ❌ No Wind Prediction Head
- ✅ Direct Map → Trunk → Conc prediction
- 📉 ~50% fewer parameters

---

## Files Created

### 1. Model Architecture
**`physics_network/model/model_no_wind.py`**
- `SimplifiedDeepONet`: 바람 없이 농도만 예측하는 모델
- Components:
  - `Conv3dBranchSimple`: 2채널 입력 (Terrain + Source)
  - `SpatioTemporalTrunk`: 좌표 인코더 (원본과 동일)
  - Single concentration head

### 2. Dataset
**`physics_network/dataset/dataset_no_wind.py`**
- `AermodDatasetNoWind`: 바람 데이터 제외된 데이터셋
- Returns:
  - `input_vol`: (2, 21, H, W)
  - `target_conc`: (1, 21, H, W)
- No wind field caching, no met sequence

### 3. Training Script
**`physics_network/train/train_no_wind.py`**
- Simple training loop
- Loss:
  - MSE (concentration value accuracy)
  - PCC (spatial pattern correlation)
- WandB project: `KARI_NoWind_Baseline`

---

## How to Run

### Step 1: Verify Data Files Exist
```bash
ls physics_network/processed_data/
# Should see:
#   input_maps.npz    ✅
#   labels_conc.npz   ✅
#   input_met.npz     (not used)
```

### Step 2: Run Training
```bash
cd physics_network/train
python train_no_wind.py
```

### Step 3: Monitor Training
- WandB dashboard: `KARI_NoWind_Baseline/SimplifiedDeepONet_v1`
- Checkpoints saved to: `physics_network/train/checkpoints_no_wind/`

---

## Expected Results

### What the Model Can Learn
1. **Spatial Correlation**: 오염원 근처 = 높은 농도
2. **Terrain Effects**: 지형 차폐 효과 (계곡/산 영향)
3. **Source Strength**: 배출량에 비례한 농도 패턴

### What the Model Cannot Learn
1. **Wind Transport**: 바람에 의한 이류 방향
2. **Temporal Dynamics**: 기상 변화에 따른 농도 변화
3. **Dispersion Patterns**: 안정도(Monin-Obukhov Length)에 따른 확산

---

## Performance Metrics

### Primary Metrics
- **MSE Loss**: 농도 값 정확도
- **Pearson Correlation (PCC)**: 공간 분포 패턴 유사도

### Expected Performance
- Training MSE: ~0.5-1.0 (normalized concentration)
- Validation PCC: ~0.6-0.7 (moderate spatial correlation)
- **비교**: Full model (with wind) PCC > 0.85

---

## Comparison with Full Model

| Metric | No Wind Model | Full Model (with wind) |
|--------|---------------|------------------------|
| Input Channels | 2 (Terrain, Source) | 5 (+ U, V, W) + Met Seq |
| Parameters | ~500K | ~1M |
| Training Speed | 2x faster | Baseline |
| MSE Loss | Higher ⬆️ | Lower ⬇️ |
| PCC (Pattern) | ~0.6-0.7 | ~0.85+ |
| Physical Realism | Low | High |

---

## Configuration

Edit `TrainConfig` in `train_no_wind.py`:

```python
class TrainConfig:
    EPOCHS = 100          # Number of training epochs
    BATCH_SIZE = 32       # Batch size (adjust for GPU memory)
    LEARNING_RATE = 1e-4  # Adam learning rate

    LAMBDA_MSE = 1.0      # MSE loss weight
    LAMBDA_PCC = 0.5      # PCC loss weight

    CROP_SIZE = 32        # Training crop size (val/test = 45)
```

---

## Debugging

### Common Issues

**1. Import Error: `dataset.config_param`**
```bash
# Make sure you run from train/ directory
cd physics_network/train
python train_no_wind.py
```

**2. GPU Out of Memory**
```python
# Reduce batch size in train_no_wind.py
BATCH_SIZE = 16  # or 8
```

**3. Data Not Found**
```bash
# Re-run preprocessing if needed
cd physics_network/dataset
python main.py
```

**4. WandB Login Required**
```bash
wandb login
# Or disable WandB:
# Comment out wandb.init() in train_no_wind.py
```

---

## Next Steps

### 1. Analyze Results
- Compare loss curves: No Wind vs Full Model
- Visualize predictions: Where does it fail?
- Check PCC per altitude layer

### 2. Ablation Studies
- Try different `fourier_scale` values (5.0, 10.0, 20.0)
- Adjust `latent_dim` (64, 128, 256)
- Test with/without PCC loss

### 3. Add Pseudo-Wind Estimation
```python
# Idea: Estimate wind direction from source→concentration gradients
# ∇C ≈ -u·∇ (advection equation)
wind_estimate = -grad(concentration, coords)
```

### 4. Multi-Task Learning
- Predict both concentration AND wind jointly
- Use concentration as supervision signal for wind

---

## Citation

If you use this simplified model:

```
This baseline model tests concentration prediction without wind data,
demonstrating the importance of meteorological forcing in atmospheric
dispersion modeling.
```

---

## Contact

For questions about this experiment, check:
- Original model: `physics_network/model/model.py`
- Full training: `physics_network/train/train_conc.py`
- Data pipeline: `physics_network/dataset/dataset.py`
