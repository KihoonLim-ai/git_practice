# Sequence-to-Sequence Concentration Prediction Model (No Wind)

## Overview

This model predicts future atmospheric pollutant concentration based on **past 30 hours of concentration data** without using explicit wind/meteorological data.

### Key Features
- **Input**: Past 30 timesteps of 3D concentration maps + Static maps (Terrain, Source)
- **Output**: Future 1 timestep of 3D concentration map
- **Architecture**: 3D CNN Encoder → Transformer → Decoder
- **No wind data required** (learns temporal patterns directly from concentration history)

---

## Architecture

```
Past Concentration (30, 21, 45, 45) → ConcentrationEncoder3D → (B, T, 128)
                                           ↓
                                    Transformer Encoder
                                           ↓
                                   Last Timestep Features → (B, 128)

Static Maps (2, 21, 45, 45) → StaticMapEncoder → (B, 128)

           ↓ Fusion (Concatenate + MLP) ↓

           Combined Features (B, 128)
                    ↓
           ConcentrationDecoder
                    ↓
      Future Concentration (1, 21, 45, 45)
```

### Components

1. **ConcentrationEncoder3D**: Processes each timestep through 3D CNN, extracts spatial-temporal features
2. **StaticMapEncoder**: Encodes terrain and emission source information
3. **TemporalTransformer**: Learns temporal dependencies across 30 timesteps
4. **ConcentrationDecoder**: Generates future 3D concentration map
5. **Fusion Layer**: Combines temporal and static context

---

## Files Created

### Model & Dataset
- `physics_network/model/model_seq2seq.py` - Seq2Seq model architecture
- `physics_network/dataset/dataset_seq2seq.py` - Dataset for temporal sequence loading

### Training & Inference
- `physics_network/train/train_seq2seq.py` - Training script
- `physics_network/train/inference_seq2seq.py` - Inference script
- `physics_network/train/visualize_seq2seq.py` - Visualization script

---

## Usage

### 1. Training

```bash
cd physics_network/train
python train_seq2seq.py
```

**Configuration** (edit `TrainConfig` in `train_seq2seq.py`):
- `SEQ_LEN = 30` - Past 30 timesteps as input
- `PRED_HORIZON = 1` - Predict 1 timestep ahead
- `CROP_SIZE = 32` - Training crop size (memory efficiency)
- `BATCH_SIZE = 8` - Adjust based on GPU memory
- `NUM_EPOCHS = 200`
- `LEARNING_RATE = 1e-4`

**Loss Function**:
- MSE (Mean Squared Error) - Pixel-wise accuracy
- PCC (Pearson Correlation) - Spatial pattern preservation
- Combined: `Loss = 1.0 * MSE + 0.5 * (1 - PCC)`

**Output**:
- Checkpoints saved to `checkpoints_seq2seq/best_seq2seq.pth`
- WandB logging (if available)

---

### 2. Inference

```bash
python inference_seq2seq.py
```

**Output** (`inference_results_seq2seq/`):
- `predictions.npz`:
  - `predictions`: (N, 21, 45, 45) - Predicted concentrations
  - `targets`: (N, 21, 45, 45) - Ground truth
  - `past_conc`: (N, 30, 21, 45, 45) - Input sequence
  - `static_maps`: (N, 2, 21, 45, 45) - Terrain + Source
  - `mse`, `mae`, `pcc`: Per-sample metrics

- `summary.json`: Overall statistics

---

### 3. Visualization

```bash
python visualize_seq2seq.py
```

**Generated Figures** (`figures_seq2seq/`):

1. **Sample Comparison** (`sample_XXX_comparison.png`)
   - Multi-level views (Z=0, 5, 10, 15, 20)
   - Columns: Terrain | Source | Ground Truth | Prediction | Error

2. **Temporal Evolution** (`sample_XXX_temporal_zYY.png`)
   - Shows past 30 hours + prediction
   - Center point time series
   - Compares prediction vs ground truth at t=0

3. **Vertical Profile** (`sample_XXX_vertical_profile.png`)
   - Concentration vs height at center point

4. **Metrics Distribution** (`metrics_distribution.png`)
   - Histograms of MSE, MAE, PCC across test set

5. **Scatter Plot** (`scatter_comparison.png`)
   - All predictions vs ground truth (log-log scale)

---

## Data Flow

### Dataset Processing

```python
# From .npz files
input_maps.npz:
  - terrain: (45, 45)
  - source_q: (45, 45)

labels_conc.npz:
  - conc: (TotalTime, 45, 45, 21)

↓ [ConcentrationSeq2SeqDataset]

# Per sample output
past_conc:     (30, 21, H, W)    # Past 30 hours
static_maps:   (2, 21, H, W)     # [Terrain, Source]
future_conc:   (1, 21, H, W)     # Target (1 hour ahead)

where H = W = 32 (train) or 45 (val/test)
```

### Data Augmentation (Training Only)
- Random 90° rotation
- Random crop (45×45 → 32×32)

---

## Model Parameters

### Default Configuration
- **Latent dimension**: 128
- **Transformer heads**: 4
- **Transformer layers**: 2
- **Dropout**: 0.1
- **Total parameters**: ~5-10M (depending on configuration)

### Memory Usage
- Training batch (B=8, crop=32): ~2-3 GB GPU
- Validation batch (B=8, full=45): ~5-6 GB GPU
- Reduce `BATCH_SIZE` if GPU OOM

---

## Key Differences from Other Models

### vs. `train_no_wind.py` (Static Model)
- ✅ **Seq2Seq uses temporal history** (past 30 hours)
- ❌ No-wind model uses only static inputs (no temporal context)

### vs. `train_conc.py` (Full Model)
- ✅ **Seq2Seq predicts from concentration history alone**
- ❌ Full model uses explicit wind fields and meteorological data

### Advantages
- Simple input (only concentration history + static maps)
- No need for accurate wind measurements
- Learns temporal dynamics implicitly
- Good for scenarios where meteorological data is unavailable

### Limitations
- Relies on data-driven learning (may not generalize to unseen conditions)
- Requires sufficient training data
- Cannot explicitly model physical wind transport

---

## Troubleshooting

### OpenMP Error
Already handled in all scripts:
```python
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

### GPU Out of Memory
Reduce batch size in config:
```python
BATCH_SIZE = 4  # or 2
```

### WandB Not Available
Automatically disabled if not installed. To enable:
```bash
pip install wandb
wandb login
```

---

## Expected Performance

Based on architecture design:

- **MSE**: Should decrease with more training epochs
- **MAE**: Typically 0.01-0.1 (depends on data scale)
- **PCC**: Target > 0.7 for good spatial pattern matching

---

## Next Steps

1. **Run training**:
   ```bash
   python train_seq2seq.py
   ```

2. **Monitor training** (if WandB enabled):
   - Check loss curves
   - Validate convergence

3. **Run inference**:
   ```bash
   python inference_seq2seq.py
   ```

4. **Visualize results**:
   ```bash
   python visualize_seq2seq.py
   ```

5. **Analyze**:
   - Check `figures_seq2seq/` for visual quality
   - Review `inference_results_seq2seq/summary.json` for metrics

---

## References

- Model architecture inspired by DeepONet and Transformers
- 3D CNN for spatial feature extraction
- Transformer for temporal modeling
- Decoder with transposed convolutions for upsampling

---

**Created**: 2026-01-07
**Author**: Claude Sonnet 4.5
**Version**: 1.0
