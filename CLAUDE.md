# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**kari-onestop-uas**: Research on physics-based models for atmospheric environment prediction and guidance laws for pollutant source tracking.

This is a deep learning project that uses a **Physics-Informed Neural Network (PINN)** based on **DeepONet architecture** to predict atmospheric pollutant concentration and wind fields in 3D space. The model integrates EPA AERMOD simulation data with physics-informed loss functions to ensure predictions follow conservation laws.

## Core Architecture

### Three-Stage Training Pipeline

The project follows a hierarchical training strategy:

1. **Wind Field Pretraining** (`train_wind.py`)
   - Learns 3D wind velocity (u, v, w) from AERMOD data
   - Uses physics loss to enforce mass conservation (∇·u = 0)
   - Checkpoint: `checkpoints/wind_pretrain_best.pth`

2. **Joint Training** (`train_conc.py`)
   - Loads pretrained wind model
   - Trains concentration prediction head
   - Fine-tunes both wind and concentration simultaneously
   - Physics loss enforces terrain boundary conditions
   - Checkpoint: `checkpoints/joint_best.pth`

3. **Inference** (`inference/inference.py`)
   - Loads joint model to predict future concentration fields
   - Visualizes 3D pollutant dispersion

### Model Components (ST_TransformerDeepONet)

The model uses DeepONet architecture with three main components:

**Branch Networks** (encode input context):
- `Conv3dBranch`: Encodes 5-channel 3D spatial data → [Terrain, Source_Q, Source_H, U, V, W]
- `TransformerObsBranch`: Encodes meteorological time series → [u_surface, v_surface, 1/L]

**Trunk Network**:
- `SpatioTemporalTrunk`: Maps (x,y,z,t) coordinates using Fourier Features to enable high-frequency learning

**Output Heads**:
- Wind Head: Predicts (u, v, w) velocity vectors
- Concentration Head: Predicts pollutant concentration C(x,y,z,t)

**Key Hyperparameters** (stored in checkpoints):
- `latent_dim`: Usually 128 or 256
- `fourier_scale`: Controls Fourier feature frequency (10.0-30.0)
- `dropout`: Regularization strength (0.1-0.3)

## Data Pipeline

### Preprocessing (MUST run before training)

```bash
cd physics_network/dataset
python main.py
```

This generates three `.npz` files in `physics_network/processed_data/`:

**1. input_maps.npz** (Static spatial data)
- `terrain`: (45, 45) normalized elevation map from `ulsan_terrain.rou`
- `source_q`: (45, 45) emission rate map with Gaussian splatting (log1p transformed)
- `source_h`: (45, 45) stack height map (normalized by 200m)
- Coordinate transform: UTM → Grid with [X,Y] transpose to align with AERMOD output

**2. input_met.npz** (Meteorological sequences)
- `met`: (Time, 43) = [u₀, v₀, u₁₀, v₁₀, ..., u₂₀₀, v₂₀₀, 1/L]
  - 21 vertical layers (0-200m, 10m spacing) via linear interpolation
  - Monin-Obukhov Length converted to inverse (1/L) to handle neutral stability singularity
- Parsed from `METEOR.DBG` GRID HEIGHT section (RECEPT section ignored)

**3. labels_conc.npz** (Ground truth concentration)
- `conc`: (Time, 45, 45, 21) 4D concentration field
- Log1p + Global StandardScaler transformation
- Synchronized with `input_met.npz` timestamps (YYMMDDHH format)
- Parsed from PLT files: `ulsan_conc_1hr_{z:06d}m.plt`

### Dataset Structure

**Grid Configuration** ([config_param.py](physics_network/dataset/config_param.py)):
- Spatial: 45×45×21 grid (NX, NY, NZ)
- Resolution: 100m×100m×10m (DX, DY, DZ)
- Origin: UTM (529800.0, 3919200.0) matching AERMOD DOMAINXY

**Data Split**: 80% train / 10% val / 10% test (time-based, no shuffling)

**Input Channels** (5 total):
1. Terrain elevation (normalized)
2. Source emission rate (log1p)
3. U wind component (3D field)
4. V wind component (3D field)
5. W wind component (3D field, initially sparse/zero)

**Physics Cache**: 3D wind fields (u,v,w) are pre-calculated in-memory during dataset initialization to avoid runtime overhead.

## Training Commands

### 1. Preprocess Data
```bash
cd physics_network/dataset
python main.py
```

### 2. Train Wind Field (Stage 1)
```bash
cd physics_network/train
python train_wind.py
```
- Output: `checkpoints/wind_pretrain_best.pth`
- Logs to WandB project: `KARI_Wind_Physics`
- Training time: ~100 epochs

### 3. Train Concentration (Stage 2)
```bash
cd physics_network/train
python train_conc.py
```
- Requires: `checkpoints/wind_pretrain_best.pth`
- Output: `checkpoints/joint_best.pth`
- Logs to WandB project: `KARI_Joint_Diffusion`

### 4. Run Inference
```bash
cd physics_network/inference
python inference.py
```
- Loads: `checkpoints/joint_best.pth`
- Generates 3D visualization

## Key Design Patterns

### Physics-Informed Loss Function

The `PhysicsInformedGridLoss` ([wind_loss.py](physics_network/train/wind_loss.py)) combines:

1. **Data Loss**: MSE between predicted and target concentration
2. **Pattern Loss**: Pearson Correlation Coefficient (PCC) for spatial distribution matching
3. **Physics Loss**:
   - Mass conservation: ∇·**u** ≈ 0 (divergence-free constraint)
   - Terrain boundary: w ≈ 0 at ground level (z=0)

Loss weights are configured per training stage:
- Wind pretraining: `lambda_phys=1.0`, `lambda_pcc=0.0`
- Joint training: All three losses active

### Coordinate System

**Critical Transpose Issue**: AERMOD outputs use [X, Y] indexing, but the model expects standard [Y, X] array indexing. The preprocessing applies coordinate swapping in:
- [process_maps.py:123-128](physics_network/dataset/process_maps.py#L123-L128) (Gaussian splatting)
- [physics_utils.py:5-17](physics_network/dataset/physics_utils.py#L5-L17) (`xy_to_grid` function)

### Checkpoint Management

All checkpoints save:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'config': {'latent_dim': int, 'fourier_scale': float, 'dropout': float}
}
```

**Loading pattern** (from [train_conc.py:73-92](physics_network/train/train_conc.py#L73-L92)):
```python
ckpt = torch.load(checkpoint_path)
saved_cfg = ckpt['config']
model = ST_TransformerDeepONet(
    latent_dim=saved_cfg['latent_dim'],
    fourier_scale=saved_cfg['fourier_scale']
)
model.load_state_dict(ckpt['model_state_dict'], strict=False)  # strict=False for partial loading
```

## Data Sources

### EPA AERMOD Simulation Files (in `epa_sim/data/`)

**Terrain**:
- `ter/ulsan_terrain.rou`: AERMAP elevation grid (ELEV keyword)

**Sources**:
- `mod/aermod.inp`: Source parameters (SRCPARAM keyword → Q, H)
- `ter/ulsan_source_elev.src`: Source locations (LOCATION POINT keyword → X, Y)

**Meteorology**:
- `met/METEOR.DBG`: Hourly wind profiles + Monin-Obukhov Length
  - Parse GRID HEIGHT section only (skip RECEPT)
  - Date format: YR MO DA HR ... L ...
  - Profile format: IDX HEIGHT WD WS ...

**Concentration** (Ground Truth):
- `conc/ulsan_conc_1hr_{z:06d}m.plt`: Hourly concentration at each vertical level
  - Columns: X Y Conc ... Date(YYMMDDHH)
  - 21 files (z = 0, 10, 20, ..., 200m)

## Important Notes

### Windows Path Compatibility
[config_param.py](physics_network/dataset/config_param.py#L26-L48) uses `os.path.join()` throughout to ensure Windows compatibility. Absolute paths are auto-detected from the script location.

### Monin-Obukhov Length Preprocessing
The stability parameter L is converted to 1/L ([process_met.py:122-136](physics_network/dataset/process_met.py#L122-L136)):
- Neutral (|L| > 5000 or L < -9000) → 1/L = 0
- Extreme values (|L| < 0.1) → clipped to ±10
- This removes singularities when L → ∞ and makes neutral conditions learnable

### WandB Integration
All training scripts log to Weights & Biases. Make sure you have:
- WandB account with entity: `jhlee98` (or modify in code)
- Project names:
  - `KARI_Wind_Physics` (wind pretraining)
  - `KARI_Joint_Diffusion` (joint training)

### Memory Considerations
- Default batch size: 2-16 (adjustable in training configs)
- Physics cache precomputes full 3D wind fields → high RAM usage
- Crop size: 32×32 for training, 45×45 for validation/inference

## File Organization

```
kari-onestop-uas/
├── epa_sim/data/           # Raw AERMOD simulation files
│   ├── ter/                # Terrain and source location
│   ├── met/                # METEOR.DBG meteorology
│   ├── mod/                # AERMOD input files
│   └── conc/               # PLT concentration outputs
├── physics_network/
│   ├── dataset/            # Data preprocessing and loading
│   │   ├── main.py         # [RUN FIRST] Preprocessing pipeline
│   │   ├── config_param.py # Grid and path configuration
│   │   ├── process_maps.py # Terrain + source → input_maps.npz
│   │   ├── process_met.py  # METEOR.DBG → input_met.npz
│   │   ├── process_labels.py # PLT files → labels_conc.npz
│   │   ├── dataset.py      # PyTorch Dataset class
│   │   └── physics_utils.py # Coordinate transforms
│   ├── model/
│   │   └── model.py        # ST_TransformerDeepONet architecture
│   ├── train/
│   │   ├── train_wind.py   # [STEP 1] Wind pretraining
│   │   ├── train_conc.py   # [STEP 2] Joint concentration training
│   │   └── wind_loss.py    # Physics-informed loss functions
│   ├── inference/
│   │   ├── inference.py    # Run inference on test set
│   │   └── vis_3d.py       # 3D visualization utilities
│   ├── checkpoints/        # Saved model weights
│   └── processed_data/     # Generated .npz files
└── wandb/                  # WandB logging artifacts
```

## Validation and Testing

- `dataset/validate_data.py`: Check preprocessed data integrity
- `train/eval_wind.py`: Evaluate wind field predictions
- `train/evaluate_vertical_profile.py`: Analyze vertical concentration profiles
- `train/compare_gt_pred.py`: Generate side-by-side GT vs Prediction plots
