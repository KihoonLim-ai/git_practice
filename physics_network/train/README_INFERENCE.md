# Inference & Visualization Guide for No-Wind Model

학습된 No-Wind 모델을 사용하여 추론을 실행하고 결과를 시각화하는 방법을 설명합니다.

---

## 파일 구조

```
physics_network/train/
├── train_no_wind.py           # 학습 스크립트
├── inference_no_wind.py       # ✨ 추론 스크립트 (새로 생성)
├── visualize_no_wind.py       # ✨ 시각화 스크립트 (새로 생성)
├── checkpoints_no_wind/       # 학습된 모델 저장 위치
│   └── best_no_wind.pth       # 최고 성능 체크포인트
├── inference_results_no_wind/ # 추론 결과 저장 (자동 생성)
│   ├── predictions.npz        # 예측값, 실제값, 입력 데이터
│   └── summary.json           # 통계 요약
└── figures_no_wind/           # 시각화 결과 저장 (자동 생성)
    ├── sample_000_comparison.png
    ├── sample_000_vertical_profile.png
    ├── metrics_distribution.png
    └── scatter_comparison.png
```

---

## Step 1: 추론 실행

학습된 모델로 테스트 데이터셋에 대해 예측을 수행합니다.

### 실행 방법

```bash
cd physics_network/train
python inference_no_wind.py
```

### 설정 변경 (선택)

`inference_no_wind.py` 파일의 `InferenceConfig` 클래스를 수정:

```python
class InferenceConfig:
    CHECKPOINT_PATH = "checkpoints_no_wind/best_no_wind.pth"  # 체크포인트 경로
    OUTPUT_DIR = "inference_results_no_wind"                  # 결과 저장 디렉토리
    BATCH_SIZE = 4                                            # 배치 크기
    NUM_SAMPLES = 20                                          # 저장할 샘플 개수
```

### 출력 예시

```
======================================================================
🔮 No-Wind Model Inference
======================================================================

🖥️ Using device: cuda

📂 Loading checkpoint from: checkpoints_no_wind/best_no_wind.pth
✅ Model loaded from epoch 45
   Best Val Loss: 0.234567

📦 Loading test data...
[TEST] Loading static maps + concentration labels...
   -> Mode: TEST | Available Samples: 182

🔮 Running inference on test set...
Inference: 100%|████████████████████| 5/5 [00:12<00:00,  2.50s/it]

📊 Inference Results (on 20 samples):
   Average MSE: 0.123456
   Average MAE: 0.098765
   Average PCC: 0.6543

💾 Saving results to: inference_results_no_wind
✅ Saved:
   - predictions.npz (predictions, targets, inputs, metrics)
   - summary.json (statistics)

======================================================================
🎉 Inference completed!
======================================================================
```

---

## Step 2: 결과 시각화

추론 결과를 다양한 형태로 시각화합니다.

### 실행 방법

```bash
python visualize_no_wind.py
```

### 설정 변경 (선택)

`visualize_no_wind.py` 파일의 `VisualizationConfig` 클래스를 수정:

```python
class VisualizationConfig:
    RESULTS_DIR = "inference_results_no_wind"     # 추론 결과 디렉토리
    OUTPUT_DIR = "figures_no_wind"                # 그림 저장 디렉토리
    DPI = 150                                     # 해상도
    NUM_SAMPLES_TO_PLOT = 5                       # 플롯할 샘플 개수
    Z_LEVELS_TO_PLOT = [0, 5, 10, 15, 20]        # 시각화할 고도 레벨
```

### 생성되는 그림

#### 1. **샘플별 비교 플롯** (`sample_XXX_comparison.png`)

각 샘플에 대해 5개 열로 구성된 비교 플롯:
- **Column 1**: Terrain Mask (입력 - 지형 마스크)
- **Column 2**: Source Map (입력 - 오염원 위치)
- **Column 3**: Ground Truth (실제 농도 분포)
- **Column 4**: Prediction (예측 농도 분포)
- **Column 5**: Absolute Error (절대 오차 맵)

각 행은 다른 고도 레벨 (Z=0, 5, 10, 15, 20)

#### 2. **수직 프로파일 플롯** (`sample_XXX_vertical_profile.png`)

특정 위치(중심점)의 수직 농도 분포:
- X축: 농도 (log scale)
- Y축: 고도 레벨 (0~20)
- 빨간선: Ground Truth
- 파란선: Prediction

#### 3. **메트릭 분포 히스토그램** (`metrics_distribution.png`)

전체 테스트 샘플의 성능 메트릭 분포:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- PCC (Pearson Correlation Coefficient)

각 히스토그램에 평균(빨간 점선)과 중앙값(파란 점선) 표시

#### 4. **산점도** (`scatter_comparison.png`)

모든 예측값 vs 실제값의 산점도:
- X축: Ground Truth Concentration (log scale)
- Y축: Predicted Concentration (log scale)
- 빨간 점선: 완벽한 예측선 (y=x)
- 전체 데이터의 Pearson 상관계수 표시

---

## 결과 해석

### 성능 메트릭

| 메트릭 | 설명 | 좋은 값 |
|--------|------|---------|
| **MSE** | 예측값과 실제값의 제곱 오차 평균 | 낮을수록 좋음 (< 0.1) |
| **MAE** | 예측값과 실제값의 절대 오차 평균 | 낮을수록 좋음 (< 0.05) |
| **PCC** | 공간 패턴 유사도 (상관계수) | 높을수록 좋음 (> 0.6) |

### No-Wind 모델의 예상 성능

README_NO_WIND.md에서 언급했듯이:

```
Expected Performance:
- Training MSE: ~0.5-1.0 (normalized concentration)
- Validation PCC: ~0.6-0.7 (moderate spatial correlation)
- 비교: Full model (with wind) PCC > 0.85
```

### 시각화에서 확인할 사항

1. **Comparison Plots**:
   - Prediction이 Ground Truth의 **공간 패턴**을 잘 따라가는가?
   - Error map에서 오차가 어느 영역에 집중되는가?
   - 오염원 근처 vs 먼 곳의 예측 정확도 차이

2. **Vertical Profiles**:
   - 고도별 농도 변화 추세를 맞추는가?
   - 특정 고도에서 체계적 over/under prediction이 있는가?

3. **Scatter Plot**:
   - 점들이 y=x 선 근처에 모이는가?
   - 특정 농도 범위에서 편향(bias)이 있는가?

---

## 고급 사용법

### 특정 샘플만 시각화

`visualize_no_wind.py`를 수정하여 관심 있는 샘플만 플롯:

```python
# main() 함수 내부 수정
indices_to_plot = [0, 5, 10, 15, 19]  # 원하는 샘플 인덱스
for i in indices_to_plot:
    plot_sample_comparison(...)
```

### 다른 고도 레벨 시각화

```python
class VisualizationConfig:
    Z_LEVELS_TO_PLOT = [0, 3, 6, 9, 12, 15, 18, 20]  # 더 세밀하게
```

### 다른 체크포인트 비교

여러 epoch의 체크포인트를 비교하려면:

```bash
# Epoch 30 체크포인트로 추론
# inference_no_wind.py 수정
CHECKPOINT_PATH = "checkpoints_no_wind/checkpoint_epoch_30.pth"
OUTPUT_DIR = "inference_results_epoch30"

python inference_no_wind.py
python visualize_no_wind.py  # VisualizationConfig.RESULTS_DIR도 변경
```

---

## 문제 해결

### 1. Checkpoint not found

```
FileNotFoundError: Checkpoint not found: checkpoints_no_wind/best_no_wind.pth
```

**해결**: 먼저 학습을 완료하세요:
```bash
python train_no_wind.py
```

### 2. Results not found

```
FileNotFoundError: Results not found: inference_results_no_wind/predictions.npz
```

**해결**: 먼저 추론을 실행하세요:
```bash
python inference_no_wind.py
```

### 3. Out of memory (GPU)

`inference_no_wind.py`에서 배치 크기 감소:
```python
BATCH_SIZE = 2  # 4 → 2로 변경
```

### 4. Matplotlib 한글 깨짐

시스템에 한글 폰트가 없는 경우, `visualize_no_wind.py` 상단에 추가:
```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'  # 영문 폰트 사용
```

---

## 결과 데이터 구조

### predictions.npz

```python
import numpy as np
data = np.load("inference_results_no_wind/predictions.npz")

data['predictions']  # (20, 21, 45, 45) - 예측 농도
data['targets']      # (20, 21, 45, 45) - 실제 농도
data['inputs']       # (20, 2, 21, 45, 45) - 입력 [Terrain, Source]
data['mse']          # (20,) - 샘플별 MSE
data['mae']          # (20,) - 샘플별 MAE
data['pcc']          # (20,) - 샘플별 PCC
```

### summary.json

```json
{
  "num_samples": 20,
  "mean_mse": 0.123456,
  "std_mse": 0.012345,
  "mean_mae": 0.098765,
  "std_mae": 0.009876,
  "mean_pcc": 0.654321,
  "std_pcc": 0.045678
}
```

---

## 다음 단계

1. **성능 분석**:
   - No-Wind 모델 vs Full 모델 성능 비교
   - 바람 정보 없이 얼마나 예측 가능한가?

2. **모델 개선**:
   - Full resolution 학습 (crop_size=45)
   - 더 깊은 네트워크 (latent_dim 증가)
   - 시계열 정보 추가 (과거 농도 사용)

3. **실제 활용**:
   - 새로운 지형/오염원 시나리오 예측
   - 실시간 추론 시스템 구축

---

## 참고 문서

- [README_NO_WIND.md](README_NO_WIND.md): No-Wind 모델 전체 개요
- [train_no_wind.py](train_no_wind.py): 학습 스크립트
- [model_no_wind.py](../model/model_no_wind.py): 모델 아키텍처
- [dataset_no_wind.py](../dataset/dataset_no_wind.py): 데이터셋 구현
