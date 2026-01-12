# Seq2Seq Model Improvements (v2)

## 문제점 분석

### 이전 모델 (model_seq2seq.py)의 심각한 결함:

1. **시간적 연관성 학습 불가**
   ```python
   # 각 timestep을 독립적으로 처리
   for t in range(T):
       x_t = x[:, t:t+1, :, :, :]  # 30번 반복
       feat = self.conv_layers(x_t)
       features.append(feat)
   ```
   - 30개 timestep을 루프로 하나씩 독립 처리
   - **Seq2Seq가 아니라 30개 이미지의 앙상블**
   - 시간 순서 정보 완전 손실

2. **공간 정보 손실**
   ```python
   nn.AdaptiveAvgPool3d((1, 1, 1))  # 모든 공간 정보를 1x1x1로 압축
   ```
   - 21×45×45 → 1×1×1로 극단적 압축
   - 농도 분포의 공간적 패턴 완전 소실

3. **비효율적 아키텍처**
   - Transformer로 이미 압축된 64차원 벡터 처리
   - Decoder가 1×1×1에서 21×45×45로 복원 (정보 없이 확대)
   - 예측이 실제 공간 구조와 무관

---

## 개선된 모델 (model_seq2seq_v2.py)

### 1. ConvLSTM 기반 시간 인코딩

**이전:**
```python
# 각 timestep 독립 처리 (시간 정보 무시)
for t in range(T):
    x_t = x[:, t:t+1, :, :, :]
    feat = self.conv_layers(x_t)
```

**개선:**
```python
# ConvLSTM으로 시간 의존성 학습
for t in range(T):
    x_t = x[:, t:t+1, :, :, :]
    for layer in lstm_cells:
        h, c = layer(x_t, (h, c))  # 이전 상태 활용
        x_t = h
```

**효과:**
- 이전 timestep의 정보를 hidden state에 누적
- 시간 순서 학습 가능
- 진짜 Seq2Seq 구조

---

### 2. 공간 구조 보존

**이전:**
```python
AdaptiveAvgPool3d((1, 1, 1))  # (B, C, 21, 45, 45) → (B, C, 1, 1, 1)
# 정보 완전 손실
```

**개선:**
```python
# ConvLSTM 출력: (B, C, 21, 45, 45)
# 공간 구조 그대로 유지
temporal_features = convlstm_encoder(past_conc)  # (B, 32, 21, 45, 45)
```

**효과:**
- 농도 분포의 공간적 패턴 보존
- Decoder가 실제 정보로부터 복원 가능

---

### 3. U-Net 스타일 Decoder

**이전:**
```python
# 1×1×1에서 시작 (정보 없음)
x = fc(latent_vector)  # (B, 128) → (B, 128×3×6×6)
x = decoder(x)  # 정보 없이 확대
```

**개선:**
```python
# 공간 정보를 유지한 상태에서 디코딩
combined = cat([temporal_features, static_features], dim=1)  # (B, 64, 21, 45, 45)
fused = fusion_conv(combined)  # (B, 32, 21, 45, 45)
output = decoder(fused)  # (B, 1, 21, 45, 45)
```

**효과:**
- 의미 있는 spatial features로부터 예측
- Skip connection 효과 (temporal + static)

---

## 아키텍처 비교

### 이전 모델 (잘못된 설계)
```
Past Conc (B, 30, 21, 45, 45)
  ↓ [각 timestep 독립 처리 - 30번 반복]
  ↓ [AdaptiveAvgPool3d → 1×1×1]
Encoded (B, 30, 128)  [공간 정보 손실]
  ↓ [Transformer]
Last timestep (B, 128)
  ↓ [Decoder: 128 → 21×45×45]
Future Conc (B, 1, 21, 45, 45)  [정보 없이 생성]
```

### 개선된 모델 (올바른 설계)
```
Past Conc (B, 30, 21, 45, 45)
  ↓ [ConvLSTM - 시간 의존성 학습]
Temporal Features (B, 32, 21, 45, 45)  [공간+시간 정보 보존]
  ↓
Static Maps (B, 3, 21, 45, 45)  [Terrain, Source_Q, Source_H]
  ↓ [Static Encoder]
Static Features (B, 32, 21, 45, 45)
  ↓
Fusion (B, 64, 21, 45, 45)  [Temporal + Static]
  ↓ [U-Net Decoder]
Future Conc (B, 1, 21, 45, 45)  [의미 있는 예측]
```

---

## 핵심 개선 사항

| 측면 | 이전 모델 | 개선 모델 |
|------|-----------|-----------|
| **시간 모델링** | 독립 처리 (❌) | ConvLSTM (✅) |
| **공간 정보** | 1×1×1 압축 (❌) | 21×45×45 유지 (✅) |
| **Seq2Seq** | 아님 (❌) | 진짜 Seq2Seq (✅) |
| **정보 흐름** | 단절 (❌) | 연속 (✅) |
| **예측 능력** | 거의 없음 (❌) | 실제 학습 가능 (✅) |

---

## 파라미터 변화

### 이전 설정:
```python
LATENT_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
```

### 새 설정:
```python
HIDDEN_CHANNELS = 32      # ConvLSTM hidden channels
NUM_LSTM_LAYERS = 2       # Number of ConvLSTM layers
```

**파라미터 수:** 약 200K → 400K (더 많은 학습 가능한 구조)

---

## 사용 방법

### 1. 새 모델로 학습
```bash
cd physics_network/train
python train_seq2seq.py
```

### 2. Inference
```bash
python inference_seq2seq.py
```

### 3. Visualization
```bash
python visualize_seq2seq.py
```

---

## 예상 성능 개선

1. **MSE**: 10배 이상 감소 예상
2. **PCC**: 0.1-0.2 → 0.7-0.9 예상
3. **시각적 품질**: 노이즈 → 실제 농도 분포 패턴

---

## 주의사항

- **메모리 사용량 증가**: ConvLSTM이 (B, C, D, H, W) 크기의 hidden state 유지
- **학습 시간**: 이전보다 1.5-2배 소요
- **Batch size 조정 필요**: GPU 메모리에 따라 4-8로 설정

---

## 결론

이전 모델은 **구조적으로 예측이 불가능한 설계**였습니다:
- 시간 정보를 학습하지 못함
- 공간 정보를 버림
- Decoder가 정보 없이 추측

새 모델은 **올바른 Seq2Seq 아키텍처**입니다:
- ConvLSTM으로 시간 의존성 학습
- 공간 구조 보존
- 의미 있는 예측 가능

**이제 다시 학습하면 실제로 예측이 될 것입니다!**
