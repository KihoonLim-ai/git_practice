# Static Maps 개선: 2채널 → 3채널

## 문제 발견

`input_maps.npz`에는 5개 항목이 있지만, 기존 코드에서는 2개만 사용하고 있었습니다:

```python
input_maps.npz 구조:
├─ terrain      (45, 45)  ✅ 사용 중
├─ source_q     (45, 45)  ✅ 사용 중
├─ source_h     (45, 45)  ❌ 사용 안 함 (중요!)
├─ terrain_max  scalar    ❌ 메타데이터 (정규화 복원용)
└─ raw_sources  (N, 4)    ❌ 메타데이터 (디버깅용)
```

### 왜 문제인가?

**배출원 높이 (source_h)는 농도 예측에 매우 중요한 물리적 요소입니다:**

1. **굴뚝 높이와 확산 관계**
   - 높은 굴뚝 (high stack): 오염물질이 더 넓게 분산 → 지면 농도 감소
   - 낮은 배출원 (low emission): 근처 지면 농도 높음
   - AERMOD의 핵심 파라미터 중 하나

2. **지형과의 상호작용**
   - 실제 유효 높이 = Stack Height - Terrain Elevation
   - 산 위의 굴뚝 vs 계곡의 굴뚝은 완전히 다른 영향

3. **현재 모델의 한계**
   - 배출량(Q)만 알고 높이(H)를 모름
   - 같은 배출량이라도 높이에 따라 농도 분포가 완전히 다름

---

## 개선 사항

### Before (2채널)
```python
static_maps = np.stack([
    self.terrain_3d,  # (21, 45, 45)
    self.source_3d    # (21, 45, 45) - source_q만 사용
], axis=0)  # (2, 21, 45, 45)
```

**문제:**
- Source_Q (배출량)만 있음
- Source_H (배출원 높이) 정보 손실
- 물리적으로 불완전한 입력

### After (3채널)
```python
static_maps = np.stack([
    self.terrain_3d,    # (21, 45, 45) - 지형 고도
    self.source_q_3d,   # (21, 45, 45) - 배출량
    self.source_h_3d    # (21, 45, 45) - 배출원 높이 ✨ NEW
], axis=0)  # (3, 21, 45, 45)
```

**개선:**
- 3개의 물리적 요소 모두 포함
- AERMOD 가우시안 플룸 모델의 핵심 파라미터 반영
- 더 정확한 농도 예측 가능

---

## 코드 변경 사항

### 1. Dataset (dataset_seq2seq.py)

**데이터 로딩:**
```python
# 이전
self.terrain = d_maps['terrain']
self.source_q = d_maps['source_q']

# 개선
self.terrain = d_maps['terrain']      # 지형 고도
self.source_q = d_maps['source_q']    # 배출량
self.source_h = d_maps['source_h']    # 배출원 높이 ✨ 추가
```

**3D 맵 생성:**
```python
def _init_static_maps(self):
    # Terrain Mask
    z_vals = np.linspace(0, 1.0, self.nz).reshape(self.nz, 1, 1)
    terrain_mask = (z_vals <= self.terrain[np.newaxis, :, :]).astype(np.float32)
    self.terrain_3d = terrain_mask  # (21, 45, 45)

    # Source Emission Rate
    self.source_q_3d = np.tile(
        self.source_q[np.newaxis, :, :],
        (self.nz, 1, 1)
    ).astype(np.float32)

    # Source Height ✨ 추가
    self.source_h_3d = np.tile(
        self.source_h[np.newaxis, :, :],
        (self.nz, 1, 1)
    ).astype(np.float32)
```

**__getitem__:**
```python
static_maps = np.stack([
    self.terrain_3d,    # (21, 45, 45)
    self.source_q_3d,   # (21, 45, 45)
    self.source_h_3d    # (21, 45, 45) ✨ 추가
], axis=0)  # (3, 21, 45, 45)
```

### 2. Model (model_seq2seq_v2.py)

**StaticEncoder:**
```python
class StaticEncoder(nn.Module):
    """
    Encoder for static maps (Terrain + Source_Q + Source_H)
    """
    def __init__(self, in_channels=3, out_channels=32):  # 2 → 3 ✨
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, D, H, W) - [Terrain, Source_Q, Source_H] ✨
        Returns:
            (B, out_channels, D, H, W)
        """
        return self.encoder(x)
```

**ConcentrationSeq2Seq_v2:**
```python
def __init__(self, hidden_channels=32, num_lstm_layers=2, output_shape=(21, 45, 45)):
    super().__init__()

    # Static encoder
    self.static_encoder = StaticEncoder(
        in_channels=3,  # 2 → 3 ✨
        out_channels=hidden_channels
    )
```

---

## Shape Flow 변화

### Before
```
Dataset:
  past_conc:   (30, 21, 45, 45)
  static_maps: (2, 21, 45, 45)   [Terrain, Source_Q]
  future_conc: (1, 21, 45, 45)

Model:
  StaticEncoder input:  (B, 2, 21, 45, 45)
  StaticEncoder output: (B, 32, 21, 45, 45)
```

### After
```
Dataset:
  past_conc:   (30, 21, 45, 45)
  static_maps: (3, 21, 45, 45)   [Terrain, Source_Q, Source_H] ✨
  future_conc: (1, 21, 45, 45)

Model:
  StaticEncoder input:  (B, 3, 21, 45, 45)  ✨
  StaticEncoder output: (B, 32, 21, 45, 45)
```

---

## 예상 효과

### 1. 물리적 정확성 향상
- AERMOD의 핵심 파라미터 (Q, H, Terrain) 모두 반영
- 가우시안 플룸 모델의 수직 확산 학습 가능

### 2. 예측 성능 개선
- 높이에 따른 농도 분포 차이 학습
- 지형-배출원 상호작용 이해
- 특히 복잡한 지형에서 성능 향상 예상

### 3. 물리적 해석 가능성
- 모델이 "왜" 이런 농도를 예측했는지 이해 가능
- 높은 굴뚝 → 낮은 지면 농도 학습
- 지형 차폐 효과 학습

---

## 재학습 필요성

⚠️ **중요: 기존 체크포인트는 사용 불가**

1. **모델 구조 변경**
   - StaticEncoder의 input channel: 2 → 3
   - 가중치 차원 불일치

2. **데이터셋 변경**
   - static_maps shape: (2, 21, 45, 45) → (3, 21, 45, 45)

3. **재학습 방법**
   ```bash
   cd physics_network/train
   python train_seq2seq.py
   ```

---

## 메타데이터 항목 설명

사용하지 않는 나머지 2개 항목의 용도:

### terrain_max (scalar)
- 정규화 전 최대 고도 값
- 용도: 예측 결과를 원래 스케일로 복원할 때 사용
- 모델 입력으로는 불필요 (이미 정규화된 terrain 사용)

### raw_sources (N, 4)
- 원본 배출원 좌표 [x, y, h, q]
- 용도: 디버깅, 시각화, 검증
- 이미 가우시안 스플래팅으로 grid에 반영됨
- 모델 입력으로는 불필요

---

## 정리

| 항목 | Shape | 용도 | 모델 입력 |
|------|-------|------|----------|
| `terrain` | (45, 45) | 지형 고도 | ✅ 사용 |
| `source_q` | (45, 45) | 배출량 | ✅ 사용 |
| `source_h` | (45, 45) | 배출원 높이 | ✅ 사용 (개선) |
| `terrain_max` | scalar | 정규화 복원 | ❌ 메타데이터 |
| `raw_sources` | (N, 4) | 디버깅/시각화 | ❌ 메타데이터 |

**결론:** 3개의 물리적 필드를 모두 사용하여 더 정확한 예측 가능!
