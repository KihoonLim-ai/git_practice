# Source Height 처리 방식 비교

## train_conc.py vs train_seq2seq.py

두 모델이 `source_h` (배출원 높이)를 처리하는 방식의 차이점을 분석합니다.

---

## input_maps.npz 구조 (공통)

```python
input_maps.npz:
├─ terrain      (45, 45)  # 정규화된 지형 고도
├─ source_q     (45, 45)  # Log1p 변환된 배출량
├─ source_h     (45, 45)  # 정규화된 배출원 높이 (0~1, 각 셀의 굴뚝 높이)
├─ terrain_max  scalar    # 메타데이터
└─ raw_sources  (N, 4)    # 메타데이터
```

---

## 방법 1: train_conc.py (Height-Aware Placement)

### 코드 (dataset.py, lines 115-122)

```python
# Source Map 생성 - 배출원을 실제 높이에 배치
src_temp = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
rows, cols = np.where(self.source_q > 0)

for r, c in zip(rows, cols):
    h_n = self.source_h[r, c]                    # 정규화된 높이 (0~1)
    z_idx = int(h_n * (self.nz - 1))             # Z 인덱스로 변환 (0~20)
    # 해당 높이에만 배출원 배치
    src_temp[np.clip(z_idx, 0, self.nz-1), r, c] = self.source_q[r, c]

self.source_3d = src_temp[np.newaxis, :, :, :]   # (1, 21, 45, 45)
```

### 물리적 의미

**3D 배치 예시:**
```
Z=20 (200m): [0, 0, 0, ...]        # 높은 고도: 배출원 없음
Z=15 (150m): [0, 0, 0, ...]
Z=10 (100m): [0, Q1, 0, ...]       # ← 100m 굴뚝 위치에만 Q1
Z=5  (50m):  [Q2, 0, 0, ...]       # ← 50m 굴뚝 위치에만 Q2
Z=0  (0m):   [0, 0, 0, ...]        # 지면: 배출원 없음
```

### 모델 입력

```python
input_vol = [
    Terrain (1, 21, 45, 45),  # 지형 마스크
    Source  (1, 21, 45, 45),  # 높이별 배출원 (이미 source_h 반영됨!)
    U       (1, 21, 45, 45),  # 바람
    V       (1, 21, 45, 45),
    W       (1, 21, 45, 45)
]  # → (5, 21, 45, 45)
```

**특징:**
- ✅ **물리적으로 정확**: 배출원이 실제 굴뚝 높이에만 존재
- ✅ **암묵적 정보**: 모델이 source의 Z 위치를 보고 높이를 알 수 있음
- ✅ **희소 표현**: 대부분 0, 배출원 위치만 값 존재
- ❌ **명시적 높이 정보 부족**: 높이 값 자체는 입력되지 않음

---

## 방법 2: train_seq2seq.py (개선 후) (Explicit Height Channel)

### 코드 (dataset_seq2seq.py, lines 105-117)

```python
# Source Emission Rate (배출량 - 모든 고도에 동일)
source_q_3d = np.tile(
    self.source_q[np.newaxis, :, :],
    (self.nz, 1, 1)
).astype(np.float32)  # (21, 45, 45)

# Source Height (배출원 높이 - 모든 고도에 동일)
source_h_3d = np.tile(
    self.source_h[np.newaxis, :, :],
    (self.nz, 1, 1)
).astype(np.float32)  # (21, 45, 45)
```

### 물리적 의미

**3D 배치 예시:**
```
Source_Q Channel:
Z=20: [0,   Q1, 0,  Q2, ...]       # 모든 고도에 동일
Z=15: [0,   Q1, 0,  Q2, ...]
Z=10: [0,   Q1, 0,  Q2, ...]
Z=5:  [0,   Q1, 0,  Q2, ...]
Z=0:  [0,   Q1, 0,  Q2, ...]

Source_H Channel:
Z=20: [0,  0.5, 0, 0.25, ...]      # 모든 고도에 동일
Z=15: [0,  0.5, 0, 0.25, ...]      # 0.5 = 100m 굴뚝
Z=10: [0,  0.5, 0, 0.25, ...]      # 0.25 = 50m 굴뚝
Z=5:  [0,  0.5, 0, 0.25, ...]
Z=0:  [0,  0.5, 0, 0.25, ...]
```

### 모델 입력

```python
static_maps = [
    Terrain  (21, 45, 45),  # 지형 마스크
    Source_Q (21, 45, 45),  # 배출량 (모든 고도 동일)
    Source_H (21, 45, 45)   # 배출원 높이 (모든 고도 동일)
]  # → (3, 21, 45, 45)
```

**특징:**
- ✅ **명시적 높이 정보**: 높이 값이 직접 입력됨
- ✅ **모델 학습 가능**: 네트워크가 Q와 H의 관계를 학습
- ❌ **물리적으로 부정확**: 배출원이 모든 고도에 존재 (비현실적)
- ❌ **중복 정보**: 같은 값이 21개 층에 반복

---

## 비교 분석

| 측면 | train_conc.py | train_seq2seq.py |
|------|---------------|------------------|
| **물리적 정확성** | ✅ 매우 높음 | ⚠️ 낮음 |
| **배출원 위치** | 실제 높이에만 존재 | 모든 고도에 존재 |
| **높이 정보** | 암묵적 (위치로 표현) | 명시적 (별도 채널) |
| **모델 학습 난이도** | ⚠️ 암묵적 정보 학습 필요 | ✅ 명시적 정보 제공 |
| **메모리 효율** | ✅ 희소 (대부분 0) | ⚠️ 중복 (21배 반복) |
| **AERMOD 일치도** | ✅ 높음 | ❌ 낮음 |

---

## 예시: 100m 굴뚝 (h_n = 0.5)

### train_conc.py 방식

```python
# Z=10 (100m) 층에만 배출원 존재
Source_3D[0, 10, r, c] = Q   # 100m 층
Source_3D[0,  5, r, c] = 0   # 50m 층
Source_3D[0, 15, r, c] = 0   # 150m 층
```

**모델이 보는 것:**
- "이 위치의 10번째 층에 배출원이 있다"
- → 암묵적으로 높이 정보 인식

### train_seq2seq.py 방식

```python
# 모든 고도에 동일한 정보
Source_Q[z, r, c] = Q      # z = 0~20, 모두 동일
Source_H[z, r, c] = 0.5    # z = 0~20, 모두 동일 (100m)
```

**모델이 보는 것:**
- "이 위치의 배출량은 Q이고 높이는 0.5이다"
- → 명시적으로 높이 정보 제공

---

## 어떤 방식이 더 좋은가?

### train_conc.py 방식 (Height-Aware Placement) 추천!

**장점:**
1. **물리적 정확성**: AERMOD 가우시안 플룸 모델과 일치
2. **메모리 효율**: 희소 표현 (배출원 위치만 값)
3. **바람과의 상호작용**: 배출원 높이의 바람장이 직접 영향

**단점:**
1. **암묵적 정보**: 모델이 위치로부터 높이를 추론해야 함
2. **복잡한 전처리**: Loop 필요 (하지만 1회만 실행)

### train_seq2seq.py 개선 방안

**Option A: train_conc.py 방식 도입 (추천)**

```python
def _init_static_maps(self):
    # 현재 방식 대신 height-aware placement
    src_temp = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
    rows, cols = np.where(self.source_q > 0)

    for r, c in zip(rows, cols):
        h_n = self.source_h[r, c]
        z_idx = int(h_n * (self.nz - 1))
        src_temp[np.clip(z_idx, 0, self.nz-1), r, c] = self.source_q[r, c]

    self.source_3d = src_temp.astype(np.float32)

    static_maps = np.stack([
        self.terrain_3d,    # (21, 45, 45)
        self.source_3d      # (21, 45, 45) - 이제 height-aware!
    ], axis=0)  # (2, 21, 45, 45)
```

**장점:**
- 물리적으로 정확
- Source_H 채널 불필요 (정보가 이미 반영됨)
- train_conc.py와 동일한 물리 표현

**Option B: 현재 방식 유지 (모델 학습에 의존)**

- 모델이 Source_Q와 Source_H의 관계를 학습
- 학습 데이터가 충분하면 작동 가능
- 하지만 물리적으로 부정확한 입력

---

## 실제 물리 프로세스

AERMOD 가우시안 플룸 모델:

```
1. 배출원 위치: (x, y, h_stack)
   └─> train_conc.py: ✅ Z=h_stack 층에만 존재
   └─> train_seq2seq.py: ❌ 모든 Z에 존재

2. 배출: Q [g/s] at height h_stack
   └─> train_conc.py: ✅ 해당 높이에서만 배출
   └─> train_seq2seq.py: ❌ 모든 높이에서 배출 (비현실적)

3. 확산: Wind field at h_stack 영향받음
   └─> train_conc.py: ✅ 자동으로 올바른 바람 사용
   └─> train_seq2seq.py: ⚠️ 모델이 학습해야 함

4. 농도 예측: C(x,y,z,t)
   └─> train_conc.py: ✅ 물리적으로 정확한 입력
   └─> train_seq2seq.py: ⚠️ 학습에 의존
```

---

## 권장 사항

### 즉각 적용 (train_seq2seq.py 개선)

**Step 1: dataset_seq2seq.py 수정**

`_init_static_maps()` 함수를 다음과 같이 변경:

```python
def _init_static_maps(self):
    """정적 3D 지도 생성 (Height-Aware Placement)"""
    # Terrain Mask
    z_vals = np.linspace(0, 1.0, self.nz).reshape(self.nz, 1, 1)
    terrain_mask = (z_vals <= self.terrain[np.newaxis, :, :]).astype(np.float32)
    self.terrain_3d = terrain_mask  # (21, 45, 45)

    # Source Map (Height-Aware) ✨
    src_temp = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
    rows, cols = np.where(self.source_q > 0)

    for r, c in zip(rows, cols):
        h_n = self.source_h[r, c]
        z_idx = int(h_n * (self.nz - 1))
        src_temp[np.clip(z_idx, 0, self.nz-1), r, c] = self.source_q[r, c]

    self.source_3d = src_temp.astype(np.float32)  # (21, 45, 45)
```

**Step 2: 모델 입력 변경 (2채널로 축소)**

```python
static_maps = np.stack([
    self.terrain_3d,    # (21, 45, 45)
    self.source_3d      # (21, 45, 45) - Height-aware!
], axis=0)  # (2, 21, 45, 45)
```

**Step 3: model_seq2seq_v2.py 수정**

```python
self.static_encoder = StaticEncoder(
    in_channels=2,  # 3 → 2로 변경
    out_channels=hidden_channels
)
```

**예상 효과:**
- ✅ 물리적으로 정확한 입력
- ✅ 메모리 효율 증가
- ✅ train_conc.py와 동일한 물리 표현
- ✅ 더 빠른 수렴 예상

---

## 결론

**train_conc.py의 height-aware placement 방식이 더 물리적으로 정확합니다.**

train_seq2seq.py를 개선하려면:
1. Source_H를 별도 채널로 제공하는 대신
2. Source_Q를 올바른 높이에 배치하는 방식 채택
3. 2채널 static_maps로 충분 (Terrain + Height-aware Source)

이는 AERMOD의 물리 프로세스를 더 정확히 반영하며, 모델 학습도 더 효율적일 것으로 예상됩니다.
