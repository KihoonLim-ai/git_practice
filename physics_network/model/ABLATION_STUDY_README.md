# Ablation Study Models

5가지 Physics-Informed DeepONet 변형 모델들을 바로 학습시킬 수 있습니다.

## 📋 모델 목록

| 모델 파일 | 설명 | 특징 |
|----------|------|------|
| `model_baseline.py` | 기본 모델 (PDE 없음) | 현재 ST_TransformerDeepONet |
| `model_soft_pde.py` | Soft PDE 제약 | PDE residual을 loss에 추가 |
| `model_annealed_pde.py` | Epoch-based Annealing | PDE weight가 점진적으로 증가 (0.1→1.0) |
| `model_adaptive_pde.py` | Adaptive Weighting (ReLoBRaLo) | 자동 loss 가중치 조정 |
| `model_hard_pde.py` | Hard Constraints | Output transformation으로 제약 강제 |

---

## 🚀 사용 방법

### 1. Baseline Model (기준선)

```python
from model.model_baseline import ST_TransformerDeepONet_Baseline

model = ST_TransformerDeepONet_Baseline(latent_dim=128)

# Training loop
for batch in dataloader:
    pred_wind, pred_conc = model(ctx_map, obs_seq, coords, global_wind)

    # Standard losses only (no PDE)
    loss = loss_mse + loss_pcc + loss_phys
    loss.backward()
```

---

### 2. Soft PDE Model

```python
from model.model_soft_pde import ST_TransformerDeepONet_SoftPDE

model = ST_TransformerDeepONet_SoftPDE(
    latent_dim=128,
    diffusion_coeff=0.1  # D값 (대기 난류 확산계수)
)

# Training loop
LAMBDA_PDE = 1.0  # PDE loss weight (고정)

for batch in dataloader:
    pred_wind, pred_conc, pde_loss = model(
        ctx_map, obs_seq, coords, global_wind,
        compute_pde_loss=True,  # PDE 계산 활성화
        source=source_term      # 배출원 항 (optional)
    )

    loss_total = (
        loss_mse +
        loss_pcc +
        loss_phys +
        LAMBDA_PDE * pde_loss  # PDE residual 추가
    )
    loss_total.backward()
```

**주의**: `coords`는 `requires_grad=True`로 설정해야 합니다!
```python
coords = make_batch_coords(...).requires_grad_(True)
```

---

### 3. Annealed PDE Model

```python
from model.model_annealed_pde import ST_TransformerDeepONet_AnnealedPDE

model = ST_TransformerDeepONet_AnnealedPDE(
    latent_dim=128,
    diffusion_coeff=0.1,
    total_epochs=100  # 전체 epoch 수
)

# Training loop
for epoch in range(total_epochs):
    model.set_epoch(epoch)  # 현재 epoch 업데이트
    lambda_pde = model.get_pde_weight()  # 자동 weight 계산

    print(f"Epoch {epoch}: PDE weight = {lambda_pde:.3f}")

    for batch in dataloader:
        pred_wind, pred_conc, pde_loss = model(
            ctx_map, obs_seq, coords, global_wind,
            compute_pde_loss=True
        )

        loss_total = (
            loss_mse +
            loss_pcc +
            loss_phys +
            lambda_pde * pde_loss  # Annealed weight
        )
        loss_total.backward()
```

**Annealing Schedule**:
- Epoch 0-30: `λ_pde = 0.1` (데이터 학습 집중)
- Epoch 30-70: `λ_pde = 0.1 → 1.0` (선형 증가)
- Epoch 70-100: `λ_pde = 1.0` (물리 제약 완전 적용)

---

### 4. Adaptive PDE Model (ReLoBRaLo)

```python
from model.model_adaptive_pde import ST_TransformerDeepONet_AdaptivePDE

model = ST_TransformerDeepONet_AdaptivePDE(
    latent_dim=128,
    diffusion_coeff=0.1,
    lookback=10  # Loss history 길이
)

# Training loop
for epoch in range(total_epochs):
    for batch in dataloader:
        pred_wind, pred_conc, pde_loss = model(
            ctx_map, obs_seq, coords, global_wind,
            compute_pde_loss=True
        )

        # 모든 loss 계산
        loss_dict = {
            'mse': loss_mse,
            'pcc': loss_pcc,
            'phys': loss_phys,
            'pde': pde_loss
        }

        # 자동으로 가중치 조정
        weights = model.compute_adaptive_weights(loss_dict)

        # 가중 합산
        loss_total = sum(weights[k] * loss_dict[k] for k in loss_dict)
        loss_total.backward()

        # Logging
        if step % 100 == 0:
            print(f"Adaptive weights: {weights}")
```

**장점**:
- 자동 가중치 조정 (hyperparameter 튜닝 불필요)
- 빠르게 변하는 loss에 낮은 가중치 부여
- 느리게 변하는 loss에 높은 가중치 부여

---

### 5. Hard PDE Model (Output Transform)

```python
from model.model_hard_pde import ST_TransformerDeepONet_HardPDE

model = ST_TransformerDeepONet_HardPDE(
    latent_dim=128,
    diffusion_coeff=0.1,
    use_soft_pde=True  # Hybrid: hard + soft constraints
)

# Training loop
for batch in dataloader:
    pred_wind, pred_conc, pde_loss = model(
        ctx_map, obs_seq, coords, global_wind,
        compute_pde_loss=True,         # Soft PDE loss도 계산 (hybrid)
        apply_hard_constraints=True     # Hard constraints 적용
    )

    loss_total = loss_mse + loss_pcc + loss_phys + pde_loss
    loss_total.backward()
```

**Hard Constraints 강제 사항**:
- ✅ 지형 내부에서 농도 = 0 (자동)
- ✅ 농도 ≥ 0 (항상 non-negative)

**Hybrid 모드** (`use_soft_pde=True`):
- Hard constraints (보장됨) + Soft PDE loss (학습 가이드)
- 더 빠른 수렴 기대

---

## 📊 Ablation Study 실행 예제

```python
# train_ablation.py
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['baseline', 'soft_pde', 'annealed_pde',
                                'adaptive_pde', 'hard_pde'])
    args = parser.parse_args()

    # 모델 선택
    if args.model == 'baseline':
        from model.model_baseline import ST_TransformerDeepONet_Baseline
        model = ST_TransformerDeepONet_Baseline()
    elif args.model == 'soft_pde':
        from model.model_soft_pde import ST_TransformerDeepONet_SoftPDE
        model = ST_TransformerDeepONet_SoftPDE(diffusion_coeff=0.1)
    elif args.model == 'annealed_pde':
        from model.model_annealed_pde import ST_TransformerDeepONet_AnnealedPDE
        model = ST_TransformerDeepONet_AnnealedPDE(total_epochs=100)
    elif args.model == 'adaptive_pde':
        from model.model_adaptive_pde import ST_TransformerDeepONet_AdaptivePDE
        model = ST_TransformerDeepONet_AdaptivePDE(lookback=10)
    elif args.model == 'hard_pde':
        from model.model_hard_pde import ST_TransformerDeepONet_HardPDE
        model = ST_TransformerDeepONet_HardPDE(use_soft_pde=True)

    # Train...
    print(f"Training {args.model} model...")

if __name__ == '__main__':
    main()
```

**실행**:
```bash
# 5가지 모델 모두 학습
python train_ablation.py --model baseline
python train_ablation.py --model soft_pde
python train_ablation.py --model annealed_pde
python train_ablation.py --model adaptive_pde
python train_ablation.py --model hard_pde
```

---

## 📈 성능 비교 지표

학습 후 다음 지표로 비교:

| Metric | 설명 | 목표 |
|--------|------|------|
| **Data MSE** | 예측 vs 실제 농도 MSE | 낮을수록 좋음 |
| **PCC** | Pattern Correlation Coefficient | 높을수록 좋음 |
| **PDE Residual** | PDE 위반 정도 | < 1e-3 |
| **Continuity Loss** | ∇·u divergence | < 1e-3 |
| **Convergence Speed** | 수렴까지 epoch 수 | 적을수록 좋음 |

---

## 🔬 예상 결과

| 모델 | Data MSE | PDE Residual | 수렴 속도 | 장점 |
|------|----------|--------------|----------|------|
| Baseline | 기준 | N/A | 보통 | 구현 단순 |
| Soft PDE | 기준 | ~1e-2 | 느림 | 물리 제약 추가 |
| Annealed PDE | **최저** | ~1e-3 | **빠름** | 안정적 수렴 |
| Adaptive PDE | 낮음 | ~1e-3 | 빠름 | 자동 튜닝 |
| Hard PDE | 보통 | **0 (보장)** | 보통 | 제약 보장 |

**추천 순서**:
1. **Annealed PDE**: 가장 균형 잡힌 성능 예상
2. **Adaptive PDE**: Hyperparameter 튜닝 불필요
3. **Hard PDE**: 제약 만족이 중요한 경우

---

## ⚙️ Hyperparameter 권장값

### Diffusion Coefficient (D)
```python
diffusion_coeff = 0.1  # 대기 난류 (0.01 ~ 0.5 범위)
```

### Annealing Schedule
```python
total_epochs = 100
# Phase 1: 0-30 epochs (30%)
# Phase 2: 30-70 epochs (40%)
# Phase 3: 70-100 epochs (30%)
```

### ReLoBRaLo Lookback
```python
lookback = 10  # 최근 10 step의 loss 변화 추적
```

---

## 🐛 Troubleshooting

### 1. PDE Loss가 NaN
**원인**: Gradient computation 실패

**해결**:
```python
# coords에 requires_grad 설정
coords = make_batch_coords(...).requires_grad_(True)

# Mixed precision 사용 시 autocast 비활성화
with torch.cuda.amp.autocast(enabled=False):
    pde_loss = model.pde_residual(...)
```

### 2. Memory 부족
**원인**: Second derivatives 계산이 메모리 많이 사용

**해결**:
```python
# Batch size 줄이기
batch_size = 4  # 8 → 4

# 또는 coords 샘플링 수 줄이기
coords = make_batch_coords(B, nz=10, ny=22, nx=22)  # 21x45x45 → 10x22x22
```

### 3. Adaptive weights가 불안정
**원인**: Lookback이 너무 짧음

**해결**:
```python
# Lookback 늘리기
model = ST_TransformerDeepONet_AdaptivePDE(lookback=20)  # 10 → 20
```

---

## 📚 참고 논문

1. **Physics-Informed DeepONets** (arXiv:2103.10974)
2. **ReLoBRaLo** (arXiv:2110.09813)
3. **Hard Constraints in PINNs** (arXiv:2306.12749)

---

## ✅ 체크리스트

학습 전 확인:
- [ ] `coords.requires_grad = True` 설정
- [ ] Diffusion coefficient 값 확인 (0.01 ~ 0.5)
- [ ] Total epochs 설정 (Annealed 모델)
- [ ] Lookback 설정 (Adaptive 모델)
- [ ] GPU 메모리 충분한지 확인
- [ ] WandB 또는 TensorBoard 로깅 설정

학습 중 모니터링:
- [ ] PDE loss 감소 추세
- [ ] Data MSE vs PDE loss 균형
- [ ] Adaptive weights 변화 (Adaptive 모델)
- [ ] Annealing schedule 진행 (Annealed 모델)

---

**준비 완료!** 5가지 모델을 바로 학습시켜 ablation study를 진행하세요! 🚀
