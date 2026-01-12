"""
Inference script for No-Wind Model
- Loads trained checkpoint
- Runs prediction on test set
- Saves results for visualization
"""
import os
import sys

# OpenMP 중복 로드 문제 해결
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from tqdm import tqdm

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.dataset_no_wind import get_dataloaders_no_wind
from dataset.physics_utils import make_batch_coords
from dataset.config_param import ConfigParam as Config
from model.model_no_wind import SimplifiedDeepONet


class InferenceConfig:
    """추론 설정"""
    CHECKPOINT_PATH = "checkpoints_no_wind/best_no_wind.pth"
    OUTPUT_DIR = "inference_results_no_wind"
    BATCH_SIZE = 4  # 추론 시에는 작은 배치로
    NUM_SAMPLES = 20  # 저장할 샘플 개수 (시각화용)


def load_model(checkpoint_path, device):
    """
    체크포인트에서 모델 로드

    Args:
        checkpoint_path: 체크포인트 파일 경로
        device: torch device

    Returns:
        model: 로드된 모델
        checkpoint: 체크포인트 딕셔너리
    """
    print(f"📂 Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 모델 설정 가져오기
    model_config = checkpoint['config']

    # 모델 초기화
    model = SimplifiedDeepONet(
        latent_dim=model_config['latent_dim'],
        fourier_scale=model_config['fourier_scale'],
        dropout=model_config['dropout']
    ).to(device)

    # 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    print(f"   Best Val Loss: {checkpoint['best_val_loss']:.6f}")

    return model, checkpoint


def run_inference(model, test_loader, device, num_samples=20):
    """
    테스트 데이터셋에 대해 추론 실행

    Args:
        model: 학습된 모델
        test_loader: 테스트 데이터 로더
        device: torch device
        num_samples: 저장할 샘플 개수

    Returns:
        results: 추론 결과 딕셔너리
    """
    print(f"\n🔮 Running inference on test set...")

    results = {
        'predictions': [],
        'targets': [],
        'inputs': [],
        'metrics': {
            'mse': [],
            'mae': [],
            'pcc': []
        }
    }

    model.eval()
    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
            inp_vol, target_vol = [b.to(device) for b in batch]

            # Generate coordinates
            B, C, D, H, W = inp_vol.shape
            coords = make_batch_coords(B, D, H, W, device=device)

            # Forward pass
            pred_conc = model(inp_vol, coords)  # (B, N, 1)

            # Reshape to original volume
            pred_vol = pred_conc.reshape(B, D, H, W)  # (B, 21, 45, 45)

            # Compute metrics per sample
            for i in range(B):
                if sample_count >= num_samples:
                    break

                pred = pred_vol[i].cpu().numpy()  # (21, 45, 45)
                target = target_vol[i, 0].cpu().numpy()  # (21, 45, 45)
                inp = inp_vol[i].cpu().numpy()  # (2, 21, 45, 45)

                # Calculate metrics
                mse = np.mean((pred - target) ** 2)
                mae = np.mean(np.abs(pred - target))

                # Pearson correlation
                pred_flat = pred.flatten()
                target_flat = target.flatten()
                pcc = np.corrcoef(pred_flat, target_flat)[0, 1]

                # Store results
                results['predictions'].append(pred)
                results['targets'].append(target)
                results['inputs'].append(inp)
                results['metrics']['mse'].append(mse)
                results['metrics']['mae'].append(mae)
                results['metrics']['pcc'].append(pcc)

                sample_count += 1

            if sample_count >= num_samples:
                break

    # Convert to numpy arrays
    results['predictions'] = np.array(results['predictions'])
    results['targets'] = np.array(results['targets'])
    results['inputs'] = np.array(results['inputs'])

    # Compute average metrics
    for key in results['metrics']:
        results['metrics'][key] = np.array(results['metrics'][key])

    print(f"\n📊 Inference Results (on {sample_count} samples):")
    print(f"   Average MSE: {results['metrics']['mse'].mean():.6f}")
    print(f"   Average MAE: {results['metrics']['mae'].mean():.6f}")
    print(f"   Average PCC: {results['metrics']['pcc'].mean():.4f}")

    return results


def save_results(results, output_dir):
    """
    추론 결과 저장

    Args:
        results: 추론 결과 딕셔너리
        output_dir: 저장 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n💾 Saving results to: {output_dir}")

    # Save predictions and targets
    np.savez_compressed(
        os.path.join(output_dir, "predictions.npz"),
        predictions=results['predictions'],
        targets=results['targets'],
        inputs=results['inputs'],
        mse=results['metrics']['mse'],
        mae=results['metrics']['mae'],
        pcc=results['metrics']['pcc']
    )

    # Save summary statistics
    summary = {
        'num_samples': len(results['predictions']),
        'mean_mse': float(results['metrics']['mse'].mean()),
        'std_mse': float(results['metrics']['mse'].std()),
        'mean_mae': float(results['metrics']['mae'].mean()),
        'std_mae': float(results['metrics']['mae'].std()),
        'mean_pcc': float(results['metrics']['pcc'].mean()),
        'std_pcc': float(results['metrics']['pcc'].std()),
    }

    import json
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Saved:")
    print(f"   - predictions.npz (predictions, targets, inputs, metrics)")
    print(f"   - summary.json (statistics)")


def main():
    """메인 추론 실행"""
    cfg = InferenceConfig()

    print("=" * 70)
    print("🔮 No-Wind Model Inference")
    print("=" * 70)

    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Using device: {device}")

    # 모델 로드
    model, checkpoint = load_model(cfg.CHECKPOINT_PATH, device)

    # 데이터 로더 생성 (테스트셋만)
    print("\n📦 Loading test data...")
    _, _, test_loader = get_dataloaders_no_wind(
        batch_size=cfg.BATCH_SIZE,
        crop_size=45,  # Full resolution
        num_workers=0
    )

    # 추론 실행
    results = run_inference(
        model=model,
        test_loader=test_loader,
        device=device,
        num_samples=cfg.NUM_SAMPLES
    )

    # 결과 저장
    save_results(results, cfg.OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("🎉 Inference completed!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Run visualization: python visualize_no_wind.py")
    print(f"  2. Check results in: {cfg.OUTPUT_DIR}/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Inference interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
