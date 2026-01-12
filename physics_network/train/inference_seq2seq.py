"""
Inference script for Sequence-to-Sequence Concentration Prediction Model
학습된 Seq2Seq 모델을 사용하여 테스트 데이터에 대한 예측 수행

Output:
    - predictions.npz: Predictions, targets, inputs, and metrics
    - summary.json: Overall performance statistics
"""
import os
import sys

# OpenMP 중복 로드 문제 해결
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import json
from tqdm import tqdm

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.dataset_seq2seq import get_dataloaders_seq2seq
from model.model_seq2seq_v2 import ConcentrationSeq2Seq_v2 as ConcentrationSeq2Seq


class InferenceConfig:
    """추론 설정"""
    CHECKPOINT_PATH = "checkpoints_seq2seq/best_seq2seq.pth"
    OUTPUT_DIR = "inference_results_seq2seq"
    BATCH_SIZE = 4  # Reduce for memory efficiency
    NUM_WORKERS = 0


def load_model(checkpoint_path, device):
    """
    체크포인트에서 모델 로드

    Args:
        checkpoint_path: 체크포인트 파일 경로
        device: 디바이스 (cuda/cpu)

    Returns:
        model: 로드된 모델
        config: 학습 시 사용된 config
    """
    print(f"📂 Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Config 복원
    if 'config' in checkpoint:
        config = checkpoint['config']
        hidden_channels = config.get('HIDDEN_CHANNELS', 32)
        num_lstm_layers = config.get('NUM_LSTM_LAYERS', 2)
        output_shape = config.get('OUTPUT_SHAPE', (21, 45, 45))
    else:
        # Default values
        print("⚠️ Config not found in checkpoint, using defaults")
        hidden_channels = 32
        num_lstm_layers = 2
        output_shape = (21, 45, 45)

    # 모델 생성
    model = ConcentrationSeq2Seq(
        hidden_channels=hidden_channels,
        num_lstm_layers=num_lstm_layers,
        output_shape=output_shape
    ).to(device)

    # 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"   Validation loss: {checkpoint.get('val_loss', 'unknown')}")

    return model, checkpoint.get('config', {})


def run_inference(model, test_loader, device):
    """
    테스트 데이터에 대한 추론 실행

    Args:
        model: 학습된 모델
        test_loader: 테스트 데이터 로더
        device: 디바이스

    Returns:
        results: 추론 결과 딕셔너리
    """
    print("\n🔮 Running inference on test set...")

    all_predictions = []
    all_targets = []
    all_past_conc = []
    all_static_maps = []

    mse_list = []
    mae_list = []
    pcc_list = []

    model.eval()

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Inference")

        for past_conc, static_maps, future_conc in pbar:
            # Move to device
            past_conc_gpu = past_conc.to(device)      # (B, 30, 21, H, W)
            static_maps_gpu = static_maps.to(device)  # (B, 2, 21, H, W)
            future_conc_gpu = future_conc.to(device)  # (B, 1, 21, H, W)

            # Forward pass
            pred_conc = model(past_conc_gpu, static_maps_gpu)  # (B, 1, 21, H, W)

            # Move to CPU for metrics calculation
            pred_np = pred_conc.squeeze(1).cpu().numpy()        # (B, 21, H, W)
            target_np = future_conc_gpu.squeeze(1).cpu().numpy()  # (B, 21, H, W)

            # Calculate metrics per sample
            for i in range(pred_np.shape[0]):
                pred_i = pred_np[i]
                target_i = target_np[i]

                # MSE
                mse = np.mean((pred_i - target_i) ** 2)
                mse_list.append(mse)

                # MAE
                mae = np.mean(np.abs(pred_i - target_i))
                mae_list.append(mae)

                # PCC (Pearson Correlation)
                pred_flat = pred_i.flatten()
                target_flat = target_i.flatten()
                pcc = np.corrcoef(pred_flat, target_flat)[0, 1]
                pcc_list.append(pcc)

            # Store results
            all_predictions.append(pred_np)
            all_targets.append(target_np)
            all_past_conc.append(past_conc.cpu().numpy())        # (B, 30, 21, H, W)
            all_static_maps.append(static_maps.cpu().numpy())    # (B, 2, 21, H, W)

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)  # (N, 21, H, W)
    targets = np.concatenate(all_targets, axis=0)          # (N, 21, H, W)
    past_conc_all = np.concatenate(all_past_conc, axis=0)  # (N, 30, 21, H, W)
    static_maps_all = np.concatenate(all_static_maps, axis=0)  # (N, 2, 21, H, W)

    mse_array = np.array(mse_list)
    mae_array = np.array(mae_list)
    pcc_array = np.array(pcc_list)

    print(f"\n✅ Inference completed on {len(predictions)} samples")

    return {
        'predictions': predictions,
        'targets': targets,
        'past_conc': past_conc_all,
        'static_maps': static_maps_all,
        'mse': mse_array,
        'mae': mae_array,
        'pcc': pcc_array
    }


def save_results(results, output_dir):
    """
    추론 결과 저장

    Args:
        results: 추론 결과 딕셔너리
        output_dir: 출력 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions
    save_path = os.path.join(output_dir, "predictions.npz")
    np.savez_compressed(
        save_path,
        predictions=results['predictions'],
        targets=results['targets'],
        past_conc=results['past_conc'],
        static_maps=results['static_maps'],
        mse=results['mse'],
        mae=results['mae'],
        pcc=results['pcc']
    )
    print(f"\n💾 Predictions saved to: {save_path}")

    # Save summary statistics
    summary = {
        'num_samples': len(results['predictions']),
        'prediction_shape': list(results['predictions'].shape),
        'metrics': {
            'mse': {
                'mean': float(results['mse'].mean()),
                'std': float(results['mse'].std()),
                'min': float(results['mse'].min()),
                'max': float(results['mse'].max()),
                'median': float(np.median(results['mse']))
            },
            'mae': {
                'mean': float(results['mae'].mean()),
                'std': float(results['mae'].std()),
                'min': float(results['mae'].min()),
                'max': float(results['mae'].max()),
                'median': float(np.median(results['mae']))
            },
            'pcc': {
                'mean': float(results['pcc'].mean()),
                'std': float(results['pcc'].std()),
                'min': float(results['pcc'].min()),
                'max': float(results['pcc'].max()),
                'median': float(np.median(results['pcc']))
            }
        }
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Summary saved to: {summary_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("📊 Inference Summary")
    print("=" * 70)
    print(f"Total samples: {summary['num_samples']}")
    print(f"Prediction shape: {summary['prediction_shape']}")
    print("\nMetrics:")
    print(f"  MSE:  {summary['metrics']['mse']['mean']:.6f} ± {summary['metrics']['mse']['std']:.6f}")
    print(f"  MAE:  {summary['metrics']['mae']['mean']:.6f} ± {summary['metrics']['mae']['std']:.6f}")
    print(f"  PCC:  {summary['metrics']['pcc']['mean']:.4f} ± {summary['metrics']['pcc']['std']:.4f}")


def main():
    cfg = InferenceConfig()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}\n")

    # Load model
    model, train_config = load_model(cfg.CHECKPOINT_PATH, device)

    # Get test data loader
    print("\n📦 Loading test data...")
    seq_len = train_config.get('SEQ_LEN', 30)
    pred_horizon = train_config.get('PRED_HORIZON', 1)

    _, _, test_loader = get_dataloaders_seq2seq(
        batch_size=cfg.BATCH_SIZE,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        crop_size=45,  # Full resolution for inference
        num_workers=cfg.NUM_WORKERS
    )

    print(f"   Test batches: {len(test_loader)}")

    # Run inference
    results = run_inference(model, test_loader, device)

    # Save results
    save_results(results, cfg.OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("🎉 Inference completed successfully!")
    print(f"   Results saved to: {cfg.OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Inference interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
