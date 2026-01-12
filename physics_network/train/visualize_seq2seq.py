"""
Visualization script for Seq2Seq Model predictions
- Loads inference results
- Creates comparison plots (Prediction vs Ground Truth)
- Visualizes temporal evolution (past 30 hours)
- Generates error maps
- Saves figures
"""
import os
import sys

# OpenMP 중복 로드 문제 해결
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class VisualizationConfig:
    """시각화 설정"""
    RESULTS_DIR = "inference_results_seq2seq"
    OUTPUT_DIR = "figures_seq2seq"
    DPI = 150
    NUM_SAMPLES_TO_PLOT = 5  # 플롯할 샘플 개수
    Z_LEVELS_TO_PLOT = [0, 5, 10, 15, 20]  # 시각화할 고도 레벨


def load_results(results_dir):
    """
    추론 결과 로드

    Args:
        results_dir: 결과 디렉토리

    Returns:
        results: 결과 딕셔너리
    """
    print(f"📂 Loading results from: {results_dir}")

    results_path = os.path.join(results_dir, "predictions.npz")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results not found: {results_path}")

    data = np.load(results_path)

    results = {
        'predictions': data['predictions'],      # (N, 21, 45, 45)
        'targets': data['targets'],              # (N, 21, 45, 45)
        'past_conc': data['past_conc'],          # (N, 30, 21, 45, 45)
        'static_maps': data['static_maps'],      # (N, 2, 21, 45, 45)
        'mse': data['mse'],
        'mae': data['mae'],
        'pcc': data['pcc']
    }

    print(f"✅ Loaded {len(results['predictions'])} samples")
    return results


def plot_sample_comparison(pred, target, static_maps, sample_idx, z_levels, output_dir, dpi=150):
    """
    단일 샘플에 대한 비교 플롯 생성

    Args:
        pred: 예측값 (21, 45, 45)
        target: 실제값 (21, 45, 45)
        static_maps: 정적 맵 (2, 21, 45, 45) [Terrain, Source]
        sample_idx: 샘플 인덱스
        z_levels: 시각화할 고도 레벨 리스트
        output_dir: 저장 디렉토리
        dpi: 해상도
    """
    num_levels = len(z_levels)

    # Create figure
    fig = plt.figure(figsize=(20, 4 * num_levels))
    gs = GridSpec(num_levels, 5, figure=fig, hspace=0.3, wspace=0.3)

    # Global colorbar range (log scale)
    vmin = max(1e-6, min(pred.min(), target.min()))
    vmax = max(pred.max(), target.max())

    for i, z_idx in enumerate(z_levels):
        # 1. Terrain (input)
        ax1 = fig.add_subplot(gs[i, 0])
        terrain_slice = static_maps[0, z_idx, :, :]
        im1 = ax1.imshow(terrain_slice, cmap='terrain', origin='lower')
        ax1.set_title(f'Z={z_idx}: Terrain Mask')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 2. Source (input)
        ax2 = fig.add_subplot(gs[i, 1])
        source_slice = static_maps[1, z_idx, :, :]
        im2 = ax2.imshow(source_slice, cmap='hot', origin='lower')
        ax2.set_title(f'Z={z_idx}: Source Map')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 3. Ground Truth
        ax3 = fig.add_subplot(gs[i, 2])
        target_slice = target[z_idx, :, :]
        im3 = ax3.imshow(target_slice, cmap='viridis', origin='lower',
                        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
        ax3.set_title(f'Z={z_idx}: Ground Truth')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Concentration')

        # 4. Prediction
        ax4 = fig.add_subplot(gs[i, 3])
        pred_slice = pred[z_idx, :, :]
        im4 = ax4.imshow(pred_slice, cmap='viridis', origin='lower',
                        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
        ax4.set_title(f'Z={z_idx}: Prediction')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Concentration')

        # 5. Error Map (Absolute Difference)
        ax5 = fig.add_subplot(gs[i, 4])
        error = np.abs(pred_slice - target_slice)
        im5 = ax5.imshow(error, cmap='Reds', origin='lower')
        ax5.set_title(f'Z={z_idx}: Absolute Error')
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label='|Pred - GT|')

    plt.suptitle(f'Sample {sample_idx} - Multi-Level Comparison', fontsize=16, y=0.995)

    # Save
    save_path = os.path.join(output_dir, f'sample_{sample_idx:03d}_comparison.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"   Saved: {save_path}")


def plot_temporal_evolution(past_conc, target, pred, sample_idx, z_idx, output_dir, dpi=150):
    """
    시간적 진화 플롯 (과거 30시간 + 예측)

    Args:
        past_conc: 과거 농도 (30, 21, 45, 45)
        target: 실제 미래 농도 (21, 45, 45)
        pred: 예측 미래 농도 (21, 45, 45)
        sample_idx: 샘플 인덱스
        z_idx: 고도 레벨
        output_dir: 저장 디렉토리
        dpi: 해상도
    """
    # Extract center point
    cy, cx = 22, 22

    # Time series at center point
    past_series = past_conc[:, z_idx, cy, cx]  # (30,)
    target_value = target[z_idx, cy, cx]
    pred_value = pred[z_idx, cy, cx]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    timesteps = np.arange(-30, 1)  # -30 to 0

    # Past concentration
    ax.plot(timesteps[:-1], past_series, 'b-o', label='Past 30 hours', linewidth=2, markersize=4)

    # Target future
    ax.plot([0], [target_value], 'ro', markersize=10, label='Ground Truth (t=0)', zorder=5)

    # Predicted future
    ax.plot([0], [pred_value], 'g^', markersize=10, label='Prediction (t=0)', zorder=5)

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time (hours relative to prediction)', fontsize=12)
    ax.set_ylabel('Concentration', fontsize=12)
    ax.set_title(f'Sample {sample_idx}: Temporal Evolution at Z={z_idx}, Center (Y={cy}, X={cx})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, f'sample_{sample_idx:03d}_temporal_z{z_idx}.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"   Saved: {save_path}")


def plot_vertical_profile(pred, target, sample_idx, x_pos, y_pos, output_dir, dpi=150):
    """
    특정 위치의 수직 프로파일 플롯

    Args:
        pred: 예측값 (21, 45, 45)
        target: 실제값 (21, 45, 45)
        sample_idx: 샘플 인덱스
        x_pos: X 좌표
        y_pos: Y 좌표
        output_dir: 저장 디렉토리
        dpi: 해상도
    """
    z_levels = np.arange(21)
    pred_profile = pred[:, y_pos, x_pos]
    target_profile = target[:, y_pos, x_pos]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(pred_profile, z_levels, 'b-o', label='Prediction', linewidth=2, markersize=4)
    ax.plot(target_profile, z_levels, 'r--s', label='Ground Truth', linewidth=2, markersize=4)

    ax.set_xlabel('Concentration', fontsize=12)
    ax.set_ylabel('Z Level (Height)', fontsize=12)
    ax.set_title(f'Sample {sample_idx}: Vertical Profile at (X={x_pos}, Y={y_pos})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    save_path = os.path.join(output_dir, f'sample_{sample_idx:03d}_vertical_profile.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"   Saved: {save_path}")


def plot_metrics_distribution(results, output_dir, dpi=150):
    """
    메트릭 분포 히스토그램

    Args:
        results: 결과 딕셔너리
        output_dir: 저장 디렉토리
        dpi: 해상도
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ('mse', 'MSE', 'blue'),
        ('mae', 'MAE', 'green'),
        ('pcc', 'PCC', 'orange')
    ]

    for ax, (key, label, color) in zip(axes, metrics):
        data = results[key]

        ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {data.mean():.4f}')
        ax.axvline(np.median(data), color='blue', linestyle='-.', linewidth=2,
                   label=f'Median: {np.median(data):.4f}')

        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{label} Distribution', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'metrics_distribution.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"   Saved: {save_path}")


def plot_scatter_comparison(pred_all, target_all, output_dir, dpi=150):
    """
    전체 예측값 vs 실제값 산점도

    Args:
        pred_all: 모든 예측값 (N, 21, 45, 45)
        target_all: 모든 실제값 (N, 21, 45, 45)
        output_dir: 저장 디렉토리
        dpi: 해상도
    """
    # Flatten all values
    pred_flat = pred_all.flatten()
    target_flat = target_all.flatten()

    # Subsample for plotting (too many points)
    max_points = 50000
    if len(pred_flat) > max_points:
        indices = np.random.choice(len(pred_flat), max_points, replace=False)
        pred_flat = pred_flat[indices]
        target_flat = target_flat[indices]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter plot
    ax.scatter(target_flat, pred_flat, alpha=0.3, s=1, c='blue')

    # Perfect prediction line
    min_val = max(1e-6, min(target_flat.min(), pred_flat.min()))
    max_val = max(target_flat.max(), pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Ground Truth Concentration', fontsize=12)
    ax.set_ylabel('Predicted Concentration', fontsize=12)
    ax.set_title('Prediction vs Ground Truth (All Points)', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add correlation coefficient
    pcc = np.corrcoef(pred_flat, target_flat)[0, 1]
    ax.text(0.05, 0.95, f'PCC: {pcc:.4f}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    save_path = os.path.join(output_dir, 'scatter_comparison.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"   Saved: {save_path}")


def main():
    """메인 시각화 실행"""
    cfg = VisualizationConfig()

    print("=" * 70)
    print("📊 Seq2Seq Model Visualization")
    print("=" * 70)

    # 결과 로드
    results = load_results(cfg.RESULTS_DIR)

    # 출력 디렉토리 생성
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(f"\n💾 Saving figures to: {cfg.OUTPUT_DIR}")

    # 1. 샘플별 비교 플롯
    print(f"\n📈 Creating sample comparison plots...")
    num_samples = min(cfg.NUM_SAMPLES_TO_PLOT, len(results['predictions']))
    for i in range(num_samples):
        plot_sample_comparison(
            pred=results['predictions'][i],
            target=results['targets'][i],
            static_maps=results['static_maps'][i],
            sample_idx=i,
            z_levels=cfg.Z_LEVELS_TO_PLOT,
            output_dir=cfg.OUTPUT_DIR,
            dpi=cfg.DPI
        )

    # 2. 시간적 진화 플롯
    print(f"\n📈 Creating temporal evolution plots...")
    for i in range(num_samples):
        for z_idx in [5, 10, 15]:  # Selected Z levels
            plot_temporal_evolution(
                past_conc=results['past_conc'][i],
                target=results['targets'][i],
                pred=results['predictions'][i],
                sample_idx=i,
                z_idx=z_idx,
                output_dir=cfg.OUTPUT_DIR,
                dpi=cfg.DPI
            )

    # 3. 수직 프로파일 플롯 (중심점)
    print(f"\n📈 Creating vertical profile plots...")
    for i in range(num_samples):
        plot_vertical_profile(
            pred=results['predictions'][i],
            target=results['targets'][i],
            sample_idx=i,
            x_pos=22,  # Center
            y_pos=22,  # Center
            output_dir=cfg.OUTPUT_DIR,
            dpi=cfg.DPI
        )

    # 4. 메트릭 분포 히스토그램
    print(f"\n📊 Creating metrics distribution plots...")
    plot_metrics_distribution(results, cfg.OUTPUT_DIR, cfg.DPI)

    # 5. 전체 산점도
    print(f"\n📊 Creating scatter comparison plot...")
    plot_scatter_comparison(
        results['predictions'],
        results['targets'],
        cfg.OUTPUT_DIR,
        cfg.DPI
    )

    print("\n" + "=" * 70)
    print("🎉 Visualization completed!")
    print("=" * 70)
    print(f"\nGenerated figures:")
    print(f"  - {num_samples} sample comparison plots")
    print(f"  - {num_samples * 3} temporal evolution plots")
    print(f"  - {num_samples} vertical profile plots")
    print(f"  - 1 metrics distribution plot")
    print(f"  - 1 scatter comparison plot")
    print(f"\nCheck results in: {cfg.OUTPUT_DIR}/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Visualization interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
