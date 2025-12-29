import os
import sys
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from dataset.dataset import get_time_split_datasets
from model import ST_TransformerDeepONet

# 설정
CHECKPOINT_PATH = "./train/checkpoints/model_fearless-sweep-1_best.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_raw_stats():
    print("=== Model Output Distribution Check ===")
    
    # 1. 데이터 & 모델 로드
    _, val_ds, _ = get_time_split_datasets(seq_len=30, pred_step=5)
    
    # 체크포인트 로드
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    loaded_dim = 128
    if 'config' in checkpoint:
        conf = checkpoint['config']
        loaded_dim = conf.get('latent_dim', 128) if isinstance(conf, dict) else getattr(conf, 'latent_dim', 128)
    
    model = ST_TransformerDeepONet(latent_dim=loaded_dim, dropout=0.0).to(DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 2. 샘플 하나 추론
    idx = 50
    ctx, met, coords, _, gt_c = val_ds[idx]
    
    # 배치 추가
    ctx = ctx.unsqueeze(0).to(DEVICE)
    met = met.unsqueeze(0).to(DEVICE)
    coords = coords.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        _, pred_c = model(ctx, met, coords)
        
    pred_raw = pred_c.squeeze().cpu().numpy()
    gt_raw = gt_c.numpy()
    
    # 3. 통계 비교 (Z-score 상태)
    print(f"\n[Raw Z-score Statistics]")
    print(f"Pred Range: {pred_raw.min():.4f} ~ {pred_raw.max():.4f} (Mean: {pred_raw.mean():.4f})")
    print(f"GT   Range: {gt_raw.min():.4f} ~ {gt_raw.max():.4f} (Mean: {gt_raw.mean():.4f})")
    
    # 4. 복원 후 비교
    c_mean, c_std = val_ds.conc_mean, val_ds.conc_std
    pred_phys = np.maximum(np.expm1(pred_raw * c_std + c_mean), 0)
    gt_phys = np.maximum(np.expm1(gt_raw * c_std + c_mean), 0)
    
    print(f"\n[Physical (ppm) Statistics]")
    print(f"Pred Range: {pred_phys.min():.4f} ~ {pred_phys.max():.4f} (Mean: {pred_phys.mean():.4f})")
    print(f"GT   Range: {gt_phys.min():.4f} ~ {gt_phys.max():.4f} (Mean: {gt_phys.mean():.4f})")

if __name__ == "__main__":
    check_raw_stats()