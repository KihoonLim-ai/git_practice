import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# [Path Setup]
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.dataset import get_dataloaders
from dataset.physics_utils import make_batch_coords
from dataset.config_param import ConfigParam as Config
from model.model import ST_TransformerDeepONet

# ==========================================
# [Configuration]
# ==========================================
CHECKPOINT_PATH = "checkpoints/joint_best_now.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# [Normalization Statistics]
GLOBAL_MEAN_LOG = 1.0229
GLOBAL_STD_LOG = 1.2663

# [Thresholds]
# Threshold for Log Scale (Normalized)
LOG_THRESHOLD = 0.05 
# Threshold for Real Scale (ppm or physical unit)
REAL_THRESHOLD = 0.05 

def load_wind_scale():
    """Load Wind Scaling Factor from metadata"""
    try:
        met_stats = np.load(os.path.join(Config.PROCESSED_DIR, Config.SAVE_MET))
        return float(met_stats['max_uv'])
    except:
        print("⚠️ Warning: Wind scale not found. Using 1.0")
        return 1.0

def denormalize_conc(conc_log_norm):
    """
    Convert Log-Normalized concentration to Real Scale
    Formula: x_real = exp(x_norm * std + mean) - 1
    """
    conc_log = conc_log_norm * GLOBAL_STD_LOG + GLOBAL_MEAN_LOG
    conc_real = torch.expm1(conc_log)
    return torch.clamp(conc_real, min=0.0)

def calculate_metrics(pred, obs, threshold):
    """
    Calculate RMSE, IOA, FMS
    """
    # 1. RMSE
    mse = torch.mean((pred - obs)**2)
    rmse = torch.sqrt(mse)
    
    # 2. IOA (Index of Agreement)
    obs_mean = torch.mean(obs)
    numerator = torch.sum((obs - pred)**2)
    denominator = torch.sum((torch.abs(pred - obs_mean) + torch.abs(obs - obs_mean))**2)
    ioa = 1.0 - (numerator / (denominator + 1e-7))
    
    # 3. FMS (Figure of Merit in Space)
    pred_mask = (pred > threshold).float()
    obs_mask = (obs > threshold).float()
    intersection = torch.sum(pred_mask * obs_mask)
    union = torch.sum(torch.max(pred_mask, obs_mask))
    
    if union == 0: 
        fms = 1.0
    else:
        fms = intersection / (union + 1e-7)
        
    return rmse.item(), ioa.item(), fms.item()

def calculate_wind_metrics(pred_wind, gt_wind):
    """Wind Accuracy Evaluation (Real Scale)"""
    # 1. RMSE (Magnitude Error)
    mse = torch.mean((pred_wind - gt_wind)**2)
    rmse = torch.sqrt(mse)
    
    # 2. Cosine Similarity (Direction Accuracy)
    dot_product = torch.sum(pred_wind * gt_wind, dim=1)
    norm_pred = torch.norm(pred_wind, dim=1)
    norm_gt = torch.norm(gt_wind, dim=1)
    
    cos_sim = dot_product / (norm_pred * norm_gt + 1e-6)
    avg_cos = torch.mean(cos_sim)
    
    return rmse.item(), avg_cos.item()

def run_evaluation_loop(model, mode_name, future_step, scale_wind):
    print(f"\n🚀 Evaluating Mode: {mode_name} (Step +{future_step})")
    
    # Load Test Set
    _, _, test_loader = get_dataloaders(
        batch_size=32, 
        seq_len=30, 
        future_step=future_step,
        crop_size=45
    )
    
    # Accumulators for Dual Mode
    metrics = {
        'w_rmse': 0.0, 'w_cossim': 0.0,
        # Log Scale Metrics
        'log_rmse': 0.0, 'log_ioa': 0.0, 'log_fms': 0.0,
        # Real Scale Metrics
        'real_rmse': 0.0, 'real_ioa': 0.0, 'real_fms': 0.0
    }
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval {mode_name}"):
            inp_vol, met_seq, target_vol, global_wind = [b.to(DEVICE) for b in batch]
            
            B, C, D, H, W = inp_vol.shape
            coords = make_batch_coords(B, D, H, W, device=DEVICE)
            
            # 1. Inference
            pred_wind_norm, pred_conc_norm = model(inp_vol, met_seq, coords, global_wind)
            
            # --------------------------------------------------------
            # A. Wind Evaluation (Always Real Scale)
            # --------------------------------------------------------
            pred_wind_real = pred_wind_norm.permute(0, 2, 1).view(B, 3, D, H, W) * scale_wind
            if target_vol.shape[1] >= 4:
                gt_wind_real = target_vol[:, 1:4, ...] * scale_wind
            else:
                gt_wind_real = inp_vol[:, 2:5, ...] * scale_wind
                
            w_rmse, w_cos = calculate_wind_metrics(pred_wind_real, gt_wind_real)
            metrics['w_rmse'] += w_rmse
            metrics['w_cossim'] += w_cos

            # --------------------------------------------------------
            # B. Concentration Evaluation (Dual Mode)
            # --------------------------------------------------------
            # Data Preparation
            pred_log = pred_conc_norm.view(B, D, H, W)
            gt_log = target_vol[:, 0, ...] 
            
            # 1. Log Scale Evaluation (Normalized)
            l_rmse, l_ioa, l_fms = calculate_metrics(pred_log, gt_log, LOG_THRESHOLD)
            metrics['log_rmse'] += l_rmse
            metrics['log_ioa'] += l_ioa
            metrics['log_fms'] += l_fms
            
            # 2. Real Scale Evaluation (Denormalized)
            pred_real = denormalize_conc(pred_log)
            gt_real = denormalize_conc(gt_log)
            
            r_rmse, r_ioa, r_fms = calculate_metrics(pred_real, gt_real, REAL_THRESHOLD)
            metrics['real_rmse'] += r_rmse
            metrics['real_ioa'] += r_ioa
            metrics['real_fms'] += r_fms
            
            count += 1
            
    # Average
    for k in metrics:
        metrics[k] /= count
        
    print(f"   📊 Results for [{mode_name}]")
    print(f"      [Wind Field]")
    print(f"      - RMSE (Speed)     : {metrics['w_rmse']:.4f} m/s")
    print(f"      - CosSim (Dir)     : {metrics['w_cossim']:.4f}")
    print(f"      ------------------------------------------------")
    print(f"      [Conc - Log Scale] (Pattern)")
    print(f"      - RMSE             : {metrics['log_rmse']:.4f}")
    print(f"      - IOA              : {metrics['log_ioa']:.4f}")
    print(f"      - FMS (Overlap)    : {metrics['log_fms']*100:.2f} %")
    print(f"      ------------------------------------------------")
    print(f"      [Conc - Real Scale] (Physical)")
    print(f"      - RMSE             : {metrics['real_rmse']:.4f}")
    print(f"      - IOA              : {metrics['real_ioa']:.4f}")
    print(f"      - FMS (Overlap)    : {metrics['real_fms']*100:.2f} %")
    
    return metrics

def evaluate_dual_mode():
    print(f"=== 🧬 Final Dual-Mode Evaluation (Log & Real) ===")
    
    scale_wind = load_wind_scale()
    print(f"   -> Wind Scale: {scale_wind:.2f} m/s")
    print(f"   -> Conc Stats: Mean={GLOBAL_MEAN_LOG}, Std={GLOBAL_STD_LOG}")
    
    # Model Load
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = ST_TransformerDeepONet(
        latent_dim=int(checkpoint['config']['latent_dim']), 
        fourier_scale=checkpoint['config']['fourier_scale']
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Run Eval
    m_curr = run_evaluation_loop(model, "Reconstruction (t=0)", 0, scale_wind)
    m_fut = run_evaluation_loop(model, "Prediction (t+1)", 1, scale_wind)
    
    print("\n" + "="*80)
    print("      🏆 FINAL PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Metric':<20} | {'Reconstruction':<25} | {'Prediction':<25}")
    print("-" * 80)
    print(f"{'Wind RMSE':<20} | {m_curr['w_rmse']:.4f} m/s {'':<16} | {m_fut['w_rmse']:.4f} m/s")
    print(f"{'Wind CosSim':<20} | {m_curr['w_cossim']:.4f} {'':<20} | {m_fut['w_cossim']:.4f}")
    print("-" * 80)
    print(f"{'Log IOA (Pattern)':<20} | {m_curr['log_ioa']:.4f} {'':<20} | {m_fut['log_ioa']:.4f}")
    print(f"{'Log FMS (Overlap)':<20} | {m_curr['log_fms']*100:.1f}% {'':<19} | {m_fut['log_fms']*100:.1f}%")
    print("-" * 80)
    print(f"{'Real RMSE (Value)':<20} | {m_curr['real_rmse']:.4f} {'':<20} | {m_fut['real_rmse']:.4f}")
    print(f"{'Real IOA (Value)':<20} | {m_curr['real_ioa']:.4f} {'':<20} | {m_fut['real_ioa']:.4f}")
    print(f"{'Real FMS (Overlap)':<20} | {m_curr['real_fms']*100:.1f}% {'':<19} | {m_fut['real_fms']*100:.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    evaluate_dual_mode()