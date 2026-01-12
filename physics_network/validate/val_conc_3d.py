import os
import sys
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# [경로 설정]
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.dataset import get_dataloaders
from dataset.physics_utils import make_batch_coords
from dataset.config_param import ConfigParam as Config
from model.model import ST_TransformerDeepONet

# ==========================================
# [사용자 설정]
# ==========================================
CHECKPOINT_PATH = "checkpoints/joint_best_now.pth"
TARGET_SAMPLE_IDXS = [0, 5, 10]
SAVE_DIR = "viz_final_dual_mode"

# [물리 그리드]
GRID_DX = 100.0
GRID_DY = 100.0
GRID_DZ = 10.0
WIND_SKIP = 3

# [통계값: 역정규화용]
GLOBAL_MEAN_LOG = 1.0229
GLOBAL_STD_LOG = 1.2663

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def denormalize_conc(conc_log_norm):
    """Log-Normalized -> Real Scale (ppm)"""
    conc_log = conc_log_norm * GLOBAL_STD_LOG + GLOBAL_MEAN_LOG
    conc_real = np.expm1(conc_log)
    return np.maximum(conc_real, 0)

def load_model_and_dataset():
    print(f"📂 Loading checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    saved_cfg = checkpoint.get('config', {})
    
    model = ST_TransformerDeepONet(
        latent_dim=int(saved_cfg.get('latent_dim', 128)), 
        fourier_scale=float(saved_cfg.get('fourier_scale', 10.0))
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # future_step=1 (t+1 예측)
    _, _, test_loader = get_dataloaders(batch_size=1, seq_len=30, future_step=1, crop_size=45)
    
    try:
        met_stats = np.load(os.path.join(Config.PROCESSED_DIR, Config.SAVE_MET))
        scale_wind = float(met_stats['max_uv'])
    except:
        scale_wind = 1.0
        
    return model, test_loader.dataset, scale_wind

def create_figure(sample_idx, gt_c, gt_w, pred_c, pred_w, terrain_voxel, source_map, mode_name):
    """
    Plotly Figure 생성 함수 (Log/Real 모드 공용)
    """
    D, H, W = terrain_voxel.shape
    
    # Grid Setup
    z_r = np.arange(D) * GRID_DZ
    y_r = np.arange(H) * GRID_DY
    x_r = np.arange(W) * GRID_DX
    Z_m, Y_m, X_m = np.meshgrid(z_r, y_r, x_r, indexing='ij')
    
    T_real = np.sum(terrain_voxel, axis=0) * GRID_DZ
    Y_t, X_t = np.meshgrid(y_r, x_r, indexing='ij')
    
    # Scaling Calculation (GT와 Pred의 Max를 맞춰야 공정함)
    # 농도 Max
    c_max = max(gt_c.max(), pred_c.max())
    if c_max < 1e-6: c_max = 1.0
    
    # 풍속 Max
    gt_s = np.sqrt(np.sum(gt_w**2, axis=-1))
    pd_s = np.sqrt(np.sum(pred_w**2, axis=-1))
    s_max = max(gt_s.max(), pd_s.max()) + 1e-6
    cone_scale = s_max * 0.4
    
    # Subplot Setup
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(f"[GT] Source + {mode_name} Conc", 
                        f"[Pred] Source + {mode_name} Conc", 
                        f"[GT] Wind", 
                        f"[Pred] Wind"),
        vertical_spacing=0.05, horizontal_spacing=0.02
    )
    
    # --- Helper Functions ---
    def add_terrain(row, col):
        fig.add_trace(go.Surface(
            z=T_real, x=X_t, y=Y_t,
            colorscale=[[0, 'rgb(50,50,50)'], [1, 'rgb(200,200,200)']],
            opacity=0.6, showscale=False,
            contours_z=dict(show=True, usecolormap=False, highlightcolor="black", project_z=True, width=1),
            name='Terrain'
        ), row=row, col=col)

    def add_vol(data, row, col, name, cmap, opacity, is_source=False):
        flat = data.flatten()
        if flat.max() < 1e-8: return
        
        # Colorbar 설정 (GT는 왼쪽, Pred는 오른쪽)
        show_cbar = False
        cbar = None
        
        if is_source:
            vmin, vmax = flat.max()*0.1, flat.max()
        else:
            vmin, vmax = c_max * 0.05, c_max
            show_cbar = True
            # [수정] GT에도 바 추가
            x_pos = 0.46 if col == 1 else 1.02
            cbar = dict(title=f"{mode_name}", len=0.4, y=0.8, x=x_pos)

        fig.add_trace(go.Volume(
            x=X_m.flatten(), y=Y_m.flatten(), z=Z_m.flatten(),
            value=flat,
            isomin=vmin, isomax=vmax,
            opacity=opacity, surface_count=10,
            colorscale=cmap,
            showscale=show_cbar,
            colorbar=cbar,
            cmin=0 if not is_source else None,
            cmax=c_max if not is_source else None, # Scale Lock
            name=name
        ), row=row, col=col)

    def add_wind(u, v, w, row, col, name):
        sl = (slice(None,None,WIND_SKIP), slice(None,None,WIND_SKIP), slice(None,None,WIND_SKIP))
        
        # [수정] GT에도 바 추가
        x_pos = 0.46 if col == 1 else 1.02
        cbar = dict(title='m/s', len=0.4, y=0.2, x=x_pos)
        
        fig.add_trace(go.Cone(
            x=X_m[sl].flatten(), y=Y_m[sl].flatten(), z=Z_m[sl].flatten(),
            u=u[sl].flatten(), v=v[sl].flatten(), w=w[sl].flatten(),
            colorscale='Blues', sizemode="scaled", sizeref=cone_scale,
            anchor="tail",
            showscale=True,
            colorbar=cbar,
            cmin=0, cmax=s_max, # Scale Lock
            name=name
        ), row=row, col=col)

    # --- Draw Traces ---
    gold = [[0, 'gold'], [1, 'gold']]
    
    # Row 1: Conc
    add_terrain(1, 1); add_vol(source_map, 1, 1, "Src", gold, 0.3, True)
    add_vol(gt_c, 1, 1, "GT", "Reds", 0.4)
    
    add_terrain(1, 2); add_vol(source_map, 1, 2, "Src", gold, 0.3, True)
    add_vol(pred_c, 1, 2, "Pred", "Blues", 0.4)
    
    # Row 2: Wind
    add_terrain(2, 1); add_wind(gt_w[...,0], gt_w[...,1], gt_w[...,2], 2, 1, "GT Wind")
    add_terrain(2, 2); add_wind(pred_w[...,0], pred_w[...,1], pred_w[...,2], 2, 2, "Pred Wind")
    
    # Layout
    ar_z = (D * GRID_DZ) / (W * GRID_DX) * 3.0
    cam = dict(eye=dict(x=1.3, y=1.3, z=0.8))
    scene_cfg = dict(camera=cam, aspectmode='manual', aspectratio=dict(x=1, y=1, z=ar_z),
                     xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z'))
    
    fig.update_layout(
        title=f"Sample {sample_idx} | Mode: {mode_name} | t+1 Prediction",
        height=1000, width=1600,
        scene=scene_cfg, scene2=scene_cfg, scene3=scene_cfg, scene4=scene_cfg,
        margin=dict(t=50, l=0, r=0, b=0)
    )
    return fig

def run_viz():
    os.makedirs(SAVE_DIR, exist_ok=True)
    model, val_ds, scale_wind = load_model_and_dataset()
    print(f"🚀 Processing {len(TARGET_SAMPLE_IDXS)} samples...")
    
    for idx in TARGET_SAMPLE_IDXS:
        if idx >= len(val_ds): continue
        print(f"  > Processing Sample {idx}...")
        
        # 1. Inference
        inp, met, tgt, g_wind = val_ds[idx]
        
        inp_b = inp.unsqueeze(0).to(DEVICE)
        met_b = met.unsqueeze(0).to(DEVICE)
        g_wind_b = g_wind.unsqueeze(0).to(DEVICE)
        B, C, D, H, W = inp_b.shape
        coords = make_batch_coords(B, D, H, W, device=DEVICE)
        
        with torch.no_grad():
            pw_flat, pc_flat = model(inp_b, met_b, coords, g_wind_b)
            
        # 2. Extract Raw Data
        terr = inp[0].numpy()
        src = inp[1].numpy()
        
        # Log Scale Data
        gt_c_log = tgt[0].numpy()
        pred_c_log = pc_flat.view(D, H, W).cpu().numpy()
        # Wind is same for both (already real scale in previous logic, ensuring consistent m/s)
        gt_w = inp[2:5].permute(1, 2, 3, 0).numpy() * scale_wind
        pred_w = pw_flat.view(D, H, W, 3).cpu().numpy() * scale_wind

        # ---------------------------------------------------------
        # [Mode 1] Log Scale Visualization (Normalized)
        # ---------------------------------------------------------
        fig_log = create_figure(idx, gt_c_log, gt_w, pred_c_log, pred_w, terr, src, "LogScale")
        fig_log.write_html(os.path.join(SAVE_DIR, f"Sample_{idx}_Log.html"))
        
        # ---------------------------------------------------------
        # [Mode 2] Real Scale Visualization (Denormalized)
        # ---------------------------------------------------------
        gt_c_real = denormalize_conc(gt_c_log)
        pred_c_real = denormalize_conc(pred_c_log)
        
        fig_real = create_figure(idx, gt_c_real, gt_w, pred_c_real, pred_w, terr, src, "RealScale")
        fig_real.write_html(os.path.join(SAVE_DIR, f"Sample_{idx}_Real.html"))

    print(f"✅ All saved in '{SAVE_DIR}'")

if __name__ == "__main__":
    run_viz()