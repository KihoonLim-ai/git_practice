import os
import sys
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

# [경로 설정]
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.dataset import get_time_split_datasets
from dataset.config_param import ConfigParam as Config
from model.model import ST_TransformerDeepONet

# ==========================================
# [사용자 설정]
# ==========================================
# 분석하고 싶은 모델 체크포인트 경로
CHECKPOINT_PATH = "checkpoints/model_winter-sweep-1_ep40.pth"
# 보고 싶은 샘플의 인덱스 (Batch 내 순서 아님, Validation Set 기준 순서)
TARGET_SAMPLE_IDXS = [0, 10, 20] 

# 시각화 설정
VIS_Z_MULT = 2.0       # Z축 과장 (가시성)
GRID_SIZE = (45, 45, 21) 
REAL_SCALE = (100.0, 100.0, 10.0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_data():
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Config 로드 (없을 경우 수동 설정 fallback)
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        cfg = checkpoint['config']
        state_dict = checkpoint['model_state_dict']
    else:
        print("⚠️ Warning: No config found. Using MANUAL settings.")
        cfg = {'latent_dim': 128, 'dropout': 0.1, 'fourier_scale': 10.0}
        state_dict = checkpoint

    model = ST_TransformerDeepONet(
        latent_dim=int(cfg['latent_dim']), 
        dropout=cfg['dropout'],
        fourier_scale=cfg['fourier_scale']
    ).to(DEVICE)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # 데이터셋 로드
    _, val_ds, stats = get_time_split_datasets(seq_len=30, pred_step=5)
    
    return model, val_ds, stats

def get_grid_coords(device):
    z = torch.linspace(0, 1, Config.NZ)
    y = torch.linspace(0, 1, Config.NY)
    x = torch.linspace(0, 1, Config.NX)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(device)
    t = torch.zeros((coords.shape[0], 1), device=device)
    return torch.cat([coords, t], dim=-1)

def visualize_comparison(sample_idx, gt_c, gt_w, pred_c, pred_w, terrain, stats, save_dir="viz_results"):
    """
    2x2 Grid Visualization
    [GT Conc]   [GT Wind]
    [Pred Conc] [Pred Wind]
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 좌표 생성
    D, H, W = GRID_SIZE[2], GRID_SIZE[1], GRID_SIZE[0]
    Z, Y, X = np.mgrid[0:D, 0:H, 0:W]
    
    X_m = X * REAL_SCALE[0]
    Y_m = Y * REAL_SCALE[1]
    Z_m = Z * REAL_SCALE[2] * VIS_Z_MULT
    
    # 지형 좌표
    Y_terr, X_terr = np.mgrid[0:H, 0:W]
    Y_terr = Y_terr * REAL_SCALE[1]
    X_terr = X_terr * REAL_SCALE[0]
    T_smooth = terrain * VIS_Z_MULT # (H, W)
    
    # Figure 생성
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(f"GT Concentration", f"GT Wind Field", 
                        f"Pred Concentration", f"Pred Wind Field"),
        vertical_spacing=0.05
    )
    
    # --- Helper Functions ---
    def add_terrain(row, col):
        fig.add_trace(go.Surface(
            z=T_smooth, x=X_terr, y=Y_terr,
            colorscale='Earth', opacity=0.3, showscale=False, name='Terrain'
        ), row=row, col=col)

    def add_volume(data, row, col, name, vmax):
        # 농도 1% 이하는 투명하게 처리
        vmin = vmax * 0.01 
        fig.add_trace(go.Volume(
            x=X_m.flatten(), y=Y_m.flatten(), z=Z_m.flatten(),
            value=data.flatten(),
            isomin=vmin, isomax=vmax,
            opacity=0.15, surface_count=20, colorscale='Jet',
            colorbar=dict(title='ppm', len=0.4, x=0.45 if col==1 else 1.0, y=0.8 if row==1 else 0.2),
            name=name
        ), row=row, col=col)

    def add_wind(u, v, w, row, col, name):
        # Stride 적용 (너무 빽빽하지 않게)
        stride = 4
        sl = (slice(None,None,stride), slice(None,None,stride), slice(None,None,stride))
        
        # W 성분 과장 (지형 효과 확인용)
        w_scaled = w * (VIS_Z_MULT * 2.0) 
        
        fig.add_trace(go.Cone(
            x=X_m[sl].flatten(), y=Y_m[sl].flatten(), z=Z_m[sl].flatten(),
            u=u[sl].flatten(), v=v[sl].flatten(), w=w_scaled[sl].flatten(),
            colorscale='Viridis', sizemode="scaled", sizeref=2.0, anchor="tail",
            colorbar=dict(title='m/s', len=0.4, x=1.0, y=0.8 if row==1 else 0.2),
            name=name
        ), row=row, col=col)

    # --- Plotting ---
    # Max 값 기준으로 스케일 통일 (비교를 위해 GT 기준)
    c_max = max(np.max(gt_c), np.max(pred_c))
    
    # 1. Top-Left: GT Conc
    add_volume(gt_c, 1, 1, "GT Conc", c_max)
    add_terrain(1, 1)
    
    # 2. Top-Right: GT Wind
    add_wind(gt_w[...,0], gt_w[...,1], gt_w[...,2], 1, 2, "GT Wind")
    add_terrain(1, 2)
    
    # 3. Bottom-Left: Pred Conc
    add_volume(pred_c, 2, 1, "Pred Conc", c_max)
    add_terrain(2, 1)
    
    # 4. Bottom-Right: Pred Wind
    add_wind(pred_w[...,0], pred_w[...,1], pred_w[...,2], 2, 2, "Pred Wind")
    add_terrain(2, 2)
    
    # Layout
    scene_cfg = dict(
        aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5 * VIS_Z_MULT),
        xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')
    )
    fig.update_layout(
        title=f"Result Analysis (Sample {sample_idx})",
        height=1200, width=1400,
        scene=scene_cfg, scene2=scene_cfg, scene3=scene_cfg, scene4=scene_cfg
    )
    
    save_path = os.path.join(save_dir, f"Analysis_Sample_{sample_idx}.html")
    fig.write_html(save_path)
    print(f"   -> Saved: {save_path}")

def run_analysis():
    # 1. Load
    model, val_ds, stats = load_model_and_data()
    grid_coords = get_grid_coords(DEVICE)
    
    # Stats Unpacking
    scale_wind = stats['scale_wind']
    scale_terr = stats['terrain_max']
    # dataset 속성이나 stats에서 가져옴
    conc_mean = getattr(val_ds, 'conc_mean', 0.0)
    conc_std = getattr(val_ds, 'conc_std', 1.0)
    
    print(f"Processing {len(TARGET_SAMPLE_IDXS)} samples...")
    
    for idx in TARGET_SAMPLE_IDXS:
        # 데이터 하나 가져오기 (Collate 없이 직접)
        # return: (input, met, target)
        inp_t, met_t, tgt_t = val_ds[idx]
        
        # Batch 차원 추가
        inp_b = inp_t.unsqueeze(0).to(DEVICE)
        met_b = met_t.unsqueeze(0).to(DEVICE)
        coords_b = grid_coords.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            pred_w_raw, pred_c_raw = model(inp_b, met_b, coords_b)
            
        # --- Data Parsing & Denormalization ---
        D, H, W = GRID_SIZE[2], GRID_SIZE[1], GRID_SIZE[0]
        
        # 1. Terrain (Input의 0번 채널)
        # 지형 마스크는 0/1이므로, 원본 높이맵은 dataset에서 따로 가져오는 게 좋음
        # 하지만 시각화용으로는 val_ds.terrain_2d 사용
        terrain_h = val_ds.terrain_2d * scale_terr
        
        # 2. Wind (Denormalize)
        # Pred: (1, N, 3) -> (D, H, W, 3)
        pred_w = pred_w_raw.view(D, H, W, 3).cpu().numpy() * scale_wind
        
        # GT: tgt_t shape (4, D, H, W) -> (Ch 1,2,3 correspond to u,v,w)
        # 주의: GT에는 w가 0으로 들어있음
        gt_w_ch = tgt_t[1:4].permute(1, 2, 3, 0).numpy() # (D, H, W, 3)
        gt_w = gt_w_ch * scale_wind
        
        # 3. Conc (Denormalize Log+Norm)
        # Pred
        pred_c_norm = pred_c_raw.view(D, H, W).cpu().numpy()
        pred_c = np.expm1(pred_c_norm * conc_std + conc_mean)
        pred_c = np.maximum(pred_c, 0)
        
        # GT
        gt_c_norm = tgt_t[0].numpy() # (D, H, W)
        gt_c = np.expm1(gt_c_norm * conc_std + conc_mean)
        gt_c = np.maximum(gt_c, 0)
        
        # Visualize
        visualize_comparison(idx, gt_c, gt_w, pred_c, pred_w, terrain_h, stats)

if __name__ == "__main__":
    run_analysis()