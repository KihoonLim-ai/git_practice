import os
import sys
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.dataset import get_time_split_datasets
from dataset.config_param import ConfigParam as Config
from model.model import ST_TransformerDeepONet

# [설정]
CHECKPOINT_PATH = "checkpoints/wind_master2.pth"
SAVE_HTML = "wind_vis_global.html"
SKIP = 2 # 화살표가 너무 많으면 2, 자세히 보려면 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_grid_coords(device):
    z = torch.linspace(0, 1, Config.NZ)
    y = torch.linspace(0, 1, Config.NY)
    x = torch.linspace(0, 1, Config.NX)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(device)
    t = torch.zeros((coords.shape[0], 1), device=device)
    return torch.cat([coords, t], dim=-1)

def create_wind_cones(u, v, w, x, y, z, name, show_legend=False):
    speed = np.sqrt(u**2 + v**2 + w**2)
    mask = speed > 0.05 # 아주 약한 바람은 생략
    return go.Cone(
        x=x[mask], y=y[mask], z=z[mask],
        u=u[mask], v=v[mask], w=w[mask],
        colorscale='Jet',
        cmin=0, cmax=np.max(speed), 
        sizemode="absolute",
        sizeref=1.5, # 화살표 크기 조절
        showscale=show_legend,
        colorbar=dict(title='Speed (m/s)', x=1.02) if show_legend else None,
        name=name,
        opacity=0.8
    )

def create_terrain_surface(terrain_2d, nz_dim, max_terr_h, max_domain_h):
    ny, nx = terrain_2d.shape
    real_height_m = terrain_2d * max_terr_h
    z_terrain = (real_height_m / max_domain_h) * (nz_dim - 1)
    x, y = np.arange(nx), np.arange(ny)
    return go.Surface(z=z_terrain, x=x, y=y, colorscale='Earth', showscale=False, opacity=0.9, name='Terrain')

def run_3d_vis():
    print("=== 🌪️ 3D Vis (Global Condition) ===")
    
    # 1. Stats Load
    try:
        met_stats = np.load(os.path.join(Config.PROCESSED_DIR, Config.SAVE_MET))
        scale_wind = float(met_stats['max_uv'])
        map_stats = np.load(os.path.join(Config.PROCESSED_DIR, Config.SAVE_MAPS))
        scale_terr = float(map_stats['terrain_max'])
    except:
        scale_wind = 1.0
        scale_terr = 93.0
    
    max_domain_h = (Config.NZ - 1) * Config.DZ

    # 2. Data & Model
    _, val_ds, _ = get_time_split_datasets(seq_len=30)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = ST_TransformerDeepONet(
        latent_dim=int(checkpoint['config']['latent_dim']), 
        fourier_scale=checkpoint['config']['fourier_scale']
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # 3. Find Windy Sample
    target_batch = None
    idx_in_batch = 0
    
    # 바람이 좀 센 샘플을 찾아야 시각화 효과가 좋음
    for batch in val_loader:
        _, _, tgt, _ = batch # 4개 unpacking
        speeds = torch.sqrt(tgt[:, 1]**2 + tgt[:, 2]**2).mean(dim=(1,2,3))
        if speeds.max() * scale_wind > 3.0: # 평균 3m/s 이상인 배치 선택
            target_batch = batch
            idx_in_batch = torch.argmax(speeds).item()
            break
            
    if target_batch is None: target_batch = next(iter(val_loader))
    
    # [수정] 4개 Unpacking
    inp_vol, met_seq, target_vol, global_wind = [b.to(DEVICE) for b in target_batch]
    
    # 4. Inference
    with torch.no_grad():
        coords = get_grid_coords(DEVICE).unsqueeze(0).expand(inp_vol.shape[0], -1, -1)
        # [수정] global_wind 전달
        pred_w, _ = model(inp_vol, met_seq, coords, global_wind)

    # 5. Plotting
    print(f"   -> Visualizing Sample Index: {idx_in_batch}")
    D, H, W = Config.NZ, Config.NY, Config.NX
    
    u_p = pred_w[idx_in_batch, ..., 0].view(D, H, W).cpu().numpy() * scale_wind
    v_p = pred_w[idx_in_batch, ..., 1].view(D, H, W).cpu().numpy() * scale_wind
    w_p = pred_w[idx_in_batch, ..., 2].view(D, H, W).cpu().numpy() * scale_wind
    
    u_g = target_vol[idx_in_batch, 1].cpu().numpy() * scale_wind
    v_g = target_vol[idx_in_batch, 2].cpu().numpy() * scale_wind
    w_g = target_vol[idx_in_batch, 3].cpu().numpy() * scale_wind

    terrain_2d = val_ds.terrain_2d 

    z_grid, y_grid, x_grid = np.mgrid[0:D:SKIP, 0:H:SKIP, 0:W:SKIP]
    def sub(arr): return arr[::SKIP, ::SKIP, ::SKIP]
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=("Ground Truth", "Prediction (Physics-Informed)")
    )

    fig.add_trace(create_terrain_surface(terrain_2d, D, scale_terr, max_domain_h), row=1, col=1)
    fig.add_trace(create_wind_cones(sub(u_g), sub(v_g), sub(w_g), x_grid, y_grid, z_grid, "GT", True), row=1, col=1)

    fig.add_trace(create_terrain_surface(terrain_2d, D, scale_terr, max_domain_h), row=1, col=2)
    fig.add_trace(create_wind_cones(sub(u_p), sub(v_p), sub(w_p), x_grid, y_grid, z_grid, "Pred"), row=1, col=2)

    scene_layout = dict(
        aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.3),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.6))
    )
    fig.update_layout(title_text="Wind Field Check (Global Condition)", height=800, scene=scene_layout, scene2=scene_layout)
    
    fig.write_html(SAVE_HTML)
    print(f"✅ Saved: {os.path.abspath(SAVE_HTML)}")

if __name__ == "__main__":
    run_3d_vis()