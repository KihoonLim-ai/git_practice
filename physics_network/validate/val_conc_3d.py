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

from dataset.dataset import get_time_split_datasets
from dataset.config_param import ConfigParam as Config
from model.model import ST_TransformerDeepONet

# ==========================================
# [사용자 설정]
# ==========================================
CHECKPOINT_PATH = "checkpoints/joint_physics_best.pth"
TARGET_SAMPLE_IDXS = [10, 20] 
SAVE_DIR = "viz_final_debug"

# 시각화 파라미터
GRID_SIZE = (Config.NX, Config.NY, Config.NZ) 
REAL_SCALE = (100.0, 100.0, 10.0) # (dx, dy, dz)
WIND_SKIP = 3        # 바람 화살표 간격
SOURCE_THRESHOLD = 0.01 # 이 값 이상인 모든 지점을 오염원으로 표시

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_data():
    print(f"📂 Loading checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    cfg = checkpoint['config']
    print(f"⚙️ Model Config: Latent={cfg['latent_dim']}, Fourier={cfg['fourier_scale']}")

    model = ST_TransformerDeepONet(
        latent_dim=int(cfg['latent_dim']), 
        dropout=cfg.get('dropout', 0.1),
        fourier_scale=cfg['fourier_scale']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    _, val_ds, _ = get_time_split_datasets(seq_len=30)
    
    try:
        met_stats = np.load(os.path.join(Config.PROCESSED_DIR, Config.SAVE_MET))
        scale_wind = float(met_stats['max_uv'])
    except:
        scale_wind = 1.0
        print("⚠️ Warning: Could not load wind scale. Using 1.0")
        
    return model, val_ds, scale_wind

def find_all_sources(source_map):
    """
    Source Map에서 임계값을 넘는 모든 좌표 반환
    source_map: (Z, Y, X) numpy array
    """
    # 값이 임계값보다 큰 인덱스 찾기 (z, y, x)
    # source_map은 보통 0~1 사이로 정규화되어 있음
    indices = np.argwhere(source_map > SOURCE_THRESHOLD)
    
    # 실제 좌표(m)로 변환
    sources = []
    for idx in indices:
        z, y, x = idx
        sources.append({
            'x': x * REAL_SCALE[0],
            'y': y * REAL_SCALE[1],
            'z': z * REAL_SCALE[2],
            'val': source_map[z, y, x]
        })
    return sources

def visualize_comparison(sample_idx, gt_c, gt_w, pred_c, pred_w, terrain, sources, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    D, H, W = GRID_SIZE[2], GRID_SIZE[1], GRID_SIZE[0]
    Z, Y, X = np.mgrid[0:D, 0:H, 0:W]
    
    X_m = X * REAL_SCALE[0]
    Y_m = Y * REAL_SCALE[1]
    Z_m = Z * REAL_SCALE[2]
    
    Y_terr, X_terr = np.mgrid[0:H, 0:W]
    Y_terr = Y_terr * REAL_SCALE[1]
    X_terr = X_terr * REAL_SCALE[0]
    T_smooth = terrain * (D * REAL_SCALE[2] * 0.5) 

    # --- Debugging Stats ---
    # GT와 Pred의 풍속 통계를 출력하여 모델이 죽었는지(Collapse) 확인
    gt_speed = np.sqrt(np.sum(gt_w**2, axis=-1))
    pred_speed = np.sqrt(np.sum(pred_w**2, axis=-1))
    
    print(f"\n[Sample {sample_idx} Wind Stats]")
    print(f"   - GT   Speed | Max: {gt_speed.max():.4f} m/s, Mean: {gt_speed.mean():.4f} m/s")
    print(f"   - Pred Speed | Max: {pred_speed.max():.4f} m/s, Mean: {pred_speed.mean():.4f} m/s")
    
    # 화살표 크기 정규화를 위한 공통 기준 계산
    max_speed_global = max(gt_speed.max(), pred_speed.max()) + 1e-6

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(f"GT Concentration", f"Pred Concentration", 
                        f"GT Wind Field", f"Pred Wind Field"),
        vertical_spacing=0.1, horizontal_spacing=0.05
    )
    
    # --- Helper: Terrain & Sources ---
    def add_static_elements(row, col):
        # 1. Terrain
        fig.add_trace(go.Surface(
            z=T_smooth, x=X_terr, y=Y_terr,
            colorscale='Earth', opacity=0.8, showscale=False,
            lighting=dict(roughness=0.5, fresnel=0.5), name='Terrain'
        ), row=row, col=col)
        
        # 2. All Sources (수정됨)
        if len(sources) > 0:
            sx = [s['x'] for s in sources]
            sy = [s['y'] for s in sources]
            sz = [s['z'] for s in sources]
            fig.add_trace(go.Scatter3d(
                x=sx, y=sy, z=sz,
                mode='markers', 
                marker=dict(size=6, color='gold', symbol='diamond', line=dict(width=1, color='black')),
                name='Source Points'
            ), row=row, col=col)

    # --- Helper: Data Plotting ---
    def add_volume_conc(data, row, col, name, c_max):
        vmin = c_max * 0.05
        fig.add_trace(go.Volume(
            x=X_m.flatten(), y=Y_m.flatten(), z=Z_m.flatten(),
            value=data.flatten(),
            isomin=vmin, isomax=c_max,
            opacity=0.2, surface_count=15, colorscale='Reds',
            showscale=(col==2),
            colorbar=dict(title='Conc', len=0.4, y=0.8 if row==1 else 0.2),
            name=name
        ), row=row, col=col)

    def add_cone_wind(u, v, w, row, col, name):
        sl = (slice(None,None,WIND_SKIP), slice(None,None,WIND_SKIP), slice(None,None,WIND_SKIP))
        
        # [중요] sizeref를 Global Max Speed에 맞춰 고정 -> GT/Pred 크기 1:1 비교 가능
        # sizemode='scaled'일 때 sizeref가 클수록 화살표가 작아짐
        # 공식: size = value / sizeref
        
        fig.add_trace(go.Cone(
            x=X_m[sl].flatten(), y=Y_m[sl].flatten(), z=Z_m[sl].flatten(),
            u=u[sl].flatten(), v=v[sl].flatten(), w=w[sl].flatten(),
            colorscale='Viridis', 
            sizemode="scaled", 
            sizeref=max_speed_global * 0.5, # 스케일 조정 (값이 작으면 이 계수를 줄여야 함)
            anchor="tail",
            showscale=(col==2),
            colorbar=dict(title='m/s', len=0.4, y=0.8 if row==1 else 0.2),
            name=name
        ), row=row, col=col)

    # --- Plotting ---
    c_max = max(np.max(gt_c), np.max(pred_c))
    
    # Row 1: Conc
    add_volume_conc(gt_c, 1, 1, "GT Conc", c_max)
    add_static_elements(1, 1)
    
    add_volume_conc(pred_c, 1, 2, "Pred Conc", c_max)
    add_static_elements(1, 2)
    
    # Row 2: Wind
    add_cone_wind(gt_w[...,0], gt_w[...,1], gt_w[...,2], 2, 1, "GT Wind")
    add_static_elements(2, 1)
    
    add_cone_wind(pred_w[...,0], pred_w[...,1], pred_w[...,2], 2, 2, "Pred Wind")
    add_static_elements(2, 2)
    
    # Layout
    scene_cfg = dict(
        aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.6),
        xaxis=dict(title='X (m)'), yaxis=dict(title='Y (m)'), zaxis=dict(title='Z (m)')
    )
    
    fig.update_layout(
        title=f"Sample {sample_idx} Analysis (Max Wind: {max_speed_global:.2f} m/s)",
        height=1000, width=1500,
        scene=scene_cfg, scene2=scene_cfg, scene3=scene_cfg, scene4=scene_cfg,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    save_path = os.path.join(save_dir, f"Debug_Sample_{sample_idx}.html")
    fig.write_html(save_path)
    print(f"✅ Saved: {save_path}")

def run_debug():
    model, val_ds, scale_wind = load_model_and_data()
    grid_coords = torch.zeros(1) # Not used in this snippet explicitly but needed for flow
    
    # Grid Coords Generator (Needed for inference if model uses it)
    z = torch.linspace(0, 1, Config.NZ); y = torch.linspace(0, 1, Config.NY); x = torch.linspace(0, 1, Config.NX)
    gz, gy, gx = torch.meshgrid(z, y, x, indexing='ij')
    base_coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).to(DEVICE)
    t = torch.zeros((base_coords.shape[0], 1), device=DEVICE)
    grid_coords = torch.cat([base_coords, t], dim=-1)

    print(f"🚀 Processing {len(TARGET_SAMPLE_IDXS)} samples...")
    
    for idx in TARGET_SAMPLE_IDXS:
        if idx >= len(val_ds): continue
        
        inp_vol, met_seq, target_vol, global_wind = val_ds[idx]
        
        inp_b = inp_vol.unsqueeze(0).to(DEVICE)
        met_b = met_seq.unsqueeze(0).to(DEVICE)
        global_b = global_wind.unsqueeze(0).to(DEVICE)
        coords_b = grid_coords.unsqueeze(0)
        
        with torch.no_grad():
            pred_w_raw, pred_c_raw = model(inp_b, met_b, coords_b, global_b)
            
        # 1. Sources (Find ALL)
        source_map = inp_vol[1, ...].numpy() # (D, H, W)
        sources = find_all_sources(source_map)
        
        # 2. GT Parsing
        gt_c = target_vol[0, ...].numpy()
        gt_w = target_vol[1:4, ...].permute(1, 2, 3, 0).numpy() * scale_wind
        
        # 3. Pred Parsing
        D, H, W = Config.NZ, Config.NY, Config.NX
        pred_c = pred_c_raw.view(D, H, W).cpu().numpy()
        pred_w = pred_w_raw.view(D, H, W, 3).cpu().numpy() * scale_wind
        
        # Clipping
        pred_c = np.maximum(pred_c, 0)
        
        visualize_comparison(idx, gt_c, gt_w, pred_c, pred_w, inp_vol[0].numpy(), sources, SAVE_DIR)

if __name__ == "__main__":
    run_debug()