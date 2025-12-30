import os
import sys
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

# [사용자 모듈 임포트]
from dataset.dataset import get_time_split_datasets
from model import ST_TransformerDeepONet 

# ==========================================
# [설정]
# ==========================================
CHECKPOINT_PATH = "./train/checkpoints/model_effortless-sweep-5_best.pth" # 모델 경로 확인
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_INDEX = 10 # 분석할 샘플 인덱스
GRID_SIZE = (45, 45, 21) # [수정됨] 실제 데이터 크기에 맞춤 (X, Y, Z)
REAL_SCALE = (100.0, 100.0, 10.0) # 시각화용 축 스케일

def load_model_and_data():
    _, val_ds, _ = get_time_split_datasets(seq_len=30, pred_step=5)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    config = checkpoint['config']
    
    # 모델 생성 (기존에 쓰던 모델 구조에 맞춰 인자 전달)
    model = ST_TransformerDeepONet(
        latent_dim=config.get('latent_dim', 128),
        fourier_scale=config.get('fourier_scale', 10.0)
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, val_ds

def predict_3d_volume(model, dataset, sample_idx):
    ctx_map, met_seq, coords, gt_w, gt_c = dataset[sample_idx]
    
    # 배치 차원 추가
    ctx_map = ctx_map.unsqueeze(0).to(DEVICE)
    met_seq = met_seq.unsqueeze(0).to(DEVICE)
    coords = coords.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred_w, pred_c = model(ctx_map, met_seq, coords)
    
    # Reshape (Z, Y, X 순서로 가정하고 변환 후 Transpose 필요시 조정)
    # 보통 데이터셋 생성 방식에 따라 다를 수 있으나, 일단 순차적으로 채움
    depth, height, width = GRID_SIZE[2], GRID_SIZE[1], GRID_SIZE[0]
    
    gt_c_vol = gt_c.cpu().numpy().reshape(depth, height, width)
    pred_c_vol = pred_c.cpu().numpy().reshape(depth, height, width)
    
    gt_w_vol = gt_w.cpu().numpy().reshape(depth, height, width, 3)
    pred_w_vol = pred_w.cpu().numpy().reshape(depth, height, width, 3)
    
    return gt_c_vol, pred_c_vol, gt_w_vol, pred_w_vol

def calculate_metrics(gt_c, pred_c, gt_w, pred_w):
    """정량적 지표 계산 및 출력"""
    print("\n" + "="*50)
    print(f"📊 Quantitative Metrics Report (Sample {SAMPLE_INDEX})")
    print("="*50)
    
    # --- 1. 농도 (Concentration) 평가 ---
    mse_c = np.mean((gt_c - pred_c)**2)
    rmse_c = np.sqrt(mse_c)
    mae_c = np.mean(np.abs(gt_c - pred_c))
    max_error = np.abs(gt_c.max() - pred_c.max())
    
    # R2 Score (Coefficient of Determination)
    ss_res = np.sum((gt_c - pred_c)**2)
    ss_tot = np.sum((gt_c - np.mean(gt_c))**2)
    r2_c = 1 - (ss_res / (ss_tot + 1e-8))

    print(f"[🏭 Concentration Metrics]")
    print(f"  > RMSE       : {rmse_c:.4f} ppm")
    print(f"  > MAE        : {mae_c:.4f} ppm")
    print(f"  > Max Diff   : {max_error:.4f} ppm (GT Max: {gt_c.max():.2f} vs Pred Max: {pred_c.max():.2f})")
    print(f"  > R2 Score   : {r2_c:.4f} (1.0 is Best)")

    # --- 2. 바람 (Wind) 평가 ---
    # 벡터 크기(Speed) 계산
    gt_speed = np.linalg.norm(gt_w, axis=-1)
    pred_speed = np.linalg.norm(pred_w, axis=-1)
    
    rmse_w = np.sqrt(np.mean((gt_speed - pred_speed)**2))
    mae_w = np.mean(np.abs(gt_speed - pred_speed))
    
    # 벡터 방향(Cosine Similarity) 계산
    # 0으로 나누기 방지
    gt_norm = gt_speed + 1e-8
    pred_norm = pred_speed + 1e-8
    
    dot_product = np.sum(gt_w * pred_w, axis=-1)
    cosine_sim = dot_product / (gt_norm * pred_norm)
    avg_cosine = np.mean(cosine_sim)

    print(f"\n[🌪️ Wind Metrics]")
    print(f"  > Speed RMSE : {rmse_w:.4f} m/s")
    print(f"  > Speed MAE  : {mae_w:.4f} m/s")
    print(f"  > Direction  : {avg_cosine:.4f} (Cosine Similarity, Max 1.0)")
    print("="*50 + "\n")

def visualize_3d(gt_c, pred_c, gt_w, pred_w):
    """Plotly 시각화"""
    # 좌표 그리드
    Z, Y, X = np.mgrid[0:GRID_SIZE[2], 0:GRID_SIZE[1], 0:GRID_SIZE[0]]
    X = X * REAL_SCALE[0]
    Y = Y * REAL_SCALE[1]
    Z = Z * REAL_SCALE[2]
    
    # 1. 농도 Plot
    fig_conc = make_subplots(
        rows=1, cols=2, 
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Ground Truth (Conc)', 'Prediction (Conc)')
    )

    isomin_val = 1.0
    isomax_val = max(gt_c.max(), pred_c.max()) * 0.8

    fig_conc.add_trace(go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=gt_c.flatten(),
        isomin=isomin_val, isomax=isomax_val,
        opacity=0.1, surface_count=20, colorscale='Jet',
        colorbar=dict(title='GT (ppm)', x=0.45)
    ), row=1, col=1)

    fig_conc.add_trace(go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=pred_c.flatten(),
        isomin=isomin_val, isomax=isomax_val,
        opacity=0.1, surface_count=20, colorscale='Jet',
        colorbar=dict(title='Pred (ppm)')
    ), row=1, col=2)

    fig_conc.update_layout(title="3D Concentration Distribution", height=600)
    fig_conc.write_html("viz_3d_conc_%d.html" % SAMPLE_INDEX)
    print("✅ 3D Concentration Plot Saved: viz_3d_conc_%d.html" % SAMPLE_INDEX)

    # 2. 바람 Plot
    fig_wind = make_subplots(
        rows=1, cols=2, 
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Ground Truth (Wind)', 'Prediction (Wind)')
    )
    
    stride = 3 # 듬성듬성 그리기
    
    fig_wind.add_trace(go.Cone(
        x=X.flatten()[::stride**3], y=Y.flatten()[::stride**3], z=Z.flatten()[::stride**3],
        u=gt_w[..., 0].flatten()[::stride**3],
        v=gt_w[..., 1].flatten()[::stride**3],
        w=gt_w[..., 2].flatten()[::stride**3],
        colorscale='Viridis', sizemode="absolute", sizeref=2,
        colorbar=dict(title='Speed (m/s)', x=0.45)
    ), row=1, col=1)

    fig_wind.add_trace(go.Cone(
        x=X.flatten()[::stride**3], y=Y.flatten()[::stride**3], z=Z.flatten()[::stride**3],
        u=pred_w[..., 0].flatten()[::stride**3],
        v=pred_w[..., 1].flatten()[::stride**3],
        w=pred_w[..., 2].flatten()[::stride**3],
        colorscale='Viridis', sizemode="absolute", sizeref=2,
        colorbar=dict(title='Speed (m/s)')
    ), row=1, col=2)

    fig_wind.update_layout(title="3D Wind Field", height=600)
    fig_wind.write_html("viz_3d_wind_%d.html" % SAMPLE_INDEX)
    print("✅ 3D Wind Field Plot Saved: viz_3d_wind_%d.html" % SAMPLE_INDEX)

if __name__ == "__main__":
    print("=== 3D Evaluation & Visualization Start ===")
    model, ds = load_model_and_data()
    
    # 3D 볼륨 예측
    gt_c, pred_c, gt_w, pred_w = predict_3d_volume(model, ds, SAMPLE_INDEX)
    
    # 1. 정량적 평가 (콘솔 출력)
    calculate_metrics(gt_c, pred_c, gt_w, pred_w)
    
    # 2. 시각화 (HTML 저장)
    visualize_3d(gt_c, pred_c, gt_w, pred_w)