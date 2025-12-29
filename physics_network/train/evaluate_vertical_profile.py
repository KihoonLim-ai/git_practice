import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from dataset.config_param import ConfigParam as Config
from dataset.dataset import get_time_split_datasets
from model import ST_TransformerDeepONet

# ==========================================
# 설정
# ==========================================
CHECKPOINT_PATH = "./train/checkpoints/model_fearless-sweep-1_best.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_vertical():
    print("=== 3D Vertical Profile Evaluation ===")
    
    # 1. 데이터 & 모델 로드
    _, val_ds, _ = get_time_split_datasets(seq_len=30, pred_step=5)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False)
    
    model = ST_TransformerDeepONet(latent_dim=128, dropout=0.0).to(DEVICE)
    
    # 1. 체크포인트 먼저 로드
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    else:
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        return

    # 2. 저장된 Config에서 latent_dim 꺼내기 (없으면 128 기본값)
    # WandB Config 객체일 수도 있고 딕셔너리일 수도 있으므로 안전하게 처리
    loaded_dim = 128 
    if 'config' in checkpoint:
        conf = checkpoint['config']
        # 딕셔너리 접근 또는 속성 접근 시도
        if isinstance(conf, dict):
            loaded_dim = conf.get('latent_dim', 128)
        else: # Namespace or WandB config object
            loaded_dim = getattr(conf, 'latent_dim', 128)
            
    print(f"ℹ️ Detected latent_dim from checkpoint: {loaded_dim}")

    # 3. 모델 초기화 (감지된 차원 사용)
    model = ST_TransformerDeepONet(latent_dim=loaded_dim, dropout=0.0).to(DEVICE)
    
    # 4. 가중치 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
            
        print(f"✅ Model loaded from {CHECKPOINT_PATH}")
    
    model.eval()
    
    # 2. 데이터 수집 (GT vs Pred)
    all_pred_c, all_gt_c = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference"):
            ctx, met, coords, _, gt_c = [b.to(DEVICE) for b in batch]
            _, pred_c = model(ctx, met, coords)
            
            all_pred_c.append(pred_c.cpu().numpy())
            all_gt_c.append(gt_c.cpu().numpy())
            
    # (Total_Samples * Points) -> (Total_Samples, NZ, NY, NX) 형태로 복원해야 함
    # 하지만 현재 coords가 flatten 되어 있으므로, 포인트 순서를 알고 있어야 함.
    # dataset.py의 coords 생성 순서는: z -> y -> x 순으로 flatten 됨 (meshgrid indexing='xy' 기준 확인 필요)
    # 가장 안전한 방법: 전체 포인트를 NZ로 나누어 계산
    
    preds = np.concatenate(all_pred_c).flatten() # (N_total * 45*45*21)
    gts = np.concatenate(all_gt_c).flatten()
    
    # 물리량 복원
    c_mean = val_ds.conc_mean
    c_std = val_ds.conc_std
    
    pred_phys = np.maximum(np.expm1(preds * c_std + c_mean), 0)
    gt_phys = np.maximum(np.expm1(gts * c_std + c_mean), 0)
    
    # 3. 고도별(Layer-wise) 성능 계산
    # 전체 데이터 개수
    total_points = len(pred_phys)
    points_per_sample = Config.NX * Config.NY * Config.NZ
    num_samples = total_points // points_per_sample
    
    # Reshape: (Sample, NY, NX, NZ)가 아니라 (Sample, Points) 상태임.
    # Dataset 생성 시 coords 순서: z가 가장 바깥인지 안쪽인지 중요.
    # 보통 np.meshgrid(x, y, z) flatten -> z값이 가장 느리게 변함 or 빠르게 변함.
    # dataset.py 확인 결과: stack([xx, yy, zz]) -> zz가 반복됨.
    # 따라서 reshape을 (N_samples, NY*NX*NZ)로 하고, 내부에서 z별로 인덱싱
    
    rmse_list = []
    r2_list = []
    heights = np.arange(0, Config.NZ * Config.DZ, Config.DZ)
    
    print("\n[Analyzing per Height Layer...]")
    
    # 한 샘플 내에서 Z축의 인덱스 간격: 1개 레이어 당 NX*NY 개 포인트
    layer_size = Config.NX * Config.NY
    
    for z_idx in range(Config.NZ):
        # 전체 데이터에서 해당 z_layer에 해당하는 인덱스만 추출
        # 데이터 구조가 [Sample 1 (z0..z20), Sample 2 (z0..z20)] 라고 가정
        # 그러면 stride를 타면서 가져와야 함.
        
        # 하지만 dataset.py의 coords는 (N, 3)으로 고정되어 있고 배치는 시간축임.
        # 각 배치 내에서 좌표 순서는 고정됨.
        # 좌표 순서: xx, yy, zz meshgrid. 
        # zz 값은 000..000 (Layer 0), 111...111 (Layer 1) 순서로 되어 있을 가능성이 높음.
        # (dataset.py의 meshgrid indexing='xy' 확인 필요 -> 보통 z가 마지막 차원이면 덩어리로 묶임)
        
        # 간단한 슬라이싱을 위해 reshaping
        # 전체 데이터를 (Total_Samples, NZ, NY*NX)로 간주해보고 검증
        # 가장 확실한 건 zz 좌표값을 보고 마스킹하는 것
        
        # 여기서는 좌표 생성 로직상 z가 가장 외부 루프(또는 내부)인지에 따라 다르나,
        # 편의상 전체 배열에서 해당 레이어만 마스킹하는 방식 사용
        pass 

    # [수정] 위 방식은 복잡하니, 마스킹 방식으로 정확히 계산
    # Validation Set의 coords 중 Z값(3번째 컬럼)을 참조하면 됨.
    # 하지만 메모리 문제로 coords를 다 로드하기 힘드니, 
    # "1개 샘플의 Z 인덱스 패턴"을 파악해서 전체에 적용.
    
    # 1개 샘플의 Z 좌표 가져오기
    dummy_ds = val_ds
    _, _, sample_coords, _, _ = dummy_ds[0] # (N_points, 4)
    z_coords = sample_coords[:, 2].numpy() # 0~1 normalized
    
    # Z값의 유니크 값들 찾기 (층별 구분)
    unique_z = np.unique(z_coords)
    unique_z.sort()
    
    metric_data = []
    
    for u_z in unique_z:
        # 해당 고도에 해당하는 인덱스 마스크 (한 샘플 내에서)
        mask_layer = (z_coords == u_z)
        
        # 전체 데이터(모든 샘플)에 대해 이 마스크를 반복 적용
        # pred_phys는 (Samples * Points) 1D array.
        # reshape -> (Samples, Points)
        pred_mat = pred_phys.reshape(num_samples, points_per_sample)
        gt_mat = gt_phys.reshape(num_samples, points_per_sample)
        
        # 해당 레이어만 추출 (Samples, Layer_Points)
        pred_layer = pred_mat[:, mask_layer].flatten()
        gt_layer = gt_mat[:, mask_layer].flatten()
        
        rmse = np.sqrt(mean_squared_error(gt_layer, pred_layer))
        r2 = r2_score(gt_layer, pred_layer)
        
        real_h = u_z * Config.MAX_Z
        metric_data.append((real_h, rmse, r2))
        print(f"   > Height {real_h:5.1f}m : RMSE={rmse:.4f}, R2={r2:.4f}")

    # 4. 그래프 그리기 (Vertical Profile)
    h_vals = [m[0] for m in metric_data]
    rmse_vals = [m[1] for m in metric_data]
    r2_vals = [m[2] for m in metric_data]
    
    fig, ax1 = plt.subplots(figsize=(6, 8))
    
    ax1.set_xlabel('RMSE (ppm)', color='red')
    ax1.plot(rmse_vals, h_vals, 'r-o', label='RMSE')
    ax1.tick_params(axis='x', labelcolor='red')
    ax1.set_ylabel('Height (m)')
    ax1.set_title(f'Vertical Error Profile (R2 avg: {np.mean(r2_vals):.2f})')
    ax1.grid(True)
    
    ax2 = ax1.twiny()
    ax2.set_xlabel('R2 Score', color='blue')
    ax2.plot(r2_vals, h_vals, 'b--s', label='R2 Score')
    ax2.tick_params(axis='x', labelcolor='blue')
    ax2.set_xlim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig("eval_vertical_profile.png")
    print("\n✅ Saved vertical profile to 'eval_vertical_profile.png'")

if __name__ == "__main__":
    evaluate_vertical()