import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from dataset.dataset import AermodDataset
from model import RecurrentDeepONet
from dataset.config_param import ConfigParam as Config

# [설정]
BEST_RUN_ID = "kari_sweep_20251224_13" 
BEST_SEQ_LEN = 6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_TESTS = 20  # 요청하신 20회 수행

def main():
    ds = AermodDataset(mode='test', seq_len=BEST_SEQ_LEN)
    model = RecurrentDeepONet().to(DEVICE)
    ckpt_path = os.path.join("./checkpoints", f"best_model_{BEST_RUN_ID}.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    results = []
    print(f"\n=== Starting Batch Evaluation ({NUM_TESTS} Random Samples) ===")
    
    # 중복 없는 랜덤 인덱스 추출
    random_indices = np.random.choice(len(ds), NUM_TESTS, replace=False)

    for i, idx in enumerate(random_indices):
        data = ds[idx]
        ctx, met, coords = [d.unsqueeze(0).to(DEVICE) for d in data[:3]]

        with torch.no_grad():
            p_wind_norm, p_conc_norm = model(ctx, met, coords)
            
            # 역정규화
            gt_w = data[3].numpy() * ds.scale_wind
            pd_w = p_wind_norm.cpu().numpy().squeeze() * ds.scale_wind
            gt_c = ds.denormalize_conc(data[4].numpy())
            pd_c = ds.denormalize_conc(p_conc_norm.cpu().numpy().squeeze())

        # 성능 지표 계산
        r2_w = r2_score(gt_w.flatten(), pd_w.flatten())
        r2_c = r2_score(gt_c.flatten(), pd_c.flatten())
        rmse_c = np.sqrt(mean_squared_error(gt_c.flatten(), pd_c.flatten()))

        results.append({
            'Sample_Idx': idx,
            'Wind_R2': r2_w,
            'Conc_R2': r2_c,
            'Conc_RMSE': rmse_c,
            'Max_GT_Conc': gt_c.max(),
            'Max_PD_Conc': pd_c.max()
        })
        
        print(f"[{i+1}/{NUM_TESTS}] Sample {idx:4d} | Wind R2: {r2_w:.4f} | Conc R2: {r2_c:.4f} | RMSE: {rmse_c:.2f}")

    # --- 결과 요약 및 통계 ---
    df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print(f"📊 FINAL STATISTICAL SUMMARY (N={NUM_TESTS})")
    print("-" * 50)
    print(f"Average Wind R2     : {df['Wind_R2'].mean():.4f}")
    print(f"Average Conc R2     : {df['Conc_R2'].mean():.4f}")
    print(f"Average Conc RMSE   : {df['Conc_RMSE'].mean():.4f}")
    print("-" * 50)
    print(f"Best Conc R2 Sample : {df.loc[df['Conc_R2'].idxmax(), 'Sample_Idx']} ({df['Conc_R2'].max():.4f})")
    print(f"Worst Conc R2 Sample: {df.loc[df['Conc_R2'].idxmin(), 'Sample_Idx']} ({df['Conc_R2'].min():.4f})")
    print("="*50)

    # 결과를 CSV로 저장 (나중에 분석용)
    df.to_csv(f"batch_eval_results_{BEST_RUN_ID}.csv", index=False)
    print(f"\nDetailed results saved to: batch_eval_results_{BEST_RUN_ID}.csv")

if __name__ == "__main__":
    main()