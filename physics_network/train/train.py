# import os
# import sys
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import wandb

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

# from dataset.dataset import get_time_split_datasets
# from dataset.config_param import ConfigParam as Config
# from model.model import ST_TransformerDeepONet

# # ... (HybridPhysicsLoss 클래스와 get_grid_coords, seed_everything 함수는 기존과 동일) ...
# # (코드 길이상 생략하지만, 실제 파일에는 포함되어야 합니다)
# class HybridPhysicsLoss(nn.Module):
#     # ... (이전과 동일한 내용) ...
#     def __init__(self, w_conc=1.0, w_wind=1.0, lambda_phys=0.5, topk_ratio=0.05, 
#                  dx=100.0, dy=100.0, dz=10.0):
#         super().__init__()
#         self.w_conc = w_conc
#         self.w_wind = w_wind
#         self.lambda_phys = lambda_phys
#         self.topk_ratio = topk_ratio
#         self.dx = dx
#         self.dy = dy
#         self.dz = dz
#         self.mse = nn.MSELoss(reduction='none')

#     def calc_divergence(self, u, v, w):
#         du_dx = (u[:, :, :, 2:] - u[:, :, :, :-2]) / (2 * self.dx)
#         dv_dy = (v[:, :, 2:, :] - v[:, :, :-2, :]) / (2 * self.dy)
#         dw_dz = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * self.dz)
#         return du_dx[:, 1:-1, 1:-1, :] + dv_dy[:, 1:-1, :, 1:-1] + dw_dz[:, :, 1:-1, 1:-1]

#     def forward(self, pred_wind, target_wind, pred_conc, target_conc):
#         B = pred_wind.shape[0]
#         D, H, W = Config.NZ, Config.NY, Config.NX
        
#         u_pred = pred_wind[..., 0].view(B, D, H, W)
#         v_pred = pred_wind[..., 1].view(B, D, H, W)
#         w_pred = pred_wind[..., 2].view(B, D, H, W)
#         c_pred = pred_conc.view(B, D, H, W)

#         c_true = target_conc[:, 0]
#         u_true = target_conc[:, 1]
#         v_true = target_conc[:, 2]

#         pixel_loss_c = self.mse(c_pred, c_true)
#         pixel_loss_flat = pixel_loss_c.view(-1)
#         k = int(pixel_loss_flat.numel() * self.topk_ratio)
#         if k < 1: k = 1
#         loss_topk, _ = torch.topk(pixel_loss_flat, k)
#         loss_c = (0.7 * loss_topk.mean()) + (0.3 * pixel_loss_flat.mean())

#         loss_u = self.mse(u_pred, u_true).mean()
#         loss_v = self.mse(v_pred, v_true).mean()
#         vec_pred_uv = torch.stack([u_pred, v_pred], dim=-1)
#         vec_true_uv = torch.stack([u_true, v_true], dim=-1)
#         cos_sim = F.cosine_similarity(vec_pred_uv, vec_true_uv, dim=-1, eps=1e-8)
#         loss_dir = (1.0 - cos_sim).mean()
#         loss_wind_data = loss_u + loss_v + (0.5 * loss_dir)

#         div = self.calc_divergence(u_pred, v_pred, w_pred)
#         loss_phys = torch.mean(div ** 2)

#         total_loss = (self.w_conc * loss_c) + (self.w_wind * loss_wind_data) + (self.lambda_phys * loss_phys)
#         return total_loss, loss_c, loss_wind_data, loss_phys

# def get_grid_coords(device):
#     z = torch.linspace(0, 1, Config.NZ)
#     y = torch.linspace(0, 1, Config.NY)
#     x = torch.linspace(0, 1, Config.NX)
#     grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
#     coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(device)
#     t = torch.zeros((coords.shape[0], 1), device=device)
#     return torch.cat([coords, t], dim=-1)

# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)


# # --- Training Loop ---
# def train(config=None):
#     with wandb.init(config=config, project="kari_pinn_3d_sweep", entity="jhlee98") as run:
#         cfg = wandb.config
#         seed_everything(cfg.seed)
#         DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Data
#         train_ds, val_ds, _ = get_time_split_datasets(seq_len=30)
#         train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
#         val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
        
#         # Model
#         model = ST_TransformerDeepONet(
#             latent_dim=int(cfg.latent_dim), 
#             dropout=cfg.dropout,
#             fourier_scale=cfg.fourier_scale
#         ).to(DEVICE)
        
#         # Optimizer
#         optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        
#         # Loss
#         criterion = HybridPhysicsLoss(
#             w_conc=cfg.w_conc,
#             w_wind=cfg.w_wind,
#             lambda_phys=cfg.lambda_phys,
#             topk_ratio=cfg.topk_ratio
#         )
        
#         grid_coords = get_grid_coords(DEVICE)
        
#         # [NEW] Best Model Tracking
#         best_val_loss = float('inf')
        
#         print(f"=== Start Training (Target Phys: {cfg.lambda_phys}) ===")
        
#         for epoch in range(cfg.epochs):
#             # Curriculum Learning
#             if epoch < 20:
#                 criterion.lambda_phys = 0.0
            
#             elif epoch < 50:
#                 criterion.lambda_phys = cfg.lambda_phys* ((epoch - 20) / 30)
#             else:
#                 criterion.lambda_phys = cfg.lambda_phys

#             model.train()
#             loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
            
#             for batch in loop:
#                 inp_vol, met_seq, target_vol = [b.to(DEVICE) for b in batch]
                
#                 optimizer.zero_grad()
#                 batch_coords = grid_coords.unsqueeze(0).expand(inp_vol.shape[0], -1, -1)
#                 pred_w, pred_c = model(inp_vol, met_seq, batch_coords)
                
#                 loss, l_c, l_w, l_p = criterion(pred_w, None, pred_c, target_vol)
                
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
#                 optimizer.step()
                
#                 loop.set_postfix(loss=loss.item(), phys=l_p.item())
#                 wandb.log({
#                     "train_loss": loss.item(),
#                     "loss_conc": l_c.item(),
#                     "loss_wind": l_w.item(),
#                     "loss_phys": l_p.item(),
#                     "curr_lambda_phys": criterion.lambda_phys
#                 })
            
#             # Validation Step
#             model.eval()
#             val_loss = 0
#             with torch.no_grad():
#                 for batch in val_loader:
#                     inp_vol, met_seq, target_vol = [b.to(DEVICE) for b in batch]
#                     batch_coords = grid_coords.unsqueeze(0).expand(inp_vol.shape[0], -1, -1)
#                     pred_w, pred_c = model(inp_vol, met_seq, batch_coords)
#                     v_loss, _, _, _ = criterion(pred_w, None, pred_c, target_vol)
#                     val_loss += v_loss.item()
            
#             avg_val = val_loss / len(val_loader)
#             wandb.log({"epoch": epoch+1, "val_loss": avg_val})
            
#             # ==========================================================
#             # [NEW] Save Best Model (Config Included)
#             # ==========================================================
#             if avg_val < best_val_loss:
#                 best_val_loss = avg_val
#                 # WandB Run 이름을 사용하여 파일명 중복 방지 (예: best_model_lilac-sweep-1.pth)
#                 save_path = f"checkpoints/best_model_{run.name}.pth"
#                 os.makedirs("checkpoints", exist_ok=True)
                
#                 checkpoint = {
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'config': dict(cfg),  # 설정 정보 저장 (Inference 시 필요)
#                     'best_val_loss': best_val_loss
#                 }
                
#                 torch.save(checkpoint, save_path)
#                 print(f"  -> Best Model Saved! (Val Loss: {best_val_loss:.4f})")
                
#             # Periodic Save (Optional)
#             if epoch % 20 == 0 and epoch > 0:
#                 # 주기적 저장도 config 포함
#                 ckpt_periodic = {
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'config': dict(cfg)
#                 }
#                 torch.save(ckpt_periodic, f"checkpoints/model_{run.name}_ep{epoch}.pth")

# if __name__ == "__main__":
#     sweep_config = {
#         'method': 'bayes',
#         'metric': {'name': 'val_loss', 'goal': 'minimize'},
#         'parameters': {
#             'epochs': {'value': 200},
#             'batch_size': {'value': 32},
#             'lr': {'values': [1e-4, 5e-4]},
#             'latent_dim': {'values': [128, 256]},
#             'fourier_scale': {'values': [10.0, 20.0, 30.0]},
#             'dropout': {'values': [0.1, 0.2, 0.3]},
#             'seed': {'value': 42},
#             'w_conc': {'values': [1.0, 5.0]},
#             'w_wind': {'values': [1.0, 2.0]},
#             'lambda_phys': {'values': [0.1, 0.5]},
#             'topk_ratio': {'values': [0.05, 0.1]},
#             'grad_clip': {'values': [0.05, 0.1, 0.5, 1.0]},
#         }
#     }
    
#     sweep_id = wandb.sweep(sweep_config, project="kari_pinn_3d_sweep", entity="jhlee98")
#     wandb.agent(sweep_id, function=train, count=10)