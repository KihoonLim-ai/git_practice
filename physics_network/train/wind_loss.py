#wind_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedGridLoss(nn.Module):
    def __init__(self, lambda_mse=1.0, lambda_pcc=0.0, lambda_phys=1.0):
        super(PhysicsInformedGridLoss, self).__init__()
        self.lambda_mse = lambda_mse
        self.lambda_pcc = lambda_pcc
        self.lambda_phys = lambda_phys
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def compute_terrain_slope(self, terrain_mask):
        """
        지형 마스크(Binary Voxel)로부터 기울기(Slope) 계산
        Input: terrain_mask (B, D, H, W) - 0:Air, 1:Ground
        """
        # 1. 지형 높이맵(2D) 추출 (Sum along Z-axis)
        # 0과 1로 된 마스크를 더하면 높이(Grid Unit)가 됨
        h_map = torch.sum(terrain_mask, dim=1)  # (B, H, W)
        
        # 2. 기울기 계산 (Central Difference)
        # dh/dx
        dh_dx = torch.zeros_like(h_map)
        dh_dx[:, :, 1:-1] = (h_map[:, :, 2:] - h_map[:, :, :-2]) / 2.0
        
        # dh/dy
        dh_dy = torch.zeros_like(h_map)
        dh_dy[:, 1:-1, :] = (h_map[:, 2:, :] - h_map[:, :-2, :]) / 2.0
        
        return dh_dx, dh_dy

    def forward(self, pred_c, target_c, pred_w, coords, target_w=None, inp_vol=None):
        """
        inp_vol: 지형 정보를 알기 위해 추가 (B, C, D, H, W)
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Shape Parsing
        B = pred_w.shape[0]
        # Heuristic Shape Recovery (Config 의존 없이 추론)
        # pred_w가 (B, N, 3)일 때 N = D*H*W
        # inp_vol이 들어오면 정확한 Shape 사용 가능
        if inp_vol is not None:
            _, _, D, H, W = inp_vol.shape
        else:
            # Fallback (고정값, 위험할 수 있음)
            D, H, W = 21, 45, 45
        
        # ------------------------------------------------
        # 1. MSE Loss
        # ------------------------------------------------
        loss_mse = 0.0
        if pred_c is not None and target_c is not None:
            loss_mse += F.mse_loss(pred_c, target_c)
            
        if target_w is not None:
            # U, V는 데이터대로 학습 (강하게)
            loss_mse += F.mse_loss(pred_w[..., :2], target_w[..., :2])
            # [중요] W는 MSE에서 제외하거나 매우 약하게! (데이터가 W=0이므로 배우면 안됨)
            # 여기서는 아예 뺍니다. W는 오직 Physics로만 만듭니다.
            # loss_mse += 0.0 * F.mse_loss(pred_w[..., 2], target_w[..., 2])

        if self.lambda_mse > 0:
            total_loss += self.lambda_mse * loss_mse
            loss_dict['mse'] = loss_mse.item()

        # ------------------------------------------------
        # 2. Physics Loss (Continuity + Boundary)
        # ------------------------------------------------
        if self.lambda_phys > 0 and inp_vol is not None:
            w_vol = pred_w.view(B, D, H, W, 3)
            u = w_vol[..., 0]
            v = w_vol[..., 1]
            w = w_vol[..., 2]
            
            # (A) Continuity (질량 보존)
            du_dx = (u[:, :, :, 2:] - u[:, :, :, :-2])
            dv_dy = (v[:, :, 2:, :] - v[:, :, :-2, :])
            dw_dz = (w[:, 2:, :, :] - w[:, :-2, :, :])
            div = du_dx[:, 1:-1, 1:-1, :] + dv_dy[:, 1:-1, :, 1:-1] + dw_dz[:, :, 1:-1, 1:-1]
            loss_continuity = torch.mean(div ** 2)
            
            # (B) Terrain Boundary Condition (지형 추종) - 핵심 추가! 🔥
            # inp_vol[:, 0] is Terrain Mask
            dh_dx, dh_dy = self.compute_terrain_slope(inp_vol[:, 0, ...])
            
            # 2D Slope를 3D로 확장 (Broadcasting)
            dh_dx_3d = dh_dx.unsqueeze(1).expand(-1, D, -1, -1)
            dh_dy_3d = dh_dy.unsqueeze(1).expand(-1, D, -1, -1)
            
            # Ideal W (물리적 목표값)
            # W_ideal ~ U * Slope_X + V * Slope_Y
            # 스케일링 이슈를 피하기 위해, 방향성(Correlation)만 맞춥니다.
            w_induced = u * dh_dx_3d + v * dh_dy_3d
            
            # 지형 표면 근처(Terrain Mask가 1인 곳의 바로 위)에 가중치를 줘야 하지만,
            # 간단하게 전체 영역에서 W가 유도된 W와 비슷해지도록 유도 (MSE)
            # 단, 지형 내부(Mask=1)나 너무 높은 곳은 제외하면 좋음. 
            # 여기서는 간단히 전체 트렌드를 맞춤.
            
            # W가 유도된 방향과 반대면 벌점 (즉, 오르막인데 W가 음수면 큰 벌점)
            # Loss = Error between Predicted W and Induced W
            # 단순 MSE보다는 부호가 다를 때 페널티를 주는 것이 좋음.
            
            # Scaling Factor (Grid unit vs m/s 보정) - 대략 0.5~1.0 사이
            loss_boundary = F.mse_loss(w, w_induced * 0.5) 

            loss_phys = loss_continuity + loss_boundary * 5.0 # Boundary 강제력 5배
            
            total_loss += self.lambda_phys * loss_phys
            loss_dict['phys'] = loss_phys.item()
        else:
            loss_dict['phys'] = 0.0
            
        return total_loss, loss_dict