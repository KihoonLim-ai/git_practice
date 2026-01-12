import numpy as np
import torch
from dataset.config_param import ConfigParam as Config

def xy_to_grid(x, y):
    """ 
    물리 좌표(UTM) -> 45x45 그리드 인덱스 반환
    범위를 벗어나면 None 반환
    """
    if x is None or y is None: return None
    
    col = int((x - Config.X_ORIGIN) / Config.DX)
    row = int((y - Config.Y_ORIGIN) / Config.DY)
    
    if 0 <= col < Config.NX and 0 <= row < Config.NY:
        return row, col
    return None

def calc_wind_profile_power_law(uref, vref, L, z_points, z_ref=10.0, slopes=None):
    """
    [PINN/GT 생성용] Power Law 기반 바람장 (u, v, w) 계산
    
    Args:
        uref, vref: (float) 기준 높이에서의 풍속 성분
        L: (float) Monin-Obukhov Length
        z_points: (N,) 각 포인트의 실제 높이(m)
        z_ref: (float) 기준 높이 (보통 10m)
        slopes: (Tuple of arrays) (slope_x, slope_y) - 각 (N,) 형태
        
    Returns:
        wind_field: (N, 3) -> [u, v, w]
    """
    # 1. Power Law 지수(p) 결정
    # L 값에 따른 대기 안정도 판별
    if L > 0 and L < 200: 
        p = 0.55   # 매우 안정 (Very Stable)
    elif L >= 200 or L < 0: 
        p = 0.15   # 중립/불안정 (Neutral/Unstable)
    else: 
        p = 0.30   # 그 외 (Stable)
    
    # 2. 기준 풍속 크기 (Scalar)
    spd_ref = np.sqrt(uref**2 + vref**2)
    
    # 높이가 0이거나 음수인 경우 방지
    z_safe = np.maximum(z_points, 1e-6)
    
    # 3. 고도별 풍속 비율 계산 (Vector: N)
    # Power Law: U(z) = U_ref * (z / z_ref)^p
    ratio = (z_safe / z_ref)**p
    
    # 아주 낮은 고도(지면)는 0으로 처리
    ratio = np.where(z_points < 1e-3, 0.0, ratio)
    
    # 4. u, v 성분 계산 (Vector: N)
    u_z = uref * ratio
    v_z = vref * ratio
    
    # 5. W (수직풍) 계산 (Slope Flow)
    # 지형 경사(Slope)를 타고 오르내리는 바람 성분 추정
    if slopes is not None:
        slope_x, slope_y = slopes
        
        # 차원 안전 장치: (N, 1)이 들어오더라도 (N,)으로 맞춰줌
        if slope_x.ndim > 1: slope_x = slope_x.flatten()
        if slope_y.ndim > 1: slope_y = slope_y.flatten()
        
        # 지형 효과 감쇄 (높이 500m 이상이면 영향 거의 없음)
        decay = np.exp(-z_points / 500.0)
        
        # w = (u * slope_x + v * slope_y) * decay
        w_z = (u_z * slope_x + v_z * slope_y) * decay
    else:
        w_z = np.zeros_like(u_z)

    # 6. 최종 병합 (N, 3)
    # u_z, v_z, w_z는 모두 (N,) 형태여야 함
    return np.stack([u_z, v_z, w_z], axis=-1).astype(np.float32)

def make_batch_coords(batch_size, nz, ny, nx, device, time_val=1.0):
    """
    배치 크기와 그리드 크기에 맞는 정규화된 (x, y, z, t) 좌표 생성
    
    Args:
        batch_size (int): 배치 크기 (B)
        nz, ny, nx (int): 그리드 차원 (D, H, W) -> Train시는 32, Val시는 45 등 가변적
        device (torch.device): GPU/CPU
        time_val (float): 정규화된 시간 값 (기본 1.0 = 예측 시점)
        
    Returns:
        coords: (B, N, 4) 텐서. 여기서 N = nz * ny * nx
                순서는 (x, y, z, t)
    """
    # 1. 각 축에 대한 1D 좌표 생성 (0 ~ 1 사이로 정규화)
    # 모델은 "Crop된 윈도우" 안에서의 상대 위치를 학습합니다.
    x = torch.linspace(0, 1, nx, device=device)
    y = torch.linspace(0, 1, ny, device=device)
    z = torch.linspace(0, 1, nz, device=device)
    
    # 2. 3D Meshgrid 생성
    # indexing='ij' 옵션: (z, y, x) 순서대로 그물망을 짬
    # 결과 shape: (nz, ny, nx)
    gz, gy, gx = torch.meshgrid(z, y, x, indexing='ij')
    
    # 3. Flatten & Stack (N, 3)
    # Target 데이터가 (B, 1, Z, Y, X) 순서로 되어 있으므로
    # 좌표도 Z축부터 순회하도록 flatten 해야 매칭이 됩니다.
    # 하지만 좌표값 자체는 (x, y, z) 순서로 넣어주는 것이 일반적입니다.
    coords_3d = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    
    # 4. 시간(t) 차원 추가 (N, 4)
    # 모든 좌표점에 대해 동일한 time_val 부여
    num_points = coords_3d.shape[0]
    t_col = torch.full((num_points, 1), time_val, device=device)
    
    coords_4d = torch.cat([coords_3d, t_col], dim=-1)
    
    # 5. 배치 차원 확장 (B, N, 4)
    batch_coords = coords_4d.unsqueeze(0).expand(batch_size, -1, -1)
    
    return batch_coords