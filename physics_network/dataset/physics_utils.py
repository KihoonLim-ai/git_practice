# physics_utils.py
import numpy as np
from dataset.config_param import ConfigParam as Config

def xy_to_grid(x, y):
    """ 물리 좌표(UTM) -> 45x45 그리드 인덱스 """
    col = int((x - Config.X_ORIGIN) / Config.DX)
    row = int((y - Config.Y_ORIGIN) / Config.DY)
    if 0 <= col < Config.NX and 0 <= row < Config.NY:
        return row, col
    return None

def calc_wind_profile_power_law(uref, vref, L, z_points, z_ref=10.0, slopes=None):
    """
    [PINN용] Power Law 기반 바람장 계산 (W 포함)
    Args:
        slopes: (slope_x, slope_y) 튜플. (Optional)
    """
    # 1. Power Law 지수 결정
    if L > 0 and L < 200: p = 0.55   # 매우 안정
    elif L >= 200 or L < 0: p = 0.15 # 중립/불안정
    else: p = 0.30
    
    spd_ref = np.sqrt(uref**2 + vref**2)
    z_safe = np.maximum(z_points, 1e-6)
    
    # Power Law 적용
    spd_z = spd_ref * (z_safe / z_ref)**p
    spd_z = np.where(z_points < 1e-3, 0, spd_z)
    
    if spd_ref < 1e-6: ratio = 0
    else: ratio = spd_z / spd_ref
        
    u_z = uref * ratio
    v_z = vref * ratio
    
    # 2. W 계산 (Slope Flow)
    w_z = np.zeros_like(u_z)
    
    if slopes is not None:
        slope_x, slope_y = slopes
        # 지형 효과 감쇄 (높이 올라갈수록 영향 감소)
        decay = np.exp(-z_points / 500.0)
        
        # 만약 u_z는 (1,)이고 slope_x는 (45,45)라면 broadcasting 필요
        if u_z.ndim != slope_x.ndim:
             # 임시로 u_z의 차원을 확장하거나 np.multiply가 처리하게 둠
             pass
             
        w_z = (u_z * slope_x + v_z * slope_y) * decay

    # [핵심 수정] Broadcasting 처리
    # 입력 형상이 서로 다를 경우 (예: u_z는 스칼라, w_z는 맵) 안전하게 확장
    try:
        if u_z.shape != w_z.shape:
             u_z, v_z, w_z = np.broadcast_arrays(u_z, v_z, w_z)
    except ValueError:
        pass # 형상이 호환되지 않으면 에러가 날 수 있으니 주의

    return np.stack([u_z, v_z, w_z], axis=-1).astype(np.float32)
