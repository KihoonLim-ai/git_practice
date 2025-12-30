import os
import torch
import numpy as np
from torch.utils.data import Dataset
from dataset.config_param import ConfigParam as Config

# 사용자가 업로드한 physics_utils 활용
from dataset.physics_utils import calc_wind_profile_power_law

class AermodDataset(Dataset):
    def __init__(self, met_data, terrain, source_q, source_h, conc_data, 
                 coords_3d, slope_flat, stats, 
                 seq_len=30, pred_step=5):
        """
        메모리 효율성을 위해 파일 로드는 외부(get_time_split_datasets)에서 수행하고
        여기서는 잘라진(Sliced) 데이터 배열만 받습니다.
        """
        self.met = met_data           # (Time, 4) -> u, v, L, wd
        self.terrain = terrain        # (45, 45) -> 0~1 normalized
        self.source_q = source_q      # (45, 45) -> log scaled (Gaussian Splatted)
        self.source_h = source_h      # (45, 45) -> 0~1 normalized
        self.conc = conc_data         # (Time, NY, NX, NZ) -> Log+Normed
        
        self.coords_3d = coords_3d    # (N, 3)
        self.slope_flat = slope_flat  # (N, 2)
        
        # 정규화 및 복원을 위한 통계치
        self.scale_wind = float(stats['scale_wind']) # max_uv
        self.scale_L = float(stats['scale_L'])       # max_L
        self.conc_mean = float(stats['conc_mean'])   # 복원용
        self.conc_std = float(stats['conc_std'])     # 복원용
        
        self.seq_len = seq_len
        self.pred_step = pred_step
        
        # 유효 인덱스 계산
        total_len = len(self.met)
        self.valid_indices = np.arange(seq_len, total_len - pred_step)

    def __len__(self):
        return len(self.valid_indices)
    
    def _shift_source_map(self, src_map, u_norm, v_norm):
        """
        [New Feature] 바람에 의한 오염원 이동 (Advected Source)
        - u, v는 정규화된 값이므로 scale_wind를 곱해 실제 m/s로 복원
        - 실제 풍속에 비례하여 맵을 shift
        """
        # 실제 풍속 복원 (m/s)
        u_real = u_norm * self.scale_wind
        v_real = v_norm * self.scale_wind
        
        # 격자 크기(100m) 단위로 얼마나 밀릴지 계산
        # 예: 풍속 5m/s * 1시간(3600s) = 18km -> 너무 멂
        # 예: 풍속 5m/s -> "현재 순간의 확산 경향"만 보여주기 위해 작게 scaling
        # 여기서는 단순히 방향성 힌트만 주기 위해 1~3칸 정도만 밀리게 설정
        # scale factor 0.5: 풍속 10m/s일 때 5칸 이동
        scale_factor = 0.5 
        
        shift_x = int(u_real * scale_factor)
        shift_y = int(v_real * scale_factor)
        
        # 맵 이동 (Rolling)
        # axis=0: y축(row), axis=1: x축(col)
        shifted = np.roll(src_map, shift_y, axis=0)
        shifted = np.roll(shifted, shift_x, axis=1)
        
        # Roll은 반대편에서 튀어나오므로, 마스킹 처리 (간단하게 구현)
        # (완벽하진 않지만 CNN이 경계선 노이즈는 무시하도록 학습됨)
        return shifted

    def __getitem__(self, idx):
        curr_idx = self.valid_indices[idx]
        future_idx = curr_idx  # t=0 (현재 예측)

        # ------------------------------------------------
        # 1. Met Data Handling (수정됨)
        # ------------------------------------------------
        # met: [u, v, L, wd]
        # target_met는 현재 시점(future_idx)의 기상 데이터
        target_met_all = self.met[future_idx] 
        
        # [중요] 4번째 컬럼(wd)은 버리고 앞의 3개(u, v, L)만 취함
        u_curr = target_met_all[0]
        v_curr = target_met_all[1]
        L_curr = target_met_all[2]
        
        # Past Sequence (Encoder Input)
        # 여기서도 :3 슬라이싱을 명확히 하여 wd가 들어가는 것 방지
        met_seq = self.met[curr_idx - self.seq_len + 1 : curr_idx + 1, :3].copy()

        # ------------------------------------------------
        # 2. Context Maps (Advected Source 추가)
        # ------------------------------------------------
        # (A) 기본 맵: 지형, 오염원(Q), 높이(H)
        # (B) 추가 맵: 바람에 밀린 오염원 (Direction Hint)
        adv_source = self._shift_source_map(self.source_q, u_curr, v_curr)
        
        # 채널 4개로 확장: [Terrain, Source_Q, Source_H, Adv_Source]
        # Model의 ConvBranch 입력 채널(in_channels)을 3 -> 4로 수정해야 함!
        ctx_map = np.stack([self.terrain, self.source_q, self.source_h, adv_source], axis=0)

        # ------------------------------------------------
        # 3. Decoder Query (4D Coordinates)
        # ------------------------------------------------
        # t=0.0 채널 추가
        time_channel = torch.zeros((self.coords_3d.shape[0], 1), dtype=torch.float32)
        coords_4d = torch.cat([torch.tensor(self.coords_3d), time_channel], dim=1)

        # ------------------------------------------------
        # 4. Ground Truth (Physics & Label)
        # ------------------------------------------------
        # A. Wind GT (Physics)
        # Power Law 계산 공식은 m/s 단위가 들어가야 정확함
        u_ref = u_curr * self.scale_wind
        v_ref = v_curr * self.scale_wind
        L_val = L_curr * self.scale_L
        
        z_real_m = self.coords_3d[:, 2] * Config.MAX_Z
        
        wind_field = calc_wind_profile_power_law(
            uref=u_ref, vref=v_ref, L=L_val, 
            z_points=z_real_m, 
            slopes=(self.slope_flat[:, 0], self.slope_flat[:, 1])
        ) # (N, 3)

        # 다시 정규화하여 모델 출력과 스케일 맞춤
        wind_field[:, 0] /= self.scale_wind
        wind_field[:, 1] /= self.scale_wind
        wind_field[:, 2] /= self.scale_wind
        
        # B. Conc GT (Label)
        c_vol = self.conc[future_idx] # (NY, NX, NZ)
        c_flat = c_vol.flatten()      # (N,)

        return (
            torch.tensor(ctx_map, dtype=torch.float32),
            torch.tensor(met_seq, dtype=torch.float32),
            coords_4d,
            torch.tensor(wind_field, dtype=torch.float32),
            torch.tensor(c_flat, dtype=torch.float32).unsqueeze(-1)
        )

# get_time_split_datasets 함수는 기존과 동일하므로 생략하거나 그대로 둡니다.
# 다만 import 부분과 클래스 정의는 위 코드로 교체해주세요.
def get_time_split_datasets(seq_len=30, pred_step=5, val_ratio=0.1):
    """
    여기서 파일 로드를 수행하고, Train/Val로 쪼개서 Dataset 클래스에 넘겨줍니다.
    """
    p_dir = Config.PROCESSED_DIR
    
    # 1. 데이터 로드 (연구자님이 원하시던 안전한 방식)
    print(f"Loading data from {p_dir}...")
    
    # (1) Maps
    d_maps = np.load(os.path.join(p_dir, Config.SAVE_MAPS))
    terrain = d_maps['terrain']
    source_q = d_maps['source_q']
    source_h = d_maps['source_h']
    terrain_max = float(d_maps.get('terrain_max', 1.0))
    
    # (2) Meteorology
    d_met = np.load(os.path.join(p_dir, Config.SAVE_MET))
    met_data = d_met['met']     # (N, 4)
    max_uv = float(d_met['max_uv'])
    max_L = float(d_met['max_L'])
    
    # (3) Labels (Concentration)
    d_lbl = np.load(os.path.join(p_dir, Config.SAVE_LBL))
    conc_data = d_lbl['conc']   # (N, 45, 45, 21)
    # process_labels.py에서 저장한 통계값 로드
    conc_mean = float(d_lbl['mean_stat'])
    conc_std = float(d_lbl['std_stat'])

    # 2. 통계 딕셔너리 구성
    stats = {
        'scale_wind': max_uv,
        'scale_L': max_L,
        'conc_mean': conc_mean,
        'conc_std': conc_std
    }
    
    # 3. 좌표계 및 Slope 생성 (한 번만 계산)
    x_real = np.arange(0, Config.NX * Config.DX, Config.DX)
    y_real = np.arange(0, Config.NY * Config.DY, Config.DY)
    z_real = np.arange(0, Config.NZ * Config.DZ, Config.DZ)
    
    xx, yy, zz = np.meshgrid(
        x_real / Config.MAX_X, 
        y_real / Config.MAX_Y, 
        z_real / Config.MAX_Z, 
        indexing='xy'
    )
    coords_3d = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1).astype(np.float32)

    # Slope
    real_terrain = terrain * terrain_max
    grad_y, grad_x = np.gradient(real_terrain, Config.DY, Config.DX)
    gxx = np.repeat(grad_x[:, :, np.newaxis], Config.NZ, axis=2).flatten()
    gyy = np.repeat(grad_y[:, :, np.newaxis], Config.NZ, axis=2).flatten()
    slope_flat = np.stack([gxx, gyy], axis=1).astype(np.float32)

    # 4. 데이터 분할 (Time Split)
    total_len = len(met_data)
    n_val = int(total_len * val_ratio)
    n_test = int(total_len * 0.1) # Test 10% (옵션)
    n_train = total_len - n_val - n_test
    
    # Index Slicing
    train_end = seq_len + n_train
    val_end = train_end + n_val
    
    # 5. Dataset 인스턴스 생성 (Slicing해서 전달)
    # Train
    train_ds = AermodDataset(
        met_data[:train_end], terrain, source_q, source_h, conc_data[:train_end],
        coords_3d, slope_flat, stats, seq_len, pred_step
    )
    # Val
    # (주의: 시계열 연속성을 위해 앞부분 seq_len만큼 겹쳐서 가져감)
    val_ds = AermodDataset(
        met_data[train_end-seq_len : val_end], terrain, source_q, source_h, conc_data[train_end-seq_len : val_end],
        coords_3d, slope_flat, stats, seq_len, pred_step
    )
    
    print(f"Data Loaded: Train={len(train_ds)}, Val={len(val_ds)}")
    return train_ds, val_ds, None