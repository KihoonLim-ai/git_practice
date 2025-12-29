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
        self.source_q = source_q      # (45, 45) -> log scaled
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

    def __getitem__(self, idx):
        curr_idx = self.valid_indices[idx]
        future_idx = curr_idx  # t=0 (현재 예측)

        # ------------------------------------------------
        # 1. Context Maps (이미 전처리된 상태)
        # ------------------------------------------------
        # 지형(0~1), 배출량(Log), 높이(0~1)
        ctx_map = np.stack([self.terrain, self.source_q, self.source_h], axis=0)

        # ------------------------------------------------
        # 2. Encoder Input (Past 30 hours)
        # ------------------------------------------------
        # met: [u, v, L, wd] -> 앞에서 3개(u,v,L)만 사용
        met_seq = self.met[curr_idx - self.seq_len + 1 : curr_idx + 1, :3].copy()
        
        # [데이터셋 내부 정규화]
        # process_met.py에서 저장한 최대값으로 나눔
        # met_seq[:, 0] /= self.scale_wind
        # met_seq[:, 1] /= self.scale_wind
        # met_seq[:, 2] /= self.scale_L

        # ------------------------------------------------
        # 3. Decoder Query (4D Coordinates)
        # ------------------------------------------------
        # t=0.0 채널 추가
        time_channel = torch.zeros((self.coords_3d.shape[0], 1), dtype=torch.float32)
        coords_4d = torch.cat([torch.tensor(self.coords_3d), time_channel], dim=1)

        # ------------------------------------------------
        # 4. Ground Truth (Physics & Label)
        # ------------------------------------------------
        # A. Wind GT (Physics) - Raw 값 사용
        target_met = self.met[future_idx] # (4,)
        
        # Power Law 계산 공식은 m/s 단위가 들어가야 정확한 프로파일을 만듭니다.
        u_ref = target_met[0] * self.scale_wind
        v_ref = target_met[1] * self.scale_wind
        L_val = target_met[2] * self.scale_L
        
        # 물리 계산을 위해 좌표를 미터 단위로 복원
        z_real_m = self.coords_3d[:, 2] * Config.MAX_Z
        
        wind_field = calc_wind_profile_power_law(
            uref=u_ref, vref=v_ref, L=L_val, 
            z_points=z_real_m, 
            slopes=(self.slope_flat[:, 0], self.slope_flat[:, 1])
        ) # (N, 3)

        wind_field[:, 0] /= self.scale_wind
        wind_field[:, 1] /= self.scale_wind
        # Z축 속도(w)는 보통 작으므로 scale_wind로 같이 나눠도 무방
        wind_field[:, 2] /= self.scale_wind
        
        # B. Conc GT (Label) - 이미 Log+Norm 되어 있음
        c_vol = self.conc[future_idx] # (NY, NX, NZ)
        c_flat = c_vol.flatten()      # (N,)

        return (
            torch.tensor(ctx_map, dtype=torch.float32),
            torch.tensor(met_seq, dtype=torch.float32),
            coords_4d,
            torch.tensor(wind_field, dtype=torch.float32),
            torch.tensor(c_flat, dtype=torch.float32).unsqueeze(-1)
        )

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
