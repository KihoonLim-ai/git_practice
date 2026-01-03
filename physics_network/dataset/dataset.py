import os
import torch
import numpy as np
from torch.utils.data import Dataset
from dataset.config_param import ConfigParam as Config

class AermodDataset(Dataset):
    def __init__(self, met_data, terrain, source_q, source_h, conc_data, stats, 
                 seq_len=30, pred_step=0):
        """
        3D CNN용 데이터셋 (Volumetric Data)
        Output Shape: (Channels, Depth(Z), Height(Y), Width(X))
        """
        self.met = met_data           # (Time, 43)
        self.conc = conc_data         # (Time, 45, 45, 21)
        
        # 2D Maps (Base Info)
        self.terrain_2d = terrain     # (45, 45) Normalized Height (0~1)
        self.source_q_2d = source_q   
        self.source_h_2d = source_h   
        
        # Stats (물리량 복원 및 계산을 위해 필요)
        self.scale_wind = float(stats['scale_wind'])
        self.scale_terr = float(stats['terrain_max']) # 지형 최대 높이 (m)
        
        self.seq_len = seq_len
        self.pred_step = pred_step
        
        self.nz, self.ny, self.nx = Config.NZ, Config.NY, Config.NX 
        
        # 1. Terrain 3D Mask
        z_vals = np.linspace(0, 1.0, self.nz).reshape(self.nz, 1, 1) 
        self.terrain_3d = (z_vals <= self.terrain_2d[np.newaxis, :, :]).astype(np.float32)
        
        # 2. Source 3D Map
        self.source_3d = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
        rows, cols = np.where(self.source_q_2d > 0)
        for r, c in zip(rows, cols):
            q_val = self.source_q_2d[r, c]
            h_norm = self.source_h_2d[r, c]
            z_idx = int(h_norm * (self.nz - 1))
            z_idx = np.clip(z_idx, 0, self.nz - 1)
            self.source_3d[z_idx, r, c] = q_val

        self.valid_indices = np.arange(len(self.met))

    def calculate_synthetic_w(self, u_3d, v_3d):
        """
        [핵심 연구 로직] 수평 바람과 지형 경사를 이용한 수직풍(w) 합성
        Formula: w = (u * slope_x + v * slope_y) * decay
        """
        real_terrain = self.terrain_2d * self.scale_terr
        dh_dy, dh_dx = np.gradient(real_terrain, Config.DY, Config.DX)
        
        slope_x = np.tile(dh_dx[np.newaxis, :, :], (self.nz, 1, 1))
        slope_y = np.tile(dh_dy[np.newaxis, :, :], (self.nz, 1, 1))
        
        z_indices = np.arange(self.nz).reshape(-1, 1, 1)
        z_meters = z_indices * (Config.MAX_Z / (self.nz - 1)) 
        
        decay = np.exp(-z_meters / 200.0) 
        
        w_synthetic = (u_3d * slope_x + v_3d * slope_y) * decay
        
        return w_synthetic.astype(np.float32)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 1. Met Data Parsing
        raw_met = self.met[idx] 
        # winds_norm shape: (NZ, 2) -> (u, v)
        winds_norm = raw_met[:-1].reshape(self.nz, 2)
        
        # ==================================================================
        # [NEW] Global Wind Condition 추출
        # ==================================================================
        # 3D로 확장하기 전, 원본 프로파일의 가장 꼭대기 층(Top Layer) 값을 추출합니다.
        # winds_norm은 0번이 바닥, -1번이 꼭대기입니다.
        u_top = winds_norm[-1, 0]
        v_top = winds_norm[-1, 1]
        
        # (2,) 형태의 텐서 생성
        global_wind = torch.tensor([u_top, v_top], dtype=torch.float32)
        # ==================================================================
        
        # 1D -> 3D Expansion
        u_3d = np.tile(winds_norm[:, 0].reshape(self.nz, 1, 1), (1, self.ny, self.nx))
        v_3d = np.tile(winds_norm[:, 1].reshape(self.nz, 1, 1), (1, self.ny, self.nx))
        
        # 합성 w 계산
        w_3d = self.calculate_synthetic_w(u_3d, v_3d) 
        
        air_mask = 1.0 - self.terrain_3d

        # u, v, w에 마스크 곱하기
        u_3d = u_3d * air_mask
        v_3d = v_3d * air_mask
        w_3d = w_3d * air_mask
        
        # 2. Met Sequence Extraction
        start_idx = idx - self.seq_len + 1
        end_idx = idx + 1
        
        if start_idx >= 0:
            raw_seq = self.met[start_idx:end_idx]
        else:
            pad_len = abs(start_idx)
            raw_part = self.met[0:end_idx]
            pad_part = np.tile(self.met[0], (pad_len, 1))
            raw_seq = np.concatenate([pad_part, raw_part], axis=0)

        u_surf = raw_seq[:, 0:1]   
        v_surf = raw_seq[:, 21:22] 
        L_val  = raw_seq[:, 42:43] 
        met_seq = np.concatenate([u_surf, v_surf, L_val], axis=1).astype(np.float32)

        # 3. Input Tensor
        input_list = [self.terrain_3d, self.source_3d, u_3d, v_3d]
        input_tensor = np.stack(input_list, axis=0).astype(np.float32)

        # 4. Target Tensor
        conc_vol = self.conc[idx].transpose(2, 0, 1) 
        
        target_list = [
            conc_vol, # Ch 0: Concentration
            u_3d,     # Ch 1: U Wind
            v_3d,     # Ch 2: V Wind
            w_3d      # Ch 3: W Wind
        ]
        target_tensor = np.stack(target_list, axis=0).astype(np.float32)

        # [수정] 반환값에 global_wind 추가 (총 4개 반환)
        return (
            torch.from_numpy(input_tensor), 
            torch.from_numpy(met_seq),
            torch.from_numpy(target_tensor),
            global_wind
        )

def get_time_split_datasets(seq_len=30, pred_step=5, val_ratio=0.1):
    p_dir = Config.PROCESSED_DIR
    print(f"Loading data from {p_dir}...")
    
    # 1. Load All
    d_maps = np.load(os.path.join(p_dir, Config.SAVE_MAPS))
    terrain = d_maps['terrain']
    source_q = d_maps['source_q']
    source_h = d_maps['source_h']
    terr_max = float(d_maps.get('terrain_max', 1.0))
    
    d_met = np.load(os.path.join(p_dir, Config.SAVE_MET))
    met_data = d_met['met'] 
    max_uv = float(d_met['max_uv'])
    
    d_lbl = np.load(os.path.join(p_dir, Config.SAVE_LBL))
    conc_data = d_lbl['conc'] 
    
    stats = {
        'scale_wind': max_uv,
        'terrain_max': terr_max
    }
    
    # 2. Split
    total_len = len(met_data)
    n_val = int(total_len * val_ratio)
    n_train = total_len - n_val
    
    # 3. Create Datasets
    train_ds = AermodDataset(
        met_data[:n_train], terrain, source_q, source_h, conc_data[:n_train], 
        stats, seq_len=seq_len
    )
    val_ds = AermodDataset(
        met_data[n_train:], terrain, source_q, source_h, conc_data[n_train:], 
        stats, seq_len=seq_len
    )
    
    print(f"Dataset Created: Train={len(train_ds)}, Val={len(val_ds)}")
    print(f"Input Shape: (4, {Config.NZ}, {Config.NY}, {Config.NX})")
    
    return train_ds, val_ds, stats