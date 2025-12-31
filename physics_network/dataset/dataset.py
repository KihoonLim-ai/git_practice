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
        self.met = met_data           # (Time, 43) -> Full Profile
        self.conc = conc_data         # (Time, 45, 45, 21) -> (T, Y, X, Z)
        
        # 2D Maps (Base Info)
        self.terrain_2d = terrain     # (45, 45) Normalized Height
        self.source_q_2d = source_q   # (45, 45) Log Scaled Q
        self.source_h_2d = source_h   # (45, 45) Normalized H
        
        # Stats
        self.scale_wind = float(stats['scale_wind'])
        self.scale_terr = float(stats['terrain_max'])
        
        self.seq_len = seq_len
        self.pred_step = pred_step
        
        # -------------------------------------------------------------------
        # [Pre-calculation] 정적인 3D Map 미리 생성 (속도 최적화)
        # -------------------------------------------------------------------
        self.nz, self.ny, self.nx = Config.NZ, Config.NY, Config.NX # 21, 45, 45
        
        # 1. Terrain 3D Mask (지형 내부는 1, 공기는 0)
        z_vals = np.linspace(0, 1.0, self.nz).reshape(self.nz, 1, 1) # (21, 1, 1)
        self.terrain_3d = (z_vals <= self.terrain_2d[np.newaxis, :, :]).astype(np.float32)
        
        # 2. Source 3D Map (오염원 위치에 Q값 할당)
        self.source_3d = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
        rows, cols = np.where(self.source_q_2d > 0)
        for r, c in zip(rows, cols):
            q_val = self.source_q_2d[r, c]
            h_norm = self.source_h_2d[r, c]
            z_idx = int(h_norm * (self.nz - 1))
            z_idx = np.clip(z_idx, 0, self.nz - 1)
            self.source_3d[z_idx, r, c] = q_val

        self.valid_indices = np.arange(len(self.met))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # ------------------------------------------------
        # 1. Met Data Parsing (1D Profile -> 3D Field)
        # ------------------------------------------------
        raw_met = self.met[idx] # (43,)
        
        # 앞 42개는 (u, v) * 21층
        winds_norm = raw_met[:-1].reshape(self.nz, 2) # (Z, 2)
        
        # 1D Profile을 3D로 확장 (Broadcasting)
        u_3d = np.tile(winds_norm[:, 0].reshape(self.nz, 1, 1), (1, self.ny, self.nx))
        v_3d = np.tile(winds_norm[:, 1].reshape(self.nz, 1, 1), (1, self.ny, self.nx))
        w_3d = np.zeros_like(u_3d) # 수직풍은 0으로 초기화
        
        # ------------------------------------------------
        # [NEW] 2. Met Sequence Extraction (For Transformer)
        # ------------------------------------------------
        # 43차원 데이터에서 대표값 (가장 낮은 고도의 u, v + 안정도 L) 추출
        # met 구조: [u0...u20, v0...v20, L]
        # u_surf = met[:, 0], v_surf = met[:, 21], L = met[:, 42]
        
        # 시퀀스 범위 계산
        start_idx = idx - self.seq_len + 1
        end_idx = idx + 1
        
        if start_idx >= 0:
            # 정상 범위
            raw_seq = self.met[start_idx:end_idx] # (Seq, 43)
        else:
            # 앞부분 패딩 (부족한 만큼 첫 번째 데이터를 반복)
            pad_len = abs(start_idx)
            raw_part = self.met[0:end_idx]
            pad_part = np.tile(self.met[0], (pad_len, 1))
            raw_seq = np.concatenate([pad_part, raw_part], axis=0) # (Seq, 43)

        # (Seq, 43) -> (Seq, 3) 차원 축소 [u_surf, v_surf, L]
        # 모델의 input_dim=3에 맞춤
        u_surf = raw_seq[:, 0:1]   # 0번 인덱스 (가장 낮은 u)
        v_surf = raw_seq[:, 21:22] # 21번 인덱스 (가장 낮은 v)
        L_val  = raw_seq[:, 42:43] # 42번 인덱스 (L)
        
        met_seq = np.concatenate([u_surf, v_surf, L_val], axis=1).astype(np.float32)

        # ------------------------------------------------
        # 3. Construct Input Tensor (X)
        # ------------------------------------------------
        # Shape: (4, D, H, W)
        input_list = [
            self.terrain_3d, 
            self.source_3d,  
            u_3d,            
            v_3d             
        ]
        input_tensor = np.stack(input_list, axis=0).astype(np.float32)

        # ------------------------------------------------
        # 4. Construct Target Tensor (Y)
        # ------------------------------------------------
        # Conc Data: (Time, Y, X, Z) -> (Z, Y, X)
        conc_vol = self.conc[idx].transpose(2, 0, 1) 
        
        target_list = [
            conc_vol, # Ch 0: Conc
            u_3d,     # Ch 1: U
            v_3d,     # Ch 2: V
            w_3d      # Ch 3: W
        ]
        target_tensor = np.stack(target_list, axis=0).astype(np.float32)

        return (
            torch.from_numpy(input_tensor), 
            torch.from_numpy(met_seq),      # [추가됨] (Seq, 3)
            torch.from_numpy(target_tensor)
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
    # seq_len 파라미터를 넘겨줍니다.
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