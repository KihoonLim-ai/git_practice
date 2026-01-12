import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset.config_param import ConfigParam as Config

class AermodDataset(Dataset):
    def __init__(self, seq_len=30, future_step=0, mode='train', split_ratio=(0.8, 0.1, 0.1), crop_size=32):
        """
        [Final Integrated Dataset]
        
        Args:
            seq_len (int): LSTM/Transformer 입력을 위한 시계열 길이
            future_step (int): 예측할 미래 시점 (0=현재, 1=1시간 뒤 ...)
            mode (str): 'train', 'val', 'test'
            split_ratio (tuple): 데이터 분할 비율
            crop_size (int): 학습 시 Random Crop 할 윈도우 크기 (기본 32x32)
        """
        self.seq_len = seq_len
        self.future_step = future_step
        self.mode = mode.lower()
        self.crop_size = crop_size
        
        # Grid Dimension
        self.nz, self.ny, self.nx = Config.NZ, Config.NY, Config.NX
        
        # -------------------------------------------------------------------
        # 1. Raw Data Load
        # -------------------------------------------------------------------
        p_dir = Config.PROCESSED_DIR
        print(f"[{self.mode.upper()}] Loading data (Future Step: +{future_step}h)...")
        
        try:
            d_maps = np.load(os.path.join(p_dir, Config.SAVE_MAPS))
            d_met  = np.load(os.path.join(p_dir, Config.SAVE_MET))
            d_lbl  = np.load(os.path.join(p_dir, Config.SAVE_LBL))
        except FileNotFoundError:
            raise FileNotFoundError(f"Processed .npz files not found in {p_dir}")

        # 전체 데이터
        full_met   = d_met['met']     # (TotalTime, 43)
        full_conc  = d_lbl['conc']    # (TotalTime, 45, 45, 21)
        
        self.terrain = d_maps['terrain']   
        self.source_q = d_maps['source_q'] 
        self.source_h = d_maps['source_h'] 
        self.scale_terr = float(d_maps['terrain_max'])
        
        # -------------------------------------------------------------------
        # 2. Data Splitting (Train / Val / Test)
        # -------------------------------------------------------------------
        total_len = len(full_met)
        r_train, r_val, r_test = split_ratio
        
        idx_train_end = int(total_len * r_train)
        idx_val_end   = int(total_len * (r_train + r_val))
        
        # 미래 예측을 위해 마지막 데이터 인덱스 안전장치 확보
        # (Target Index가 전체 범위를 넘어가면 안 되므로)
        safe_margin = future_step 
        
        if self.mode == 'train':
            self.met_data  = full_met[:idx_train_end]
            self.conc_data = full_conc[:idx_train_end]
        elif self.mode == 'val':
            self.met_data  = full_met[idx_train_end:idx_val_end]
            self.conc_data = full_conc[idx_train_end:idx_val_end]
        elif self.mode == 'test':
            self.met_data  = full_met[idx_val_end:]
            self.conc_data = full_conc[idx_val_end:]
        
        # 실제 사용 가능한 길이 (Seq len 및 Future step 고려)
        self.data_len = len(self.met_data) - self.future_step
        
        if self.data_len <= 0:
            raise ValueError(f"Dataset length is too short for future_step={future_step}")

        print(f"   -> Mode: {self.mode.upper()} | Available Steps: {self.data_len}")

        # -------------------------------------------------------------------
        # 3. Physics Pre-calculation (In-memory Caching)
        # -------------------------------------------------------------------
        self._init_physics_cache()

    def _init_physics_cache(self):
        """메모리 내 3D 바람장 계산 및 캐싱"""
        print(f"   -> Pre-calculating 3D Physics (u,v,w)...")

        # (1) Wind Profile Reshape
        # 미래 예측 시나리오: "미래 기상(t+k)"을 알면 "미래 농도(t+k)"를 알 수 있다.
        # 따라서 Cache는 met_data 전체 길이에 대해 다 만들어둠.
        winds_only = self.met_data[:, :-1]
        winds_reshaped = winds_only.reshape(len(self.met_data), self.nz, 2)

        # (2) Global Wind (Top Layer)
        self.global_wind_cache = winds_reshaped[:, -1, :].astype(np.float32)

        # (3) 3D Expansion
        u_prof = winds_reshaped[:, :, 0]
        v_prof = winds_reshaped[:, :, 1]
        
        u_3d = np.tile(u_prof[:, :, np.newaxis, np.newaxis], (1, 1, self.ny, self.nx))
        v_3d = np.tile(v_prof[:, :, np.newaxis, np.newaxis], (1, 1, self.ny, self.nx))

        # (4) Vertical Wind (w) Vectorized Calc
        w_3d = self._calculate_synthetic_w(u_3d, v_3d)

        # (5) Static Maps 3D Expansion
        z_vals = np.linspace(0, 1.0, self.nz).reshape(self.nz, 1, 1)
        
        # Terrain Mask (1, Z, Y, X)
        self.terrain_3d = (z_vals <= self.terrain[np.newaxis, :, :]).astype(np.float32)
        self.terrain_3d = self.terrain_3d[np.newaxis, :, :, :] 

        # Source Map (1, Z, Y, X)
        src_temp = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
        rows, cols = np.where(self.source_q > 0)
        for r, c in zip(rows, cols):
            h_n = self.source_h[r, c]
            z_idx = int(h_n * (self.nz - 1))
            src_temp[np.clip(z_idx, 0, self.nz-1), r, c] = self.source_q[r, c]
        self.source_3d = src_temp[np.newaxis, :, :, :]

        # (6) Apply Terrain Mask to Wind
        air_mask = (1.0 - self.terrain_3d)
        self.wind_3d_cache = np.stack([u_3d, v_3d, w_3d], axis=1).astype(np.float32)
        self.wind_3d_cache *= air_mask
        
        print(f"   -> Cache Ready. RAM: {self.wind_3d_cache.nbytes / 1e9:.2f} GB")

    def _calculate_synthetic_w(self, u, v):
        real_terr = self.terrain * self.scale_terr
        dh_dy, dh_dx = np.gradient(real_terr, Config.DY, Config.DX)
        
        slope_x = dh_dx[np.newaxis, np.newaxis, :, :]
        slope_y = dh_dy[np.newaxis, np.newaxis, :, :]
        
        z_idx = np.arange(self.nz).reshape(1, self.nz, 1, 1)
        decay = np.exp(-(z_idx * (Config.MAX_Z/(self.nz-1))) / 200.0)
        
        return (u * slope_x + v * slope_y) * decay

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # ----------------------------------------------------------------
        # 1. Time Indexing (Future Prediction)
        # ----------------------------------------------------------------
        # 모델은 "Target 시점의 기상"을 보고 "Target 시점의 농도"를 맞춘다.
        target_idx = idx + self.future_step
        
        # ----------------------------------------------------------------
        # 2. Input & Target Construction
        # ----------------------------------------------------------------
        # (A) Wind Volume (Target Time)
        wind_vol = self.wind_3d_cache[target_idx] # (3, Z, Y, X)
        
        # (B) Input Volume 결합: [Terrain(1), Source(1), Wind(3)] -> (5, Z, Y, X)
        input_vol = np.concatenate([self.terrain_3d, self.source_3d, wind_vol], axis=0)
        
        # (C) Target Concentration (Target Time)
        conc_vol = self.conc_data[target_idx].transpose(2, 0, 1)[np.newaxis, ...] # (1, Z, Y, X)
        
        # (D) Global Wind (Target Time)
        g_wind = self.global_wind_cache[target_idx].copy() # (2,)

        # (E) Met Sequence (Optional: LSTM용) - Target Time 기준 과거 30개
        start, end = target_idx - self.seq_len + 1, target_idx + 1
        if start >= 0: raw_seq = self.met_data[start:end]
        else:
            pad = abs(start)
            raw_seq = np.concatenate([np.tile(self.met_data[0],(pad,1)), self.met_data[0:end]])
        
        # Feature: Surface U, V, 1/L (Index 0, 1, -1)
        met_seq = np.concatenate([raw_seq[:, 0:1], raw_seq[:, 1:2], raw_seq[:, -1:]], axis=1).astype(np.float32)

        # ----------------------------------------------------------------
        # 3. Robustness Augmentation (Train Only)
        #    - Random Rotation (지형 변화 대응)
        #    - Random Crop (오염원 개수/위치 변화 대응)
        # ----------------------------------------------------------------
        if self.mode == 'train':
            # --- [Step A] Random Rotation (0, 90, 180, 270) ---
            k = np.random.randint(0, 4)
            if k > 0:
                # 3D Volume 회전 (H, W축: axis 2, 3)
                input_vol = np.rot90(input_vol, k, axes=(2, 3)).copy()
                conc_vol  = np.rot90(conc_vol, k, axes=(2, 3)).copy()
                
                # 벡터 회전 (u, v 성분 보정)
                u_tmp, v_tmp = input_vol[2].copy(), input_vol[3].copy()
                
                if k == 1:   # 90 deg
                    input_vol[2], input_vol[3] = -v_tmp, u_tmp
                    g_wind[0], g_wind[1] = -g_wind[1], g_wind[0]
                elif k == 2: # 180 deg
                    input_vol[2], input_vol[3] = -u_tmp, -v_tmp
                    g_wind[0], g_wind[1] = -g_wind[0], -g_wind[1]
                elif k == 3: # 270 deg
                    input_vol[2], input_vol[3] = v_tmp, -u_tmp
                    g_wind[0], g_wind[1] = g_wind[1], -g_wind[0]

            # --- [Step B] Random Crop (Fixed Source Overfitting 방지) ---
            # 전체 45x45 -> 부분 crop_size (예: 32x32)
            if self.crop_size < self.nx:
                y_s = np.random.randint(0, self.ny - self.crop_size + 1)
                x_s = np.random.randint(0, self.nx - self.crop_size + 1)
                
                input_vol = input_vol[:, :, y_s:y_s+self.crop_size, x_s:x_s+self.crop_size]
                conc_vol  = conc_vol[:, :, y_s:y_s+self.crop_size, x_s:x_s+self.crop_size]
                
                # 주의: Crop 시 met_seq나 g_wind는 전역변수라 영향 없음 (Good)
        
        # ----------------------------------------------------------------
        # 4. Returns
        # ----------------------------------------------------------------
        return (
            torch.from_numpy(input_vol), # (5, Z, H', W')
            torch.from_numpy(met_seq),   # (Seq, 3)
            torch.from_numpy(conc_vol),  # (1, Z, H', W')
            torch.from_numpy(g_wind)     # (2,)
        )

# ==============================================================================
# Helper for easy usage
# ==============================================================================
def get_dataloaders(batch_size=32, seq_len=30, future_step=1, crop_size=32):
    """
    미래 예측용 DataLoader 생성 함수
    """
    # 1. Train (Augmentation ON: Rotation + Crop)
    train_ds = AermodDataset(seq_len, future_step, 'train', crop_size=crop_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    # 2. Val (No Augmentation, Full Size)
    # Val은 Crop 없이 전체 맵(45x45)을 평가
    val_ds = AermodDataset(seq_len, future_step, 'val')
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 3. Test
    test_ds = AermodDataset(seq_len, future_step, 'test')
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"\n[Loaders Ready] Future Step: +{future_step}h | Train Crop: {crop_size}x{crop_size}")
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test Code
    tr, val, te = get_dataloaders(batch_size=4)
    sample = next(iter(tr))
    print("\n[Sample Shapes]")
    print(f"Input Vol  : {sample[0].shape}")
    print(f"Met Seq    : {sample[1].shape}")
    print(f"Target Conc: {sample[2].shape}")
    print(f"Global Wind: {sample[3].shape}")