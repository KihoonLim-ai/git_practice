import os
import torch
import numpy as np
from torch.utils.data import Dataset
from dataset.config_param import ConfigParam as Config
from dataset.physics_utils import calc_wind_profile_power_law

class AermodDataset(Dataset):
    def __init__(self, mode='train', seq_len=30):
        self.seq_len = seq_len
        p_dir = Config.PROCESSED_DIR
        print(f"Loading {mode} dataset (Dynamic Max-Abs Scaling)...")
        
        # 1. 데이터 로드
        d_maps = np.load(os.path.join(p_dir, Config.SAVE_MAPS))
        self.terrain = d_maps['terrain']
        self.source_q = d_maps['source_q']
        self.source_h = d_maps['source_h']
        self.max_h = float(d_maps.get('terrain_max', 1.0))
        
        d_met = np.load(os.path.join(p_dir, Config.SAVE_MET))
        self.met = d_met['met'] # (N, 4) -> u, v, L, wd
        
        d_lbl = np.load(os.path.join(p_dir, Config.SAVE_LBL))
        self.conc = d_lbl['conc']
        
        # --------------------------------------------------------
        # [핵심 수정] 데이터 통계 기반 스케일링 팩터 계산
        # --------------------------------------------------------
        # 1) Wind (u, v): 전체 데이터에서 u, v 중 절대값이 가장 큰 값을 찾음
        #    예: u가 -15, v가 +12면 max_uv는 15.0
        #    최소 1.0은 보장 (0으로 나누기 방지)
        max_uv_val = np.max(np.abs(self.met[:, :2]))
        self.scale_wind = max(float(max_uv_val), 1.0)
        
        # 2) Stability (L): L값의 최대 절대값
        max_L_val = np.max(np.abs(self.met[:, 2]))
        self.scale_L = max(float(max_L_val), 1.0)
        
        # 3) Concentration (C): 농도 최대값 (농도는 항상 양수이므로 그냥 max)
        max_conc_val = np.max(self.conc)
        #self.scale_conc = max(float(max_conc_val), 1.0)

        print(f"   [Stats] Wind Scale: {self.scale_wind:.2f} m/s (mapped to -1~1)")
        print(f"   [Stats] L Scale:    {self.scale_L:.2f} m (mapped to -1~1)")
        print(f"   [Stats] Conc Norm:  Log1p Transformation Applied (Inversion: expm1)")
        
        # --------------------------------------------------------
        # 기울기 계산 (기존 동일)
        real_terrain = self.terrain * self.max_h
        grad_y, grad_x = np.gradient(real_terrain, Config.DY, Config.DX)
        self.coords, self.slope_flat = self._make_coords_and_slopes(grad_x, grad_y)

    def _make_coords_and_slopes(self, gx, gy):
        # (기존 코드와 동일)
        x = np.linspace(0, 1, Config.NX)
        y = np.linspace(0, 1, Config.NY)
        z = np.linspace(0, 1, Config.NZ)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='xy')
        gxx = np.repeat(gx[:, :, np.newaxis], Config.NZ, axis=2)
        gyy = np.repeat(gy[:, :, np.newaxis], Config.NZ, axis=2)
        coords = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1).astype(np.float32)
        slopes = np.stack([gxx.flatten(), gyy.flatten()], axis=1).astype(np.float32)
        return coords, slopes

    def __len__(self):
        return len(self.met)

    def __getitem__(self, idx):
        # 1. 시퀀스 추출
        if idx < self.seq_len:
            pad_len = self.seq_len - (idx + 1)
            past_data = self.met[:idx+1, :3]
            padding = np.tile(self.met[0, :3], (pad_len, 1))
            met_seq = np.vstack([padding, past_data])
        else:
            met_seq = self.met[idx - self.seq_len + 1 : idx + 1, :3]

        # --------------------------------------------------------
        # [수정] 동적 스케일링 적용 (Input Normalization)
        # --------------------------------------------------------
        met_seq_norm = met_seq.copy()
        met_seq_norm[:, 0] /= self.scale_wind # u -> -1 ~ 1
        met_seq_norm[:, 1] /= self.scale_wind # v -> -1 ~ 1
        met_seq_norm[:, 2] /= self.scale_L    # L -> -1 ~ 1
        
        # 2. Context Map
        ctx_map = np.stack([self.terrain, self.source_q, self.source_h], axis=0)
        
        # 3. Label Calculation (Physics)
        curr_met = self.met[idx]
        z_real = self.coords[:, 2] * 200.0 # (0~1) -> (0~200m)
        slope_x = self.slope_flat[:, 0]
        slope_y = self.slope_flat[:, 1]
        
        # 물리적 바람장 계산 (Raw Scale)
        wind_label = calc_wind_profile_power_law(
            uref=curr_met[0], vref=curr_met[1], L=curr_met[2], 
            z_points=z_real, slopes=(slope_x, slope_y)
        )
        conc_label = self.conc[idx].flatten()[:, None]
        
        # --------------------------------------------------------
        # [수정] 정답 라벨 정규화 (Output Normalization)
        # --------------------------------------------------------
        # 바람 벡터 (u,v,w) 모두 scale_wind로 나눔 
        # (w는 u,v보다 작으므로 보통 -1~1 안에 안전하게 들어옴)
        wind_label_norm = wind_label / self.scale_wind
        
        # 농도
        #conc_label_norm = conc_label / self.scale_conc
        conc_label_norm = np.log1p(conc_label)
        
        return (torch.tensor(ctx_map, dtype=torch.float32), 
                torch.tensor(met_seq_norm, dtype=torch.float32), 
                torch.tensor(self.coords, dtype=torch.float32), 
                torch.tensor(wind_label_norm, dtype=torch.float32), 
                torch.tensor(conc_label_norm, dtype=torch.float32))

    # [추가] 추론 결과를 원래 값으로 복원해주는 유틸리티 함수
    def denormalize_conc(self, val):
        # log1p의 역함수는 expm1 (exp(x) - 1)
        # val이 Tensor일 수도 있고 numpy일 수도 있으므로 처리
        if isinstance(val, torch.Tensor):
            return torch.expm1(val)
        return np.expm1(val)