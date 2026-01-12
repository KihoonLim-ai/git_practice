"""
Sequence-to-Sequence Dataset for Concentration Prediction
과거 30시간의 농도 시계열 → 미래 1시간 농도 예측

입력:
- Past concentration: (30, 21, 45, 45) - 과거 30 timesteps의 농도 맵
- Static maps: (2, 21, 45, 45) - [Terrain, Source] 정적 정보

출력:
- Future concentration: (1, 21, 45, 45) - 미래 1 timestep의 농도 맵
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset.config_param import ConfigParam as Config


class ConcentrationSeq2SeqDataset(Dataset):
    """
    시계열 예측 데이터셋: 과거 농도 → 미래 농도

    Architecture:
        Input:
            - past_conc: (seq_len, 21, 45, 45) - 과거 30시간의 농도
            - static_maps: (2, 21, 45, 45) - Terrain + Source
        Output:
            - future_conc: (1, 21, 45, 45) - 미래 1시간 농도
    """

    def __init__(self, mode='train', split_ratio=(0.8, 0.1, 0.1),
                 seq_len=30, pred_horizon=1, crop_size=32):
        """
        Args:
            mode: 'train', 'val', 'test'
            split_ratio: 데이터 분할 비율
            seq_len: 입력 시퀀스 길이 (과거 몇 시간을 볼지)
            pred_horizon: 예측 시간 간격 (몇 시간 후를 예측할지)
            crop_size: 학습 시 Random Crop 크기
        """
        self.mode = mode.lower()
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.crop_size = crop_size
        self.nz, self.ny, self.nx = Config.NZ, Config.NY, Config.NX

        # -------------------------------------------------------------------
        # 1. 데이터 로딩
        # -------------------------------------------------------------------
        p_dir = Config.PROCESSED_DIR
        print(f"[{self.mode.upper()}] Loading data for Seq2Seq model...")

        try:
            d_maps = np.load(os.path.join(p_dir, Config.SAVE_MAPS))
            d_lbl = np.load(os.path.join(p_dir, Config.SAVE_LBL))
        except FileNotFoundError:
            raise FileNotFoundError(f"Processed .npz files not found in {p_dir}")

        # 정적 지도
        self.terrain = d_maps['terrain']      # (45, 45) - 정규화된 지형 고도
        self.source_q = d_maps['source_q']    # (45, 45) - Log1p(배출량)
        self.source_h = d_maps['source_h']    # (45, 45) - 정규화된 배출원 높이

        # 농도 시계열 (전체)
        full_conc = d_lbl['conc']  # (TotalTime, 45, 45, 21)

        # -------------------------------------------------------------------
        # 2. 데이터 분할 (Train / Val / Test)
        # -------------------------------------------------------------------
        total_len = len(full_conc)
        r_train, r_val, r_test = split_ratio

        idx_train_end = int(total_len * r_train)
        idx_val_end = int(total_len * (r_train + r_val))

        if self.mode == 'train':
            self.conc_data = full_conc[:idx_train_end]
        elif self.mode == 'val':
            self.conc_data = full_conc[idx_train_end:idx_val_end]
        elif self.mode == 'test':
            self.conc_data = full_conc[idx_val_end:]

        # 유효한 샘플 수 계산
        # 최소 seq_len개 과거 데이터가 필요하고, pred_horizon만큼 미래가 있어야 함
        self.valid_indices = list(range(self.seq_len, len(self.conc_data) - self.pred_horizon))
        self.data_len = len(self.valid_indices)

        # -------------------------------------------------------------------
        # 3. 정적 3D 맵 생성
        # -------------------------------------------------------------------
        self._init_static_maps()

        print(f"   -> Mode: {self.mode.upper()}")
        print(f"   -> Sequence length: {self.seq_len} timesteps")
        print(f"   -> Prediction horizon: +{self.pred_horizon} timesteps")
        print(f"   -> Valid samples: {self.data_len}")

    def _init_static_maps(self):
        """정적 3D 지도 생성 (Height-Aware Placement)"""
        # Terrain Mask (지하는 0, 지상은 1)
        z_vals = np.linspace(0, 1.0, self.nz).reshape(self.nz, 1, 1)
        terrain_mask = (z_vals <= self.terrain[np.newaxis, :, :]).astype(np.float32)
        self.terrain_3d = terrain_mask  # (21, 45, 45)

        # Source Map (Height-Aware Placement)
        # Place emissions ONLY at the actual stack height
        source_3d = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
        rows, cols = np.where(self.source_q > 0)

        for r, c in zip(rows, cols):
            h_n = self.source_h[r, c]                    # Normalized height [0, 1]
            z_idx = int(h_n * (self.nz - 1))             # Convert to z-index [0, 20]
            source_3d[np.clip(z_idx, 0, self.nz-1), r, c] = self.source_q[r, c]

        self.source_3d = source_3d.astype(np.float32)  # (21, 45, 45)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        """
        Returns:
            past_conc: (seq_len, 21, H, W) - 과거 농도 시퀀스
            static_maps: (2, 21, H, W) - [Terrain, Source (height-aware)]
            future_conc: (1, 21, H, W) - 미래 농도 (target)
        """
        # 실제 시간 인덱스
        actual_idx = self.valid_indices[idx]

        # -------------------------------------------------------------------
        # A) 과거 농도 시퀀스 (t-seq_len ~ t-1)
        # -------------------------------------------------------------------
        start_idx = actual_idx - self.seq_len
        end_idx = actual_idx

        past_conc_list = []
        for t in range(start_idx, end_idx):
            conc_t = self.conc_data[t]  # (45, 45, 21) [Y, X, Z]
            conc_t = conc_t.transpose(2, 0, 1)  # (21, 45, 45) [Z, Y, X]
            past_conc_list.append(conc_t)

        past_conc = np.stack(past_conc_list, axis=0)  # (30, 21, 45, 45)

        # -------------------------------------------------------------------
        # B) 정적 맵 (Height-Aware)
        # -------------------------------------------------------------------
        static_maps = np.stack([
            self.terrain_3d,    # (21, 45, 45) - 지형 고도
            self.source_3d      # (21, 45, 45) - 배출원 (높이별 배치)
        ], axis=0)  # (2, 21, 45, 45)

        # -------------------------------------------------------------------
        # C) 미래 농도 (t + pred_horizon)
        # -------------------------------------------------------------------
        future_idx = actual_idx + self.pred_horizon
        future_conc = self.conc_data[future_idx]  # (45, 45, 21)
        future_conc = future_conc.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 21, 45, 45)

        # -------------------------------------------------------------------
        # D) Data Augmentation (Training Only)
        # -------------------------------------------------------------------
        if self.mode == 'train':
            # Random 90° rotation
            k = np.random.randint(0, 4)
            past_conc = np.rot90(past_conc, k, axes=(2, 3))
            static_maps = np.rot90(static_maps, k, axes=(2, 3))
            future_conc = np.rot90(future_conc, k, axes=(2, 3))

            # Random crop
            if self.crop_size < 45:
                top = np.random.randint(0, 45 - self.crop_size + 1)
                left = np.random.randint(0, 45 - self.crop_size + 1)

                past_conc = past_conc[:, :, top:top+self.crop_size, left:left+self.crop_size]
                static_maps = static_maps[:, :, top:top+self.crop_size, left:left+self.crop_size]
                future_conc = future_conc[:, :, top:top+self.crop_size, left:left+self.crop_size]

        # -------------------------------------------------------------------
        # E) Convert to PyTorch Tensors
        # -------------------------------------------------------------------
        return (
            torch.from_numpy(past_conc.copy()).float(),      # (30, 21, H, W)
            torch.from_numpy(static_maps.copy()).float(),    # (2, 21, H, W)
            torch.from_numpy(future_conc.copy()).float()     # (1, 21, H, W)
        )


def get_dataloaders_seq2seq(batch_size=8, seq_len=30, pred_horizon=1,
                             crop_size=32, num_workers=0):
    """
    Seq2Seq DataLoader 생성

    Args:
        batch_size: 배치 크기
        seq_len: 입력 시퀀스 길이 (과거 몇 시간)
        pred_horizon: 예측 간격 (몇 시간 후)
        crop_size: 학습 시 크롭 크기
        num_workers: DataLoader worker 수

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = ConcentrationSeq2SeqDataset(
        mode='train',
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        crop_size=crop_size
    )

    val_ds = ConcentrationSeq2SeqDataset(
        mode='val',
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        crop_size=45  # Full resolution
    )

    test_ds = ConcentrationSeq2SeqDataset(
        mode='test',
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        crop_size=45  # Full resolution
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
