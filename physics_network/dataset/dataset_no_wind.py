import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset.config_param import ConfigParam as Config

class AermodDatasetNoWind(Dataset):
    """
    바람/기상 데이터 없이 정적 지도만 사용하는 간소화 데이터셋

    입력: Terrain + Source maps (시간 무관)
    출력: Concentration field (시간별 Ground Truth)

    이 데이터셋은 "오염원과 지형만으로 농도를 예측할 수 있는가?"를 테스트합니다.
    """
    def __init__(self, mode='train', split_ratio=(0.8, 0.1, 0.1), crop_size=32):
        """
        Args:
            mode: 'train', 'val', 'test'
            split_ratio: 데이터 분할 비율 (train, val, test)
            crop_size: 학습 시 Random Crop 크기 (기본 32x32)
        """
        self.mode = mode.lower()
        self.crop_size = crop_size
        self.nz, self.ny, self.nx = Config.NZ, Config.NY, Config.NX

        # -------------------------------------------------------------------
        # 1. 데이터 로딩 (바람/기상 제외)
        # -------------------------------------------------------------------
        p_dir = Config.PROCESSED_DIR
        print(f"[{self.mode.upper()}] Loading static maps + concentration labels...")

        try:
            d_maps = np.load(os.path.join(p_dir, Config.SAVE_MAPS))
            d_lbl = np.load(os.path.join(p_dir, Config.SAVE_LBL))
        except FileNotFoundError:
            raise FileNotFoundError(f"Processed .npz files not found in {p_dir}")

        # 정적 지도 (시간 무관)
        self.terrain = d_maps['terrain']    # (45, 45)
        self.source_q = d_maps['source_q']  # (45, 45)

        # 농도 시계열 (Ground Truth)
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

        self.data_len = len(self.conc_data)

        # -------------------------------------------------------------------
        # 3. 정적 3D 맵 생성
        # -------------------------------------------------------------------
        self._init_static_maps()

        print(f"   -> Mode: {self.mode.upper()} | Available Samples: {self.data_len}")

    def _init_static_maps(self):
        """정적 3D 지도 생성 (Terrain + Source)"""
        # Terrain Mask: 원본 dataset과 동일한 방식
        # z_vals는 0~1 범위, terrain은 이미 정규화되어 있음
        z_vals = np.linspace(0, 1.0, self.nz).reshape(self.nz, 1, 1)
        # terrain은 이미 process_maps.py에서 정규화됨 (0~1 범위)
        terrain_mask = (z_vals <= self.terrain[np.newaxis, :, :]).astype(np.float32)
        self.terrain_3d = terrain_mask[np.newaxis, :, :, :]  # (1, 21, 45, 45)

        # Source Map: 모든 Z 레벨에 동일하게 복제
        source_3d = np.tile(
            self.source_q[np.newaxis, np.newaxis, :, :],
            (1, self.nz, 1, 1)
        )  # (1, 21, 45, 45)
        self.source_3d = source_3d.astype(np.float32)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        """
        Returns:
            input_vol: (2, 21, H, W) - [Terrain, Source]
            target_conc: (1, 21, H, W) - Concentration GT
        """
        # -------------------------------------------------------------------
        # A) Input Volume: 2채널 (Terrain + Source)
        # -------------------------------------------------------------------
        input_vol = np.concatenate([
            self.terrain_3d,  # (1, 21, 45, 45)
            self.source_3d    # (1, 21, 45, 45)
        ], axis=0)  # (2, 21, 45, 45)

        # -------------------------------------------------------------------
        # B) Target Concentration
        # -------------------------------------------------------------------
        conc_vol = self.conc_data[idx]  # (45, 45, 21) [Y, X, Z]
        conc_vol = conc_vol.transpose(2, 0, 1)[np.newaxis, ...]
        # Transpose to (21, 45, 45) then add channel dim
        # Final: (1, 21, 45, 45)

        # -------------------------------------------------------------------
        # C) Data Augmentation (Training Only)
        # -------------------------------------------------------------------
        if self.mode == 'train':
            # Random 90° rotation
            k = np.random.randint(0, 4)
            input_vol = np.rot90(input_vol, k, axes=(2, 3))
            conc_vol = np.rot90(conc_vol, k, axes=(2, 3))

            # Random crop 45×45 → crop_size×crop_size
            if self.crop_size < 45:
                top = np.random.randint(0, 45 - self.crop_size + 1)
                left = np.random.randint(0, 45 - self.crop_size + 1)
                input_vol = input_vol[:, :, top:top+self.crop_size, left:left+self.crop_size]
                conc_vol = conc_vol[:, :, top:top+self.crop_size, left:left+self.crop_size]

        # -------------------------------------------------------------------
        # D) Convert to PyTorch Tensors
        # -------------------------------------------------------------------
        return (
            torch.from_numpy(input_vol.copy()),   # (2, 21, H, W)
            torch.from_numpy(conc_vol.copy())     # (1, 21, H, W)
        )


def get_dataloaders_no_wind(batch_size=32, crop_size=32, num_workers=0):
    """
    바람 데이터 없는 DataLoader 생성

    Args:
        batch_size: 배치 크기
        crop_size: 학습 시 크롭 크기 (검증/테스트는 45로 고정)
        num_workers: DataLoader worker 수 (Windows에서는 0 권장)

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = AermodDatasetNoWind(mode='train', crop_size=crop_size)
    val_ds = AermodDatasetNoWind(mode='val', crop_size=45)  # Full resolution
    test_ds = AermodDatasetNoWind(mode='test', crop_size=45)

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
