import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBranch(nn.Module):
    """
    [Static Branch] 2D 공간 정보(지형, 오염원) 처리 (CNN)
    - 역할: 변하지 않거나 느리게 변하는 공간적 특징(Spatial Features) 추출
    - Input: (Batch, 3, 45, 45) -> [Terrain, Source_Q, Source_H]
    - Output: (Batch, latent_dim)
    """
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        # 45x45 이미지를 점진적으로 압축
        self.conv_layers = nn.Sequential(
            # Conv 1: 45x45 -> 23x23
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            
            # Conv 2: 23x23 -> 12x12
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            # Conv 3: 12x12 -> 6x6
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            # Conv 4: 6x6 -> 3x3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        # 128채널 * 3 * 3 크기를 latent_dim으로 변환
        self.fc = nn.Linear(128 * 3 * 3, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

class SequentialMetBranch(nn.Module):
    """
    [Dynamic Branch] 기상 시계열 데이터 처리 (LSTM)
    - 역할: 과거의 기상 변화 패턴(Temporal Dynamics)을 학습하여 미래 상태 예측에 기여
    - 논문 기여점: 단순 수치 입력이 아닌 '유체 역학적 상태 흐름'을 Latent Vector로 인코딩
    - Input: (Batch, Seq_Len, 3) -> [u_ref, v_ref, L] Sequence
    - Output: (Batch, latent_dim)
    """
    def __init__(self, input_dim=3, hidden_dim=64, latent_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)
        # output: 전체 시퀀스의 출력
        # h_n: (Num_Layers, Batch, Hidden_Dim) -> 마지막 타임스텝의 히든 상태
        _, (h_n, _) = self.lstm(x)
        
        # 마지막 레이어의 마지막 시점(Last Time Step) 히든 스테이트 사용
        last_hidden = h_n[-1] 
        return self.fc(last_hidden)

class TrunkNet(nn.Module):
    """
    [Trunk Net] 3D 좌표 정보 처리 (MLP)
    - 역할: 쿼리 위치(Query Location)에 대한 기저 함수(Basis Function) 생성
    - 특징: PINN 학습(미분)을 위해 Tanh 활성화 함수 사용 (GELU보다 미분이 부드러움)
    - Input: (Batch, N_points, 3) -> [x, y, z]
    - Output: (Batch, N_points, latent_dim)
    """
    def __init__(self, input_dim=3, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class RecurrentDeepONet(nn.Module):
    """
    [Spatio-Temporal DeepONet Model]
    구조: (Static Map + Weather Sequence) * Location -> (Wind Field + Concentration)
    """
    def __init__(self):
        super().__init__()
        self.latent_dim = 128
        
        # --- 1. Branch Networks (Context Encoders) ---
        self.map_branch = ConvBranch(in_channels=3, latent_dim=self.latent_dim)
        self.met_branch = SequentialMetBranch(input_dim=3, latent_dim=self.latent_dim)
        
        # --- 2. Trunk Network (Location Encoder) ---
        self.trunk = TrunkNet(input_dim=3, latent_dim=self.latent_dim)
        
        # --- 3. Fusion Layer ---
        # 지형 정보와 기상 정보를 하나로 통합
        self.context_fusion = nn.Linear(self.latent_dim * 2, self.latent_dim)
        
        # --- 4. Prediction Heads ---
        # Head A: 3D 바람장 (u, v, w) 예측
        self.head_wind = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3) # Output: u, v, w
        )
        
        # Head B: 오염물질 농도 (C) 예측
        self.head_conc = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1), # Output: Concentration
            nn.Softplus()     # 농도는 물리적으로 음수가 될 수 없음 (0 이상 보장)
        )

    def forward(self, ctx_map, ctx_met_seq, coords):
        """
        Args:
            ctx_map: (Batch, 3, 45, 45)      - 정적 지형/오염원 맵
            ctx_met_seq: (Batch, Seq_Len, 3) - 과거 T시간 기상 시퀀스
            coords: (Batch, N_points, 3)     - 예측할 3D 좌표점들
        Returns:
            pred_wind: (Batch, N_points, 3)
            pred_conc: (Batch, N_points, 1)
        """
        # 1. Context Encoding
        b_map = self.map_branch(ctx_map)       # (B, 128)
        b_met = self.met_branch(ctx_met_seq)   # (B, 128) - LSTM의 결과
        
        # 2. Context Fusion
        # 지형 특징과 기상 트렌드를 결합
        b_combined = torch.cat([b_map, b_met], dim=1) # (B, 256)
        context = self.context_fusion(b_combined)     # (B, 128)
        
        # 3. Broadcasting for DeepONet
        # Context 벡터를 좌표 개수(N)만큼 복제하여 Trunk 출력과 연산 준비
        # (B, 128) -> (B, 1, 128) -> (B, N, 128)
        context = context.unsqueeze(1).expand(-1, coords.shape[1], -1)
        
        # 4. Location Encoding (Trunk)
        trunk_out = self.trunk(coords) # (B, N, 128)
        
        # 5. DeepONet Product (Element-wise Multiplication)
        # 특정 상황(Context)에서의 특정 위치(Trunk) 값 계산
        merged = context * trunk_out # (B, N, 128)
        
        # 6. Final Prediction
        pred_wind = self.head_wind(merged) # (B, N, 3)
        pred_conc = self.head_conc(merged) # (B, N, 1)
        
        return pred_wind, pred_conc