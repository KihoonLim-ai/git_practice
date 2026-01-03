import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. Sub-Modules (Encoders & Trunk) - 기존 구조 유지
# ==============================================================================

class Conv3dBranch(nn.Module):
    """지형 및 3D 정보 인코더 (Terrain Encoder)"""
    def __init__(self, in_channels=4, latent_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)) # (B, 64, 1, 1, 1)
        )
        self.fc = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x)

class TransformerObsBranch(nn.Module):
    """기상 시계열 정보 인코더 (Met Sequence Encoder)"""
    def __init__(self, input_dim=3, latent_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        # x: (B, Seq_len, 3)
        x = self.embedding(x)
        x = self.transformer(x)
        # 마지막 시점(t)의 정보만 사용 (Summary)
        x = x[:, -1, :] 
        return self.fc(x)

class SpatioTemporalTrunk(nn.Module):
    """좌표(x,y,z,t) 인코더 (Trunk Network)"""
    def __init__(self, input_dim=4, latent_dim=128, hidden_dim=256, num_layers=4, dropout=0.1, fourier_scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.fourier_scale = fourier_scale
        
        # Fourier Feature Mapping (High frequency details)
        self.fourier_mapping = nn.Linear(input_dim, latent_dim // 2) 
        
        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        # Fourier Embedding: sin(Wx), cos(Wx)
        proj = self.fourier_mapping(coords) * self.fourier_scale
        x = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(x)

# ==============================================================================
# 2. Main Model (ST-TransformerDeepONet) - 수정됨
# ==============================================================================

class ST_TransformerDeepONet(nn.Module):
    def __init__(self, latent_dim=128, dropout=0.1, num_heads=4, fourier_scale=10.0):
        super().__init__()
        
        # 1. Encoders
        self.map_encoder = Conv3dBranch(in_channels=4, latent_dim=latent_dim)
        self.obs_encoder = TransformerObsBranch(input_dim=3, latent_dim=latent_dim, num_heads=num_heads, dropout=dropout)
        self.trunk = SpatioTemporalTrunk(input_dim=4, latent_dim=latent_dim, dropout=dropout, fourier_scale=fourier_scale)
        
        # 2. Fusion (Map + Obs)
        self.fusion = nn.Linear(latent_dim * 2, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 3. Heads (DeepONet Output Logic)
        # Wind Head: Changes due to terrain (Perturbation)
        self.head_wind = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3) # (du, dv, w)
        )
        
        # Conc Head: Concentration
        # Input: Latent + Predicted Wind (Physics-Informed)
        self.head_conc = nn.Sequential(
            nn.Linear(latent_dim + 3, 64),
            nn.GELU(),
            nn.Linear(64, 1) # Concentration
        )

    def forward(self, ctx_map, obs_seq, query_coords, global_wind):
        """
        [Inputs]
        ctx_map: (B, 4, D, H, W) - Terrain Info
        obs_seq: (B, Seq, 3) - Meteorological Sequence
        query_coords: (B, N, 4) - [x, y, z, t]
        global_wind: (B, 2) - [u_top, v_top] (상층부 평균 풍속)
        
        [Outputs]
        pred_wind: (B, N, 3) - Final Wind (u, v, w)
        pred_conc: (B, N, 1) - Concentration
        """
        
        # -----------------------------------------------------------
        # 1. Feature Encoding & Trunk
        # -----------------------------------------------------------
        z_map = self.map_encoder(ctx_map)
        z_obs = self.obs_encoder(obs_seq)
        
        # Context Fusion
        z_ctx = self.fusion(torch.cat([z_map, z_obs], dim=1)) # (B, Latent)
        z_ctx = self.dropout(z_ctx)
        
        # Trunk Output (Coordinate Features)
        z_trunk = self.trunk(query_coords) # (B, N, Latent)
        
        # DeepONet Merge (Dot product style via element-wise mul)
        merged = z_ctx.unsqueeze(1) * z_trunk # (B, N, Latent)
        
        # -----------------------------------------------------------
        # 2. Wind Prediction (Physics-Informed Residual Learning)
        # -----------------------------------------------------------
        # 모델은 "지형에 의한 변화량(Delta)"만 예측
        # pred_delta = self.head_wind(merged) # (B, N, 3) -> (du, dv, w)
        
        # [핵심] Global Condition Injection
        # global_wind (B, 2)를 (B, N, 2)로 확장
        # 상층부 풍속을 모든 좌표의 기본값(Base)으로 설정
        # 1. Base Wind 생성
        num_coords = query_coords.shape[1]
        base_uv = global_wind.unsqueeze(1).expand(-1, num_coords, -1) 
        base_w = torch.zeros((base_uv.shape[0], num_coords, 1), device=base_uv.device)
        base_wind = torch.cat([base_uv, base_w], dim=-1) # (B, N, 3)

        # 2. Delta 예측
        pred_delta = self.head_wind(merged) # (B, N, 3)

        # 3. [수정됨] 합친 후에 물리 제약 걸기!
        # 먼저 베이스와 델타를 합칩니다.
        raw_wind = base_wind + pred_delta

        # 4. [물리 법칙: Log Profile 강제 적용]
        # z 좌표 가져오기 (0.0 ~ 1.0)
        z_vals = query_coords[..., 2:3] # (B, N, 1)
        
        # 지표면(z=0)에서 무조건 0이 되도록 마스크 생성
        # z^0.3 (일반적인 대기 경계층 프로파일)
        height_factor = torch.pow(z_vals + 1e-6, 0.3)
        
        # ★★★ 핵심 수정 ★★★
        # 최종 결과에 마스크를 곱해버립니다. 
        # 이렇게 하면 신경망이 아무리 엉뚱한 값을 내뱉어도 바닥은 무조건 0이 됩니다.
        pred_wind = raw_wind * height_factor
        
        # -----------------------------------------------------------
        # 3. Concentration Prediction
        # -----------------------------------------------------------
        # 농도 예측 시, 방금 구한 "물리적으로 타당한 바람"을 입력으로 같이 줌
        conc_input = torch.cat([merged, pred_wind], dim=-1)
        pred_conc = self.head_conc(conc_input)
        
        return pred_wind, pred_conc