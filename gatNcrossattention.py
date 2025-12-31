import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
class Config:
    # 학습 하이퍼파라미터
    BATCH_SIZE = 32
    NUM_EPOCHS = 1000
    LEARNING_RATE = 1e-4
    
    # Loss Balancing (Radar Loss 스케일 보정용 가중치)
    # 속도(m/s) 오차가 위치(m) 오차보다 클 경우 조절 필요
    LAMBDA_RADAR = 0.5 
    
    # 데이터 구조
    TIME_STEPS = 4       # t-4 ~ t-1 (Input)
    NUM_POINTS = 1024    # Max Points
    
    # Unified Vector Dimension: 11
    # [x, y, dt, v_lin, v_ang, I, vr, S, id1, id2, id3]
    INPUT_DIM = 11       
    
    # 모델 구조
    EMBED_DIM = 64
    GAT_HEADS = 4
    K_NEIGHBORS = 16
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 유틸리티: 2D Odom 변환 행렬
# ==========================================
def get_transformation_matrix_2d(source_pose, target_pose):
    """
    source_pose (Past): [x, y, theta]
    target_pose (Curr): [x, y, theta]
    Return: 3x3 Homogeneous Matrix (Source -> Target Frame)
    """
    sx, sy, sth = source_pose
    T_source = np.array([
        [np.cos(sth), -np.sin(sth), sx],
        [np.sin(sth),  np.cos(sth), sy],
        [0, 0, 1]
    ])
    
    tx, ty, tth = target_pose
    T_target = np.array([
        [np.cos(tth), -np.sin(tth), tx],
        [np.sin(tth),  np.cos(tth), ty],
        [0, 0, 1]
    ])
    
    T_inv_target = np.linalg.inv(T_target)
    T_rel = np.matmul(T_inv_target, T_source)
    return T_rel

def apply_transform_2d(points_xy, T_matrix):
    N = len(points_xy)
    if N == 0: return points_xy
    
    ones = np.ones((N, 1))
    pts_homo = np.hstack([points_xy, ones]) # [N, 3]
    pts_trans_homo = np.matmul(pts_homo, T_matrix.T)
    
    return pts_trans_homo[:, :2]

# ==========================================
# 3. 데이터셋 (Dataset)
# ==========================================
class FusionDataset(Dataset):
    def __init__(self):
        # 파일 경로 (사용자가 준비한 데이터 파일)
        self.files = {
            'lidar': "LiDAR_DOGM_train1.txt",
            'radar1': "Radar1_DOGM_train1.txt",
            'radar2': "Radar2_DOGM_train1.txt",
            'odom': "Odom_DOGM_train1.txt"
        }
        
        self.data_cache = {'lidar': {}, 'radar1': {}, 'radar2': {}, 'odom': {}}
        self.timestamps = []
        
        print(f"[Dataset] Loading text files from disk...")
        self._load_all_files()
        
        if len(self.timestamps) == 0:
            raise RuntimeError("No synchronized timestamps found. Check your data files.")

        self.timestamps = sorted(list(self.data_cache['odom'].keys()))
        self.total_frames = len(self.timestamps)
        print(f"[Dataset] Total Synchronized Frames: {self.total_frames}")

    def _load_all_files(self):
        # 1. LiDAR Load
        if os.path.exists(self.files['lidar']):
            with open(self.files['lidar'], 'r') as f:
                for line in f:
                    vals = [float(x) for x in line.strip().split()]
                    t = int(vals[0])
                    if t not in self.data_cache['lidar']: self.data_cache['lidar'][t] = []
                    self.data_cache['lidar'][t].append(vals[1:])
        else:
            print(f"Warning: {self.files['lidar']} not found.")

        # 2. Radar Load
        for key in ['radar1', 'radar2']:
            if os.path.exists(self.files[key]):
                with open(self.files[key], 'r') as f:
                    for line in f:
                        vals = [float(x) for x in line.strip().split()]
                        t = int(vals[0])
                        if t not in self.data_cache[key]: self.data_cache[key][t] = []
                        self.data_cache[key][t].append(vals[1:])
            else:
                print(f"Warning: {self.files[key]} not found.")
                    
        # 3. Odom Load
        if os.path.exists(self.files['odom']):
            with open(self.files['odom'], 'r') as f:
                for line in f:
                    vals = [float(x) for x in line.strip().split()]
                    t = int(vals[0])
                    self.data_cache['odom'][t] = vals[1:]
        else:
            raise FileNotFoundError("Odom file is required!")

        # Convert to numpy
        for t in self.data_cache['lidar']: self.data_cache['lidar'][t] = np.array(self.data_cache['lidar'][t])
        for t in self.data_cache['radar1']: self.data_cache['radar1'][t] = np.array(self.data_cache['radar1'][t])
        for t in self.data_cache['radar2']: self.data_cache['radar2'][t] = np.array(self.data_cache['radar2'][t])

    def __len__(self):
        return max(0, self.total_frames - Config.TIME_STEPS)

    def __getitem__(self, idx):
        start_idx = idx
        input_sequence_tensor = []
        
        # 기준 시점: 시퀀스의 마지막 입력 프레임 (t-1)
        curr_frame_idx = start_idx + Config.TIME_STEPS - 1
        curr_t = self.timestamps[curr_frame_idx]
        curr_odom = self.data_cache['odom'][curr_t]
        curr_pose = curr_odom[:3]
        
        # --- Time Step Loop (t-4 ~ t-1) ---
        for i in range(Config.TIME_STEPS):
            past_idx = start_idx + i
            past_t = self.timestamps[past_idx]
            
            l_raw = self.data_cache['lidar'].get(past_t, np.zeros((0, 3)))
            r1_raw = self.data_cache['radar1'].get(past_t, np.zeros((0, 4)))
            r2_raw = self.data_cache['radar2'].get(past_t, np.zeros((0, 4)))
            past_odom = self.data_cache['odom'][past_t]
            
            # Ego-Motion Compensation (Past -> Curr)
            T_mat = get_transformation_matrix_2d(past_odom[:3], curr_pose)
            
            if len(l_raw) > 0: l_raw[:, :2] = apply_transform_2d(l_raw[:, :2], T_mat)
            if len(r1_raw) > 0: r1_raw[:, :2] = apply_transform_2d(r1_raw[:, :2], T_mat)
            if len(r2_raw) > 0: r2_raw[:, :2] = apply_transform_2d(r2_raw[:, :2], T_mat)
            
            dt = (past_idx - curr_frame_idx) * 0.1
            frame_vec = self.create_unified_vector(l_raw, r1_raw, r2_raw, dt, past_odom[3], past_odom[4])
            input_sequence_tensor.append(frame_vec)
            
        input_tensor = torch.stack(input_sequence_tensor) # [T, N, 11]
        
        # [Placeholder for Labels]
        # 실제 학습을 위해서는 여기에 정답 데이터 로드 로직이 필요합니다.
        # LiDAR용 정답: Position (Next Frame Position)
        target_pos = torch.randn(Config.NUM_POINTS, 2) 
        # Radar용 정답: Velocity (Current Velocity)
        target_vel = torch.randn(Config.NUM_POINTS, 2)
        
        return input_tensor, target_pos, target_vel

    def create_unified_vector(self, l_raw, r1_raw, r2_raw, dt, v_lin, v_ang):
        points_list = []
        
        # LiDAR (ID: 1, 0, 0)
        if len(l_raw) > 0:
            feat = np.zeros((len(l_raw), Config.INPUT_DIM), dtype=np.float32)
            feat[:, 0:2] = l_raw[:, 0:2]; feat[:, 2] = dt; feat[:, 3] = v_lin; feat[:, 4] = v_ang
            feat[:, 5] = l_raw[:, 2]; feat[:, 8] = 1.0
            points_list.append(feat)
        # Radar1 (ID: 0, 1, 0)
        if len(r1_raw) > 0:
            feat = np.zeros((len(r1_raw), Config.INPUT_DIM), dtype=np.float32)
            feat[:, 0:2] = r1_raw[:, 0:2]; feat[:, 2] = dt; feat[:, 3] = v_lin; feat[:, 4] = v_ang
            feat[:, 6] = r1_raw[:, 2]; feat[:, 7] = r1_raw[:, 3]; feat[:, 9] = 1.0
            points_list.append(feat)
        # Radar2 (ID: 0, 0, 1)
        if len(r2_raw) > 0:
            feat = np.zeros((len(r2_raw), Config.INPUT_DIM), dtype=np.float32)
            feat[:, 0:2] = r2_raw[:, 0:2]; feat[:, 2] = dt; feat[:, 3] = v_lin; feat[:, 4] = v_ang
            feat[:, 6] = r2_raw[:, 2]; feat[:, 7] = r2_raw[:, 3]; feat[:, 10] = 1.0
            points_list.append(feat)
            
        if len(points_list) > 0:
            all_pts = np.vstack(points_list)
        else:
            all_pts = np.zeros((1, Config.INPUT_DIM), dtype=np.float32)
            
        N = all_pts.shape[0]
        final = np.zeros((Config.NUM_POINTS, Config.INPUT_DIM), dtype=np.float32)
        if N >= Config.NUM_POINTS: final = all_pts[:Config.NUM_POINTS, :]
        else: final[:N, :] = all_pts
        return torch.from_numpy(final)

# ==========================================
# 4. 모델 아키텍처 (Multi-Head ST-GAT)
# ==========================================
class PointEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32), # Changed to match transposed dim
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
    def forward(self, x):
        # x: [B, T, N, 11]
        B, T, N, C = x.shape
        x = x.view(B * T, N, C) 
        
        # Optimize: Apply Linear, then transpose for BN
        # Layer 1
        x = self.mlp[0](x)          # Linear [B*T, N, 32]
        x = x.transpose(1, 2)       # [B*T, 32, N] for BN
        x = self.mlp[1](x)          # BN
        x = self.mlp[2](x)          # ReLU
        x = x.transpose(1, 2)       # [B*T, N, 32]
        
        # Layer 2
        x = self.mlp[3](x)          # Linear [B*T, N, 64]
        x = x.transpose(1, 2)       # [B*T, 64, N] for BN
        x = self.mlp[4](x)          # BN
        x = self.mlp[5](x)          # ReLU
        x = x.transpose(1, 2)       # [B*T, N, 64]
        
        return x

class DenseGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, k=16):
        super().__init__()
        self.heads = heads
        self.k = k
        self.head_dim = out_dim // heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, pos):
        B_tot, N, D = x.shape
        dist = torch.cdist(pos, pos)
        _, indices = torch.topk(dist, k=self.k, dim=-1, largest=False)
        
        h = self.W(x).view(B_tot, N, self.heads, self.head_dim)
        
        h_flat = h.view(B_tot * N, self.heads, self.head_dim)
        indices_flat = indices.view(B_tot * N * self.k) + \
                       (torch.arange(B_tot * N, device=x.device) * N).repeat_interleave(self.k)
        
        h_neighbors = h_flat.index_select(0, indices_flat)
        h_neighbors = h_neighbors.view(B_tot, N, self.k, self.heads, self.head_dim)
        h_self = h.unsqueeze(2).expand(-1, -1, self.k, -1, -1)
        
        cat_feat = torch.cat([h_self, h_neighbors], dim=-1)
        scores = (cat_feat * self.att.view(1, 1, 1, self.heads, -1)).sum(dim=-1)
        alpha = F.softmax(F.leaky_relu(scores, 0.2), dim=2)
        
        out = (h_neighbors * alpha.unsqueeze(-1)).sum(dim=2)
        out = out.view(B_tot, N, -1)
        return F.relu(out + x)

class TemporalCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.q_layer = nn.Linear(dim, dim)
        self.kv_layer = nn.Linear(dim, dim)
        
    def forward(self, x):
        last_frame = x[:, :, -1, :] # t-1
        Q = self.q_layer(last_frame).unsqueeze(2) 
        K = V = self.kv_layer(x)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dim)
        return torch.matmul(F.softmax(scores, dim=-1), V).squeeze(2)

class STGAT_SensorFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Shared Backbone
        self.embedding = PointEmbedding(Config.INPUT_DIM, Config.EMBED_DIM)
        self.gat = DenseGATLayer(Config.EMBED_DIM, Config.EMBED_DIM, heads=Config.GAT_HEADS, k=Config.K_NEIGHBORS)
        self.cross_attn = TemporalCrossAttention(Config.EMBED_DIM)
        
        # 2. Multi-Heads (분리된 출력층)
        # Head A: LiDAR -> Position (x, y) & Uncertainty
        self.lidar_head = nn.Sequential(
            nn.Linear(Config.EMBED_DIM, 32), nn.ReLU(),
            nn.Linear(32, 4) # [mu_x, mu_y, sigma_x, sigma_y]
        )
        
        # Head B: Radar -> Velocity (vx, vy) & Uncertainty
        self.radar_head = nn.Sequential(
            nn.Linear(Config.EMBED_DIM, 32), nn.ReLU(),
            nn.Linear(32, 4) # [mu_vx, mu_vy, sigma_vx, sigma_vy]
        )

    def forward(self, x):
        # Backbone (공유)
        B, T, N, C = x.shape
        h = self.embedding(x)
        raw_xy = x.view(B*T, N, C)[:, :, :2] 
        h_spatial = self.gat(h, raw_xy)
        h_temporal = h_spatial.view(B, T, N, -1).permute(0, 2, 1, 3)
        h_fused = self.cross_attn(h_temporal) # [B, N, Embed_Dim]
        
        # Head A: LiDAR Output
        out_lidar = self.lidar_head(h_fused)
        lidar_mu = out_lidar[:, :, :2]
        lidar_sigma = F.softplus(out_lidar[:, :, 2:]) + 1e-6
        
        # Head B: Radar Output
        out_radar = self.radar_head(h_fused)
        radar_mu = out_radar[:, :, :2]
        radar_sigma = F.softplus(out_radar[:, :, 2:]) + 1e-6
        
        return (lidar_mu, lidar_sigma), (radar_mu, radar_sigma)

# ==========================================
# 5. 메인 실행 및 학습 루프 (Training)
# ==========================================
def gaussian_nll_loss(pred_mu, pred_sigma, target):
    var = pred_sigma ** 2
    loss = 0.5 * (torch.log(var) + (target - pred_mu)**2 / var)
    return loss.mean()

if __name__ == "__main__":
    # 데이터셋 로드
    try:
        dataset = FusionDataset()
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        print("Please check if 'LiDAR_DOGM_train1.txt' and other files exist.")
        exit(1)
        
    train_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 모델 초기화
    model = STGAT_SensorFusion().to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    print(f"\n[Training Start] Device: {Config.DEVICE}")
    print(f"Strategy: Multi-Head (LiDAR->Pos, Radar->Vel)")
    
    model.train()
    
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0
        lidar_loss_accum = 0
        radar_loss_accum = 0
        
        # Target_Pos: LiDAR Ground Truth (위치)
        # Target_Vel: Radar Ground Truth (속도)
        for batch_idx, (data, target_pos, target_vel) in enumerate(train_loader):
            data = data.to(Config.DEVICE)
            target_pos = target_pos.to(Config.DEVICE)
            target_vel = target_vel.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # 1. 모델 예측 (두 헤드 모두 계산)
            (l_mu, l_sigma), (r_mu, r_sigma) = model(data)
            
            # 2. 마스크 생성 (t-1 시점 기준)
            # Input Feature Index: 8(LiDAR), 9(Radar1), 10(Radar2)
            curr_input = data[:, -1, :, :]
            mask_lidar = (curr_input[:, :, 8] == 1.0)
            mask_radar = (curr_input[:, :, 9] == 1.0) | (curr_input[:, :, 10] == 1.0)
            
            # 3. 분기 Loss 계산 (Masked Loss)
            loss_lidar = torch.tensor(0.0, device=Config.DEVICE)
            loss_radar = torch.tensor(0.0, device=Config.DEVICE)
            
            # LiDAR: Position Regression
            if mask_lidar.sum() > 0:
                loss_lidar = gaussian_nll_loss(
                    l_mu[mask_lidar], 
                    l_sigma[mask_lidar], 
                    target_pos[mask_lidar]
                )
            
            # Radar: Velocity Regression
            if mask_radar.sum() > 0:
                loss_radar = gaussian_nll_loss(
                    r_mu[mask_radar], 
                    r_sigma[mask_radar], 
                    target_vel[mask_radar]
                )
            
            # 4. 최종 Loss 합산 (Balancing)
            loss = loss_lidar + (Config.LAMBDA_RADAR * loss_radar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            lidar_loss_accum += loss_lidar.item()
            radar_loss_accum += loss_radar.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Total: {loss.item():.4f} (L: {loss_lidar.item():.4f}, R: {loss_radar.item():.4f})")
        
        avg_loss = total_loss / len(train_loader)
        print(f"==> Epoch {epoch+1} Avg Loss: {avg_loss:.4f} [L_avg: {lidar_loss_accum/len(train_loader):.4f}, R_avg: {radar_loss_accum/len(train_loader):.4f}]\n")
    
    print("[Training Complete] Model saved to 'stgat_multihead.pth'")
    torch.save(model.state_dict(), "stgat_multihead.pth")