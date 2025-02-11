import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# Custom function
from conv_util import *
from config import *

def setALlSeed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

# ---------------Graph----------------


class EdgeFeature(nn.Module):
    def __init__(self, num_hidden=64, rbf_num=16, top_k=5, D_max=20.):
        super(EdgeFeature, self).__init__()
        self.top_k = top_k
        self.rbf_num = rbf_num
        self.D_max = D_max
        self.edge_emb = nn.Linear(rbf_num * 6 + 25, num_hidden)
        self.norm_edge = nn.LayerNorm(num_hidden)
        self.node_emb = nn.Linear(rbf_num * 15 + 27, num_hidden)
        self.norm_node = nn.LayerNorm(num_hidden)

    def forward(self, xyz, mask):
        # 根据xyz生成节点特征 N-CA CA-C CA-R CA-O
        #                  0  1  1  2 1 4 2  3
        CaX = xyz[:, :, 1]  # [B, N, 3]
        edge_index = self._distance(CaX, mask)
        
        node_angle = self._node_angle(xyz, mask)  # [B,N,12]
        # N-Ca Ca-R Ca-C C-O
        node_dir,edge_dir = self._node_direct(xyz,edge_index)  # [B,N,12]

        node_rbf = self._node_rbf(xyz)  # [B,N, rbf_num*15]
        
        geo_node_feat = torch.cat([node_dir, node_angle, node_rbf], dim=-1)  # [B, N, 12 + 15 + rbf_num * 15]
        # edge, edge_index = self._edge_feature(xyz, mask)
        edge_rbf = self._edge_rbf(xyz, edge_index)
        edge_ori = self._edge_orientations(CaX, edge_index)
        geo_edge_feat = torch.cat([edge_dir, edge_ori, edge_rbf], dim=-1) # [B, N, K, 18 + 7 + 6 * rbf_num]
        # rfeat: [B,N,C] xyz: [B,N,5,3] mask: [B,N]
        # 由于xyz是残基中5个点的坐标，我们将其转化为所需要的
        # N,CA,C,O,R
        # [B, N, 12+15+rbf_num*15] [B, N, K, 7 + 18 + 6 * rbf_num]
        node = self.norm_node(self.node_emb(geo_node_feat))
        edge = self.norm_edge(self.edge_emb(geo_edge_feat))
        return node, edge, edge_index # [B, N, 20+C] [B, L, K, self.num_rbf + 7] [B, L, K]

    def _distance(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)  # 距离矩阵 [N, L, L]
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max  # 距离矩阵 [N, L, L]
        # [N, L, k]  D_neighbors为具体距离值（从小到大），E_idx为对应邻居节点的编号
        _, E_idx = torch.topk(D_adjust, self.top_k, dim=-1, largest=False)
        return E_idx

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))
        def _R(i, j): return R[:, :, :, i, j]
        signs = torch.sign(torch.stack([
            _R(2, 1) - _R(1, 2),
            _R(0, 2) - _R(2, 0),
            _R(1, 0) - _R(0, 1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        return Q

    def _edge_rbf(self, X, edge_index, D_min=0., D_max=20.):
        D_count = self.rbf_num
        # Distance radial basis function
        # X: [B, L, 6, 3] edge_index: [B, L, K]
        K = edge_index.shape[-1]
        X_expand = X.unsqueeze(2).expand(-1, -1, K, -1, -1)  # [B, N, K, 6, 3]
        # 根据邻接矩阵索引获取相邻点的位置，并扩展维度以匹配expanded_positions
        X_neigh = X_expand.gather(1, edge_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 6, 3))
        # 计算差值
        CaX = X[:, :, 1]  # [B, L, 3]
        # 计算欧几里得距离，即在最后一个维度上进行平方和开根号
        D = torch.norm(X_neigh - CaX.unsqueeze(2).unsqueeze(3), dim=-1)  # [B, L, K, 6] 6个点的距离
        
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)  # [self.num_rbf]
        D_mu = D_mu.view([1, 1, 1, 1, -1])  # [1, 1, 1, self.num_rbf]
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1).to(D.device)  # [B, L, K, 1]
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)  # [B, L, K, self.num_rbf]
        RBF = RBF.flatten(-2)  # [B, L, K, 6*self.num_rbf]
        return RBF  # [B, L, K, 6*self.num_rbf]

    def _node_rbf(self, X, D_min=0., D_max=20.):
        # Distance radial basis function
        D_count = self.rbf_num
        D_mu = torch.linspace(D_min, D_max, D_count, device=X.device)
        D_mu = D_mu.view([1, -1])
        D_sigma = (D_max - D_min) / D_count
        rel_list = [[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4], [1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]]  # 对应元素顺序
        D = torch.norm(X[:, :, rel_list[0]] - X[:, :, rel_list[1]], dim=-1)  # [B, N, 15]
        D_expand = torch.unsqueeze(D, -1)
        D_out = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2).flatten(-2)
        return D_out

    def _node_angle(self, X, mask, eps=1e-7):
        # psi, omega, phi
        # print([3*X.shape[0], 3])
        # X [B, N, 6, 3]
        B = X.shape[0]
        X = torch.reshape(X[:, :, :3], [B, 3*X.shape[1], 3])

        dX = X[:, 1:] - X[:, :-1]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2]
        u_1 = U[:, 1:-1]
        u_0 = U[:, 2:]

        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
        
        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [B, -1, 3])
        dihedral = torch.cat([torch.cos(D), torch.sin(D)], -1)
        # alpha, beta, gamma
        cosD = (u_2 * u_1).sum(-1)  # alpha_{i}, gamma_{i}, beta_{i+1}
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.acos(cosD)
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [B, -1, 3])
        bond_angles = torch.cat((torch.cos(D), torch.sin(D)), -1)
        node_angles = torch.cat((dihedral, bond_angles), -1)
        # 修正最后一个特征 1,2,->1  4,5->0 
        # 根据每个mask的长度来改正
        for idx in range(node_angles.shape[0]):
            node_angles[idx][mask[idx].sum()-1] = 0
        return node_angles  # B,N,12

    def _node_direct(self, X, edge_index):
        # X [B, N, 6, 3] edge_index [B, N, K]
        # src -> dst
        A_n = X[:, :, 0]  # [B,N,3]
        A_ca = X[:, :, 1]  # [B,N,3]
        A_c = X[:, :, 2]  # [B,N,3]
        u = F.normalize(A_n-A_ca, dim=-1)
        v = F.normalize(A_ca-A_c, dim=-1)
        b = F.normalize(u - v, dim=-1)
        n = F.normalize(torch.cross(u, v, dim=-1), dim=-1)
        local_frame = torch.stack([b, n, torch.cross(b, n, dim=-1)], dim=-1)  # [B, N, 3, 3]
        t = F.normalize(X[:, :, [0, 2, 3, 4, 5]] - A_ca.unsqueeze(-2), dim=-1)  # [B, N, 5, 3]
        node_direct = torch.matmul(t, local_frame).flatten(-2)

        X_expand = X.unsqueeze(2).expand(-1, -1, self.top_k, -1, -1)  # [B, N, K, 6, 3]
        X_neigh = X_expand.gather(1, edge_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 6, 3)) # [B, N, K, 6, 3]
        t = F.normalize(X_neigh - A_ca.unsqueeze(-2).unsqueeze(-2), dim=-1)  # [B, N, K, 6, 3]
        edge_direction = torch.matmul(t, local_frame.unsqueeze(2)).flatten(-2)  # [B, N, K, 15]
        
        return node_direct,edge_direction  # [B, N, 15] [B, N, K, 18]

    def _edge_orientations(self, X, E_idx, eps=1e-6):
        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)  # 少了第一个（u0）
        u_2 = U[:, :-2, :]  # u 1~n-2
        u_1 = U[:, 1:-1, :]  # u 2~n-1
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)  # n 1~n-2
        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)  # b 角平分线向量
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2, dim=-1)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0, 0, 1, 2), 'constant', 0)  # [B, L, 9]
    
        O_neighbors = Func.gather_nodes(O, E_idx)  # [B, L, K, 9]
        X_neighbors = Func.gather_nodes(X, E_idx)  # [B, L, K, 3]

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3, 3])  # [B, L, 3, 3]
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])  # [B, L, K, 3, 3]

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)  # [B, L, K, 3]
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)  # [B, L, K, 3]
        dU = F.normalize(dU, dim=-1)
        
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2),
                         O_neighbors)  # [B, L, K, 3, 3]
        Q = self._quaternions(R)  # [B, L, K, 4]
        # Orientation features
        O_features = torch.cat((dU, Q), dim=-1)  # [B, L, K, 7]
        return O_features

class LABind(nn.Module):
    def __init__(self, rfeat_dim=1024, ligand_dim=64, hidden_dim=256, heads=4, augment_eps=0.1, rbf_num=16, top_k=5, attn_drop=0.2, dropout=0.2, num_layers=2):
        super(LABind, self).__init__()
        self.augment_eps = augment_eps
        self.in_mlp = LMlp(rfeat_dim, rfeat_dim//2, hidden_dim)
        self.lig_mlp = easyMLP(ligand_dim, hidden_dim)
        self.edge_feature = EdgeFeature(hidden_dim, rbf_num, top_k)
        self.f_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim*2, eps=1e-6), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim*2, hidden_dim), 
            nn.SiLU(), 
            nn.LayerNorm(hidden_dim, eps=1e-6))  # [B, N, hidden_dim] 将atom position in residue, surface feature, node feature进行处理到hidden_dim 融合三种特征作为图卷积的输入

        self.conv_layers = nn.ModuleList([
            GraphTransformer(hidden_dim, hidden_dim*2, heads, attn_drop, dropout)
            for _ in range(num_layers)])

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, rfeat, ligand, xyz, mask=None):
        # rfeat: [B, N, rfeat_dim] ligand: [B, N, ligand_dim] xyz: [B, N, 6, 3] hb_attr: [B, L, 3] hb_idx: [B, L] sb_attr: [B, L, 3] sb_idx: [B, L] lig_idx: [B] mask: [B, N]
        if self.training and self.augment_eps > 0.:
            xyz = xyz + torch.randn_like(xyz) * self.augment_eps
            rfeat = rfeat + torch.randn_like(rfeat) * self.augment_eps
            ligand = ligand + torch.randn_like(ligand) * self.augment_eps
        rfeat = self.in_mlp(rfeat)  # [B, N, hidden_dim] 将残基特征进行处理到hidden_dim
        ligand = self.lig_mlp(ligand)  
        node, edge, e_idx = self.edge_feature(xyz, mask)
        node = torch.cat([rfeat,node], dim=-1)
        node = self.f_mlp(node)  # [B, N, hidden_dim]

        mask_attend = Func.gather_nodes(mask.unsqueeze(-1),  e_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.conv_layers:
            node, edge, ligand = layer(node, edge, e_idx, ligand, mask, mask_attend)
        out = self.out_mlp(node)  # node [B,N,C]
        return out.squeeze(-1)
    