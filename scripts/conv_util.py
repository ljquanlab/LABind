import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import func_help as Func
from config import *
device = torch.device('cuda') 
    
class CrossAttention(nn.Module):
    def __init__(self, d_model=256, heads=8, attn_dropout=0.1,dropout=0.1):
        super(CrossAttention, self).__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model,d_model* heads)
        self.key = nn.Linear(d_model,d_model* heads)
        self.value = nn.Linear(d_model,d_model* heads)
        self.out = nn.Linear(d_model* heads,d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, node, ligand):
        # 
        B,N,C = node.size()
        _, L, _ = ligand.size()
        query = self.query(node).view(B,N,-1,self.d_model).transpose(1,2) # [B, heads, N, C]
        key = self.key(ligand).view(B,L,-1,self.d_model).transpose(1,2) # [B, heads, N, C]
        value = self.value(ligand).view(B,L,-1,self.d_model).transpose(1,2) # [B, heads, N, C]
        attn = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(self.d_model) # [B, heads, N, N]
        attn = F.softmax(attn,dim=-1) # 修改为softmax
        attn = self.attn_dropout(attn)
        x = torch.matmul(attn,value).transpose(1,2).contiguous().view(B,N,-1) # [B, N, C*heads]
        x = self.dropout(x)
        x = self.out(x)
        return x
   
class LMlp(nn.Module):
    def __init__(self,in_dim,hidden_dim=256,out_dim=256):
        super(LMlp,self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(out_dim)
    def forward(self,x):
        x = F.leaky_relu(self.fc1(x))
        x = self.ln1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.ln2(x)
        return x
class FeedForward(nn.Module):
    '''
        这是一个前馈神经网络，用于序列数据的处理
    '''
    def __init__(self,d_model=256,d_ff=512,dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        # x: [batch_size,seq_len,d_model]
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4,attn_drop=0.2):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.attn_drop = nn.Dropout(attn_drop)
        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """ Numerically stable masked softmax """
        negative_inf = np.finfo(np.float32).min
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf).to(attend_logits.device))
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend * attend
        return attend

    def forward(self, h_V, h_E, mask_attend=None):

        # Queries, Keys, Values
        n_batch, n_nodes, n_neighbors = h_E.shape[:3]
        n_heads = self.num_heads

        d = int(self.num_hidden / n_heads)
        Q = self.W_Q(h_V).view([n_batch, n_nodes, 1, n_heads, 1, d])
        K = self.W_K(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d, 1])
        V = self.W_V(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d])

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K).view([n_batch, n_nodes, n_neighbors, n_heads]).transpose(-2,-1)
        attend_logits = attend_logits / np.sqrt(d)

        if mask_attend is not None:
            # Masked softmax
            mask = mask_attend.unsqueeze(2).expand(-1,-1,n_heads,-1)
            attend = self._masked_softmax(attend_logits, mask) # [B, L, heads, K]
        else:
            attend = F.softmax(attend_logits, -1)
        attend = self.attn_drop(attend)
        # Attentive reduction
        h_V_update = torch.matmul(attend.unsqueeze(-2), V.transpose(2,3)) # [B, L, heads, 1, K] × [B, L, heads, K, d]
        h_V_update = h_V_update.view([n_batch, n_nodes, self.num_hidden])
        h_V_update = self.W_O(h_V_update)
        return h_V_update
class GraphTransformer(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, attn_drop=0.2, dropout=0.2):
        super(GraphTransformer, self).__init__()
        self.dropout = nn.Dropout(dropout) # dropout layer
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(3)]) # attention后的norm层
        
        self.CrossAttn = CrossAttention(num_hidden, num_heads, attn_drop, dropout) # 与配体的交叉注意力
        self.NeighAttn = NeighborAttention(num_hidden, num_in, num_heads, attn_drop) # 邻居注意力 关注邻居节点
        
        self.dense = FeedForward(num_hidden, num_hidden * 4, dropout) # 前馈神经网络
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden, dropout)
        
    def forward(self, node, edge, edge_index, ligand, mask=None, mask_attend=None):    # mask_attend [B, L, K]
        # node: [B, L, D] edge: [B, L, K, D] E_idx: [B, L, K]
        # Concatenate node_i to h_E_ij       
        """ Parallel computation of full transformer layer """
        # Self-attention  
        h_EV = Func.cat_neighbors_nodes(node, edge, edge_index) # 聚合邻居节点和边特征 [edge K nodej]
         
        dh = self.NeighAttn(node, h_EV, mask_attend) # [nodei] 同  [edge K*nodej] 进行注意力聚合
        node = self.norm[0](node + self.dropout(dh)) # Add & Norm
        
        dh = self.CrossAttn(node, ligand) 
        node = self.norm[1](node + self.dropout(dh)) # Add & Norm
        
        dh = self.dense(node) # 前馈神经网络
        node = self.norm[2](node + self.dropout(dh)) # Add & Norm
        if mask is not None: 
            node = mask.unsqueeze(-1) * node # mask
        # global update for ligand
        ligand = self.context(node, ligand, mask) # 使用全局信息更新配体特征
        edge = self.edge_update(node, edge, edge_index)
        return node, edge, ligand
        
    
class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(num_hidden)
        self.EdgeMLP = nn.Sequential(
            nn.Linear(3 * num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU()
        )

    def forward(self, node, edge, edge_index):
        h_VE = Func.gather_edges(edge, node, edge_index) # [B, N, K, 3*C]
        edge = self.norm(edge + self.dropout(self.EdgeMLP(h_VE)))
        return edge


class Context(nn.Module):
    def __init__(self, num_hidden,dropout=0.2):
        super(Context, self).__init__()
        self.ContextMLP = nn.Sequential(
            nn.Linear(3 * num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU()
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(num_hidden)
    def forward(self, node, ligand, mask=None):
        # node [B,N,C] ligand [B,1,C] mask [B,N]
        mean_node = node * mask.unsqueeze(-1)
        mean_node = torch.sum(mean_node,dim=1)/torch.sum(mask,dim=1).unsqueeze(-1)
        max_node,_= torch.max(node,dim=1)
        mean_node = mean_node.unsqueeze(1)
        max_node = max_node.unsqueeze(1)
        h_L = torch.cat([mean_node,max_node,ligand],dim=-1)
        ligand = self.norm(ligand + self.dropout(self.ContextMLP(h_L)))
        return ligand
    
class easyMLP(nn.Module):
    def __init__(self,in_dim=64,out_dim=64):
        super(easyMLP,self).__init__()
        self.fc1 = nn.Sequential(
                nn.LayerNorm(in_dim, eps=1e-6)
                ,nn.Linear(in_dim, out_dim)
                ,nn.LeakyReLU()
                )
    def forward(self,x):
        return self.fc1(x)
