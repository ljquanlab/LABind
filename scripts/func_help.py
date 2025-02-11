import os
import random
import numpy as np
import torch
import torch.nn.functional as F
def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features
def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn
def gather_edges(edges,nodes,e_idx):
    # edges [B,N,K,C] nodes [B,N,C] e_idx [B,N,K]
    B, N, K, C = edges.shape
    node_j = torch.gather(nodes.unsqueeze(2).expand(-1, -1, K, -1), 1, e_idx.unsqueeze(-1).expand(-1, -1, -1, C))
    node_i = nodes.unsqueeze(2).repeat(1, 1, K, 1)
    return torch.cat([node_i, edges, node_j], dim=-1) # [B, N, K, 3*C]

def setALlSeed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
        
def get_std_opt(len_size, batch_size, parameters, d_model, top_lr=0.0005):
    warmup_epoch = 11
    step_each_epoch = int(len_size / batch_size)
    warmup = warmup_epoch * step_each_epoch
    factor = top_lr / (d_model ** (-0.5) * min(warmup ** (-0.5), warmup * warmup ** (-1.5)))

    return NoamOpt(
        d_model, factor, warmup, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9)
    )
