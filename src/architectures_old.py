import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

def positional_encoding(batch_size, dm, n):
    '''
    Function encoding positions of observations throught periodic functions
    '''
    d = torch.arange(dm, dtype=torch.double)
    pe = torch.zeros(batch_size, n, dm)
    i = torch.arange(n, dtype=torch.double)
    pe[:, :, ::2] = torch.sin(i.view(-1, n, 1)*10000**(torch.floor(d[::2].view(1, 1, -1)/2)/dm))
    pe[:, :, 1::2] = torch.cos(i.view(-1, n, 1)*10000**(torch.floor(d[1::2].view(1, 1, -1)/2)/dm))
    return pe


class LinearEmbedding(nn.Module):
    def __init__(self, batch_size=16, n=20, dm=128):
        super().__init__()
        self.n, self.dm  = n, dm
        self.projection = nn.Linear(2, dm)
        self.register_buffer("pe", positional_encoding(batch_size, dm, n))
        
    def forward(self, x):
        
        # x.size == (batch_size, n, 2)
        # s.size == (batch_size, n)
        # self.projection(x).size == (batch_size, n, dm)
        # positional_encoding(s, self.dm, self.n) == (batch_size, n, dm)
        
        return self.projection(x) + self.pe


class AttentionLayer(nn.Module):
    def __init__(self, dm=128, dk=128, dq=128, dv=128):
        super().__init__()
        self.dk = dk
        self.linear_v = nn.Linear(dm, dv) # (batch_size, n, dm) -> (batch_size, n, dv)
        self.linear_q = nn.Linear(dm, dq) # (batch_size, n, dm) -> (batch_size, n, dq)
        self.linear_k = nn.Linear(dm, dk) # (batch_size, n, dm) -> (batch_size, n, dk)
        
    def forward(self, h_a):
        V = self.linear_v(h_a) # V.shape == (batch_size, n, dv)
        K = self.linear_k(h_a) # K.shape == (batch_size, n, dk)
        Q = self.linear_q(h_a) # Q.shape == (batch_size, n, dq)
        
        KtQ_normed = torch.matmul(K, Q.transpose(-1, -2))/np.sqrt(self.dk) # KtQ_normed.shape == (batch_size, n, n)

        # returns shape (batch_size, n, dv)
        return torch.matmul(F.softmax(KtQ_normed, dim=-1), V)

    
class EncoderLayer(nn.Module):
    def __init__(self, dm=128, dk=128, dq=128, dv=128, ff_dim=512):
        super().__init__()
        self.s_att = AttentionLayer(dm=dm, dk=dk, dq=dq, dv=dv)
        self.batch_norm_1 = nn.BatchNorm1d(dv)
        self.batch_norm_2 = nn.BatchNorm1d(dv)
        self.ff = nn.Sequential(nn.Linear(dm, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dm))
        
    def forward(self, x):
        x = self.s_att(x) + x
        x = self.batch_norm_1(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.ff(x) + x
        x = self.batch_norm_2(x.transpose(-1, -2)).transpose(-1, -2)
        return x

    
class Encoder(nn.Module):
    def __init__(self, dm=128, dk=128, dq=128, dv=128, ff_dim=512, N=3):
        super().__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(dm=128, dk=128, dq=128, dv=128, ff_dim=512) for _ in range(N)])
        
    def forward(self, x):
        for module in self.encoder_layers:
            x = module(x)
        return x

    
class DecoderActor(nn.Module):
    def __init__(self, n=20, dm=128, C=10):
        super().__init__()
        self.n, self.C = n, C
        self.dm = dm
        self.linear_graph = nn.Linear(dm,dm)
        self.linear_nodes = nn.Linear(dm,dm)
        self.linear_k = nn.Linear(dm,dm)
        self.linear_q = nn.Linear(dm,dm)
        self.register_buffer('eye_minus_inf', torch.eye(self.n)*1e20)
        
    def forward(self, x):
        x_graph, _ = x.max(dim=1)
        lp_graph = self.linear_graph(x_graph)
        lp_nodes = self.linear_nodes(x)
        x = lp_nodes + lp_graph[:, None, :]
        
        K = self.linear_k(x)
        Q = self.linear_q(x)
        
        M = torch.matmul(K, Q.transpose(-1, -2))/np.sqrt(self.dm)
        M_hat = self.C*torch.tanh(M)
        M_hat -= self.eye_minus_inf
        
        return F.softmax(M_hat.view(-1, self.n**2), dim=1).view(-1, self.n, self.n)

    
class Actor(nn.Module):
    def __init__(self, batch_size=16, n=20, dm=128, dk=128, dq=128, dv=128, ff_dim=512, N=3, C=10):
        super().__init__()
        
        self.linear_embedding = LinearEmbedding(batch_size=batch_size, n=n, dm=dm)
        self.encoder = Encoder(dm=dm, dk=dk, dq=dq, dv=dv, ff_dim=ff_dim, N=N)
        self.decoder = DecoderActor(n=n, dm=dm, C=C)
        
    def forward(self, x):
        x = self.linear_embedding(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
class DecoderCritic(nn.Module):
    def __init__(self, n=20, dm=128, C=10):
        super().__init__()
        self.n, self.C = n, C
        self.linear_graph = nn.Linear(dm,dm)
        self.linear_nodes = nn.Linear(dm,dm)
        self.linear_dm = nn.Linear(dm,1)
        self.activation = nn.ReLU()
        self.linear_n = nn.Linear(n,1)
        
    def forward(self, x):
        x_graph = x.mean(dim=1)
        lp_graph = self.linear_graph(x_graph)
        lp_nodes = self.linear_nodes(x)
        x = lp_nodes + lp_graph[:, None, :]
        
        x = self.linear_dm(x).squeeze()
        x = self.activation(x)
        x = self.linear_n(x).squeeze()
        
        return x

    
class Critic(nn.Module):
    def __init__(self, batch_size=16, n=20, dm=128, dk=128, dq=128, dv=128, ff_dim=512, N=3, C=10):
        super().__init__()
        
        self.linear_embedding = LinearEmbedding(batch_size=batch_size, n=n, dm=dm)
        self.encoder = Encoder(dm=dm, dk=dk, dq=dq, dv=dv, ff_dim=ff_dim, N=N)
        self.decoder = DecoderCritic(n=n, dm=dm, C=C)
        
    def forward(self, x):
        x = self.linear_embedding(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x