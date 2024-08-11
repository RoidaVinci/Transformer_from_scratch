import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    def __init__(self,n_heads=6,dim_model=512,d_k=None,d_v=None):
        super(MultiHeadedAttention, self).__init__()
        self.n_heads=n_heads
        self.dim_model=dim_model

        self.d_k = d_k if d_k is not None else dim_model // n_heads
        self.d_v = d_v if d_v is not None else dim_model // n_heads

        self.Wq=nn.Linear(dim_model,self.d_k*n_heads)
        self.Wk=nn.Linear(dim_model,self.d_k*n_heads)
        self.Wv=nn.Linear(dim_model,self.d_v*n_heads)
        self.Wo=nn.Linear(dim_model,dim_model)
    
    def forward(self,Q,K,V):
        N = Q.shape[0]
        seq_len = Q.shape[1]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = Q.view(N, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(N, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(N, seq_len, self.n_heads, self.d_v).transpose(1, 2)

        W=torch.nn.functional.softmax(torch.matmul(Q,K.transpose(-2,-1))/torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)),dim=-1)
        Attention=torch.matmul(W,V)

        Attention = Attention.transpose(1, 2).contiguous().view(N, seq_len, self.dim_model)

        return self.Wo(Attention)