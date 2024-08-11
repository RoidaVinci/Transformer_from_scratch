import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self,dim_model,eps=1e-6):
        super(LayerNormalization,self).__init__()
        self.layer_norm=nn.LayerNorm(dim_model, eps=eps)
        self.gamma = nn.Parameter(torch.ones(dim_model))
        self.beta = nn.Parameter(torch.zeros(dim_model))
        self.eps=eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        x_normalized = (x-mean)/torch.sqrt(variance+self.eps)

        return self.gamma * x_normalized + self.beta