from .Embedder import Embedder
from .Encoder import Encoder
from .OutputLayer import OutputLayer
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, dict_size, dim_model=512, n_layers=6, n_heads=8, dim_ffn=2048, max_seq_len=10000, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedder = Embedder(dict_size, dim_model, max_seq_len)
        self.encoder = Encoder(n_layers, dim_model, n_heads, dim_ffn, dropout)
        self.output_layer = OutputLayer(dim_model, dict_size)
    
    def forward(self, x):
        x = self.embedder(x)
        x = self.encoder(x)
        return self.output_layer(x)