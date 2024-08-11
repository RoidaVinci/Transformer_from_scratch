import torch.nn as nn
from .MultiHeadedAttention import MultiHeadedAttention
from .FeedForward import FeedForward
from .LayerNormalization import LayerNormalization

class EncoderLayer(nn.Module):
    def __init__(self, dim_model, n_heads, dim_ffn, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadedAttention(n_heads=n_heads,dim_model=dim_model)
        self.ffn = FeedForward(dim_model=dim_model,dim_ffn=dim_ffn)
        self.ln1 = LayerNormalization(dim_model)
        self.ln2 = LayerNormalization(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        attention = self.mha(x, x, x)
        x = x + self.dropout(attention)
        x = self.ln1(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.ln2(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, n_layers, dim_model, n_heads, dim_ffn, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(dim_model, n_heads, dim_ffn, dropout) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x