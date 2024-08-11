import torch
import torch.nn as nn
import math


class Embedder(nn.Module):
    def __init__(self,dict_size,dim_model,max_seq_len):
        super(Embedder,self).__init__()
        self.token_embedding = nn.Embedding(dict_size,dim_model)
        self.register_buffer('pos_encoding', self.pos_encoder(max_seq_len, dim_model))

    def pos_encoder(self,max_seq_len,dim_model):

        pos_encoding = torch.zeros(max_seq_len,dim_model)

        for pos in range(max_seq_len):
            for i in range(0,dim_model,2):
                pos_encoding[pos,i] = math.sin(pos/(10000**(i/dim_model)))
                if i+1<dim_model:
                    pos_encoding[pos,i+1] = math.cos(pos/(10000**((i+1)/dim_model)))

        return pos_encoding.unsqueeze(0)
    
    def forward(self,x):
        seq_len = x.size(1)

        token_embeddings = self.token_embedding(x)
        pos_encoding=self.pos_encoding[:,:seq_len,:].to(x.device)

        return token_embeddings + pos_encoding