import torch.nn as nn

class OutputLayer(nn.Module):
    def __init__(self, dim_model, dict_size):
        super(OutputLayer,self).__init__()
        self.fc = nn.Linear(dim_model, dict_size)
    
    def forward(self, x):
        return self.fc(x)
        