import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim_model, dim_ffn):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim_model, dim_ffn)
        self.fc2 = nn.Linear(dim_ffn, dim_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
    