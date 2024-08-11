import torch

class Tokenize():
    def __init__(self, vocab=None, max_seq_len=100, device=None):
        if vocab is None:
            vocab = {
                'hello': 0,
                'world': 1,
                '<pad>': 2,
                '<unk>': 3
            }
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.device = device or torch.device('cpu')

    def forward(self,x):
        if not isinstance(x, str):
            raise ValueError("Input must be a string.")
        
        tokens=x.lower().split()
        token_indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        if len(token_indices) < self.max_seq_len:
            token_indices += [self.vocab['<pad>']] * (self.max_seq_len - len(token_indices))
        elif len(token_indices) > self.max_seq_len:
            token_indices = token_indices[:self.max_seq_len]

        return torch.tensor([token_indices]).to(self.device)