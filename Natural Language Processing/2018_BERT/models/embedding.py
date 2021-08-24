import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model = 512):
        super().__init__(vocab_size, d_model, padding_idx = 0)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model = 512, max_len = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SegmentEmbedding(nn.Embedding):
    def __init__(self, d_model):
        super().__init__(3, d_model, padding_idx = 0)
