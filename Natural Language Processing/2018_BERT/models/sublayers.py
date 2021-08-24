import torch
import torch.nn as nn
import math
from .embeddings import TokenEmbedding, PositionEmbedding, SegmentEmbedding

class InputBlock(nn.Module):
    def __init__(self, max_len, vocab_size, d_model = 512, dropout = 0.1):
        super(InputBlock, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_embedding = TokenEmbedding(vocab_size = vocab_size, d_model = d_model)
        self.pos_embedding = PositionEmbedding(d_model = d_model, max_len = max_len)
        self.seg_embedding = SegmentEmbedding(d_model = d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, seq, seg_label):
        x = self.token_embedding(sequence) + self.pos_embedding(seq) + self.seg_embedding(seg_label)
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    '''
    std : a_2
    mean : b_2
    '''
    def __init__(self, features, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.a_2*(x-mean) / (std+self.eps) + self.b_2

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_k = 64, d_v = 64, d_model = 512, h = 8, dropout = 0.1):
        super(MultiheadAttentionBlock, self).__init__()
        self.multiheadattn = nn.MultiheadAttention(embed_dim = d_model, num_heads = h, dropout = dropout, kdim = d_k, vdim = d_v)
        self.norm = LayerNorm(features = d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input, mask):
        x = self.multiheadattn(query = input, key = input, value = input, key_padding_mask = mask)
        x = self.norm(x)
        x = self.dropout(x)
        x = x + input
        return x
        
class FFNBlock(nn.Module):
    def __init__(self, d_model = 512, d_ff = 2048, dropout = 0.1):
        super(FFNBlock, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.norm = LayerNorm(features = d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input):
        x = self.linear_1(input)
        x = self.relu(x)
        x = self.linear_2(input)
        x = self.norm(x)
        x = self.dropout(x)
        x = x + input
        return x
