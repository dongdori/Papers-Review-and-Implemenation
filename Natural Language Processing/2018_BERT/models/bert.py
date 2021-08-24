import torch
import torch.nn as nn
import math
from .transformer_block import TransformerBlock
from .sublayers import InputBlock 

class BERT(nn.Module):
    def __init__(self, max_len, vocab_size, L = 12, d_model = 512, d_v = 64, d_k = 64, h = 8, d_ff = 2048, dropout = 0.1):
        '''
        max_len : max length of input tokens
        vocab_size : vocabulary size
        L : n_layers of transformer block
        d_model : embedding dimension
        d_v : dimension of value in MultiheadAttention
        d_k : dimension of key in MultiheadAttention
        h : number of attention heads
        d_ff : dimension of positionwise feedforward network 
        '''
        super(BERT, self).__init__()
        self.d_model = d_model
        self.input = InputBlock(max_len, vocab_size, d_model, dropout)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, d_v, d_k, h, d_ff, dropout) for _ in range(L)])
        
    def forward(self, token, segment):
        padding_mask = (token == 0)
        x = self.input(token, segment)
        for tb in self.transformer_blocks:
            x = tb.forward(x, padding_mask)
        return x
