import torch 
import torch.nn as nn
import math
from .sublayers import MultiheadAttentionBlock, FFNBlock

class TransformerBlock(nn.Module):
    def __init__(self, d_model = 512, d_v = 64, d_k = 64, h = 8, d_ff = 2048, dropout = 0.1):
        super(TransformerBlock, self).__init__()
        self.mha_block = MultiheadAttentionBlock(d_k, d_v, d_model, h, dropout)
        self.ffn_block = FFNBlock(d_model, d_ff, dropout)
    def forward(self, input, mask):
        x = self.mha_block(input, mask)
        x = self.ffn_block(x)
        return x
