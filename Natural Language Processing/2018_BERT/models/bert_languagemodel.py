import torch
import torch.nn as nn
import math
from .bert import BERT

# BERT Masked Language Model
class MLM(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(MLM, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim = -1)
    def forward(self, x):
        return self.softmax(self.linear(x))

# BERT Next Sentence Prediction
class NSP(nn.Module):
    def __init__(self, d_model):
        super(NSP, self).__init__()
        self.linear = nn.Linear(d_model, 2)
        self.softmax = nn.LogSoftmax(dim = -1)
    def forward(self, x):
        return self.softmax(self.linear(x))

# BERT LM
class BERT_LM(nn.Module):
    def __init__(self, max_len, vocab_size, L = 12, d_model = 512, d_v = 64, d_k = 64, h = 8, d_ff = 2048, dropout = 0.1):
        super(BERT_LM, self).__init__()
        self.bert = BERT(max_len, vocab_size, L, d_model, d_v, d_k, h, d_ff, dropout)
        self.MLM = MLM(self.bert.d_model, vocab_size)
        self.NSP = NSP(self.bert.d_model)
    def forward(self, token, segment):
        x = self.bert(token, segment)
        return self.MLM(x), self.NSP(x)
