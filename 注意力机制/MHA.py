import torch
import torch.nn as nn
import math
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model%n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    def forward(self, x, mask=None):
        B, S, D = x.shape
        # bathc_size, seq_len, d_model

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, S, self.n_heads, self.d_k).transpose(1,2)
        K = K.view(B, S, self.n_heads, self.d_k).transpose(1,2)
        V = V.view(B, S, self.n_heads, self.d_k).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(self.d_k)

        if mask is None:
            mask = torch.triu(torch.ones(S, S, device = x.device), diagonal = 1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        atten_weight = torch.softmax(scores, dim=-1)
        context = torch.matmul(atten_weight, V)

        context = context.transpose(1,2).contiguous().view(B, S, D)
        output = self.W_o(context)

        return output
