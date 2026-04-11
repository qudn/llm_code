import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力 (Multi-Head Attention)
    Transformer 核心组件，通过多个注意力头并行捕捉不同子空间的特征。
    支持自注意力 (Q=K=V) 和交叉注意力 (Q != K=V)。
    """

    def __init__(self, model_dim: int, num_heads: int, dropout_p: float = 0.0):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim 必须能被 num_heads 整除"
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads   # 每个头的维度

        # 三个投影层 + 一个输出投影层
        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)
        self.w_o = nn.Linear(model_dim, model_dim)
        
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x_query: torch.Tensor, x_context: torch.Tensor = None, mask: torch.Tensor = None):
        """
        前向传播
        Args:
            x_query:   [batch, seq_len_q, model_dim]
            x_context: [batch, seq_len_k, model_dim]，None 表示自注意力
            mask:      注意力掩码 [batch, 1, seq_len_q, seq_len_k]
        """
        batch = x_query.size(0)
        seq_len_q = x_query.size(1)

        # ---------- 线性投影 ----------
        q = self.w_q(x_query)
        k = self.w_k(x_context if x_context is not None else x_query)
        v = self.w_v(x_context if x_context is not None else x_query)

        # ---------- 分头 ----------
        # [batch, seq_len, model_dim] -> [batch, num_heads, seq_len, head_dim]
        q = q.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # ---------- Scaled Dot-Product Attention ----------
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)          # [batch, num_heads, seq_len_q, head_dim]

        # ---------- 合并多头 ----------
        context = context.transpose(1, 2).contiguous().view(batch, seq_len_q, self.model_dim)

        # 输出投影
        output = self.w_o(context)
        return output
