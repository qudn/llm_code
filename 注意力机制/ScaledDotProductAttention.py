import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力 (Scaled Dot-Product Attention)
    Transformer 注意力机制的核心计算模块。
    公式: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    """

    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """
        前向传播
        Args:
            q: [batch, num_heads, seq_len_q, head_dim]
            k: [batch, num_heads, seq_len_k, head_dim]
            v: [batch, num_heads, seq_len_k, head_dim]
            mask: 注意力掩码 [batch, 1, seq_len_q, seq_len_k]
        Returns:
            output: [batch, num_heads, seq_len_q, head_dim]
        """
        head_dim = q.size(-1)

        # ---------- 1. 计算缩放点积 ----------
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        # ---------- 2. 应用 mask ----------
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # ---------- 3. softmax + dropout ----------
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # ---------- 4. 加权求和 ----------
        output = torch.matmul(attn_weights, v)

        return output
