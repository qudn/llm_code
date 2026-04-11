import torch
import torch.nn as nn

# 预计算 cos sin → rotate_half → qcos + rotate_half(q)sin
class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding, RoPE)
    通过旋转向量的方式把绝对位置编码为相对位置，具有极好的长度外推性。
    广泛用于 Llama、Mistral、Gemma 等现代大模型。
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算 cos 和 sin（面试高频考点）
        cos, sin = self._precompute_freqs()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_freqs(self):
        """预计算旋转频率"""
        # inv_freq: [head_dim//2]
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        # t: [max_seq_len]
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        # angles: [max_seq_len, head_dim]
        angles = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
        return angles.cos(), angles.sin()

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        对 Query 和 Key 施加 RoPE
        Args:
            q, k: [batch, seq_len, num_heads, head_dim]
        """
        seq_len = q.shape[1]

        # 取出当前序列长度的 cos / sin
        cos = self.cos[:seq_len].view(1, seq_len, 1, self.head_dim)
        sin = self.sin[:seq_len].view(1, seq_len, 1, self.head_dim)

        def rotate_half(x):
            """RoPE 核心技巧：把 [x1, x2] 变成 [-x2, x1]"""
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        # ---------- 核心旋转公式 ----------
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

        return q_rot, k_rot
