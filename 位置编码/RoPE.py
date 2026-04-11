import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding, RoPE)
    核心思想：通过旋转矩阵把绝对位置信息编码为相对位置信息，具有极好的长度外推能力。
    广泛用于 Llama、Mistral、Gemma、Qwen 等现代大模型。
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim          # 每个 attention head 的维度（通常 64 或 128）
        self.max_seq_len = max_seq_len    # 支持的最大序列长度
        self.theta = theta                # 旋转角度的基数（论文中默认 10000.0）

        # 预计算 cos 和 sin（面试高频考点！）
        # 优点：避免每次 forward 都重复计算，节省时间和显存
        cos, sin = self._precompute_freqs()
        self.register_buffer("cos", cos, persistent=False)  # 不会被 optimizer 更新
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_freqs(self):
        """预计算旋转频率（inv_freq）和角度"""
        # inv_freq: [head_dim//2]
        # 公式：1 / (theta ** (2i / head_dim))，i = 0,1,2,...,head_dim//2-1
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )

        # t: [max_seq_len]，代表每个 token 的位置索引
        t = torch.arange(self.max_seq_len, dtype=torch.float32)

        # angles: [max_seq_len, head_dim//2]
        angles = torch.outer(t, inv_freq)

        # 把角度复制一份，扩展到完整 head_dim 维度
        angles = torch.cat([angles, angles], dim=-1)  # [max_seq_len, head_dim]

        return angles.cos(), angles.sin()

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        对 Query 和 Key 施加旋转位置编码
        Args:
            q: [batch_size, seq_len, num_heads, head_dim]
            k: [batch_size, seq_len, num_heads, head_dim]
        Returns:
            q_rot, k_rot: 旋转后的 Query 和 Key
        """
        seq_len = q.shape[1]   # 当前实际序列长度（可小于 max_seq_len）

        # 取出当前序列长度对应的 cos / sin，并调整形状以支持广播
        # cos, sin: [1, seq_len, 1, head_dim]
        cos = self.cos[:seq_len].view(1, seq_len, 1, self.head_dim)
        sin = self.sin[:seq_len].view(1, seq_len, 1, self.head_dim)

        def rotate_half(x):
            """
            RoPE 的核心技巧：把向量 [x1, x2] 变成 [-x2, x1]
            这相当于把旋转矩阵 [cos, -sin; sin, cos] 简化为线性运算
            """
            x1, x2 = x.chunk(2, dim=-1)          # 切成两半 [..., head_dim//2]
            return torch.cat((-x2, x1), dim=-1)  # 得到 [-x2, x1]

        # 核心旋转公式（面试必讲！）
        # x' = x * cos + rotate_half(x) * sin
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

        return q_rot, k_rot
