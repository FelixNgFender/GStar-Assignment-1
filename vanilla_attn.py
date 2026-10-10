import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "head_dim not divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        """x: (B, T, C)"""
        B, T, C = x.shape
        x_qkv = self.qkv(x)  # (B, T, 3*C)
        x_q, x_k, x_v = torch.split(x_qkv, self.embed_dim, dim=-1)  # (B, T, C) each

        def reshape_for_heads(x: torch.Tensor) -> torch.Tensor:
            """x: (B, T, C)"""
            return x.view(B, T, self.num_heads, self.head_dim).transpose(
                -2, -3
            )  # (B, nh, T, hd)

        x_q = reshape_for_heads(x_q)
        x_k = reshape_for_heads(x_k)
        x_v = reshape_for_heads(x_v)

        scores = (x_q @ x_k.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )  # (B, nh, T, T)
        if is_causal:
            mask = torch.tril(torch.ones(T, T).view(1, 1, T, T))
            scores = torch.masked_fill(scores, mask == 0, -torch.inf)
        scores = scores.softmax(dim=-1)  # (B, nh, T, T)
        y = scores @ x_v  # (B, nh, T, hd)
        y = y.transpose(-2, -3).contiguous().view(B, T, C)
        return self.proj(y)


def standard_multi_head_attention(Q, K, V, mask=None):
    """Q, K, V (B, T, C), mask (B, 1, 1, T)"""
    B, T, C = Q.shape
    num_heads = 8
    head_dim = C // num_heads

    def reshape_for_heads(x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C)"""
        return x.view(B, T, num_heads, head_dim).transpose(-2, -3)  # (B, nh, T, hd)

    Q_heads = reshape_for_heads(Q)
    K_heads = reshape_for_heads(K)
    V_heads = reshape_for_heads(V)

    scores = Q_heads @ K_heads.transpose(-1, -2)  # PERF:
    scores = scores / math.sqrt(head_dim)  # (B, nh, T, T)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -torch.inf)

    attn_weights = F.softmax(scores, dim=-1)  # PERF:
    output = attn_weights @ V_heads
    return output.transpose(-2, -3).contiguous().view(B, T, C), attn_weights


def main() -> None:
    B, T, C = 4, 3, 16
    nh = 4

    x = torch.randn((B, T, C))
    attn = CausalSelfAttention(C, nh)
    assert attn(x).shape == (B, T, C)
    print(x)


if __name__ == "__main__":
    main()
