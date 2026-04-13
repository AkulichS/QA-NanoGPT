import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len = None
        self._cos = None
        self._sin = None

    def forward(self, seq_len, dtype):
        device = self.inv_freq.device
        if (
            seq_len != self._seq_len
            or self._cos.device != device
            or self._cos.dtype != dtype
        ):
            positions = torch.arange(seq_len, device=device).float()
            theta = torch.outer(positions, self.inv_freq)
            self._cos = theta.cos().to(dtype).unsqueeze(0).unsqueeze(0)
            self._sin = theta.sin().to(dtype).unsqueeze(0).unsqueeze(0)
            self._seq_len = seq_len

        return self._cos, self._sin
    

def apply_rope(q, k, cos, sin):
    # q, k: (B, H, T, D)
    # cos, sin: (1, 1, T, D/2)

    # Чётные и нечётные координаты (q1, q2) -> (x0​,x1​), (x2​,x3​), ...
    q1 = q[..., ::2]   # q1 = (B, H, T, D/2),  D/2 -> [ x0, x2, x4, ... ] 
    q2 = q[..., 1::2]  # q1 = (B, H, T, D/2),  D/2 -> [ x1, x3, x5, ... ]
    k1 = k[..., ::2]   # k1 = (B, H, T, D/2),  D/2 -> [ x0, x2, x4, ... ] 
    k2 = k[..., 1::2]  # k1 = (B, H, T, D/2),  D/2 -> [ x1, x3, x5, ... ]

    q_rot = torch.empty_like(q)
    k_rot = torch.empty_like(k)

    # Rotation q and k, broadcast (1,1,T,D/2)→(B,H,T,D/2)
    q_rot[..., ::2] = q1 * cos - q2 * sin
    q_rot[..., 1::2] = q1 * sin + q2 * cos

    k_rot[..., ::2] = k1 * cos - k2 * sin
    k_rot[..., 1::2] = k1 * sin + k2 * cos

    return q_rot, k_rot