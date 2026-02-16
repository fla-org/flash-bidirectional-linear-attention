import torch
import torch.nn as nn

from flash_bla.ops.simple_la.fused import simple_la
from flash_bla.layers.mlla import RoPE


class RALALinearAttention(nn.Module):
    r""" Reweighted Attention Linear Attention (RALA) from RAVLT.

    This layer implements a gated linear attention mechanism with
    efficiency-based key reweighting, RoPE, and local position encoding (LePE).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution (H, W).
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value, gate. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)

        self.qkvo = nn.Linear(dim, dim * 4, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim // num_heads))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mode='triton'):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = self.head_dim

        qkvo = self.qkvo(x)
        q, k, v, o = qkvo.reshape(b, n, 4, c).unbind(2)

        # LePE on v
        lepe = self.lepe(v.reshape(b, h, w, c).permute(0, 3, 1, 2))

        # Feature map
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        # Reshape to multi-head: (b, num_heads, n, head_dim)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

        # Efficiency factor: reweight keys
        q_mean = q.mean(dim=-2, keepdim=True)  # (b, heads, 1, d)
        eff = self.scale * q_mean @ k.transpose(-1, -2)  # (b, heads, 1, n)
        eff = torch.softmax(eff, dim=-1).transpose(-1, -2)  # (b, heads, n, 1)
        k = k * eff * n

        # RoPE in multi-head space: (b, num_heads, h, w, head_dim)
        q_rope = self.rope(q.reshape(b, num_heads, h, w, head_dim)).reshape(b, num_heads, n, head_dim).contiguous()
        k_rope = self.rope(k.reshape(b, num_heads, h, w, head_dim)).reshape(b, num_heads, n, head_dim).contiguous()

        # Normalization factor (computed with non-RoPE'd q, k)
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)

        if mode == 'torch':
            kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
            x = q_rope @ kv
        elif mode == 'triton':
            x = simple_la(q_rope, k_rope, v, scale=n ** -1.0)
        else:
            raise NotImplementedError

        x = x * z

        x = x.transpose(1, 2).reshape(b, n, c)
        x = x + lepe.permute(0, 2, 3, 1).reshape(b, n, c)

        # Output gating
        x = x * o
        x = self.proj(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'


if __name__ == "__main__":
    B, H, L, D = 4, 16, 256, 32
    dtype = torch.float32

    x = torch.randn((B, L, H*D), dtype=dtype, device="cuda", requires_grad=True)
    do = torch.randn_like(x).cuda()

    model = RALALinearAttention(dim=H*D, input_resolution=(16, 16), num_heads=H).cuda()

    # naive
    ref = model(x, mode='torch')

    x.retain_grad()
    ref.backward(do, retain_graph=True)
    ref_dx, x.grad = x.grad.clone(), None

    # triton
    tri = model(x, mode='triton')

    x.retain_grad()
    tri.backward(do, retain_graph=True)
    tri_dx, x.grad = x.grad.clone(), None

    assert torch.allclose(ref, tri, rtol=0, atol=1e-3)
    assert torch.allclose(ref_dx, tri_dx, rtol=0, atol=1e-3)
    print("Triton and Torch match")
