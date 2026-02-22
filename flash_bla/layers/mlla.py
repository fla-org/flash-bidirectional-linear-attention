# https://github.com/LeapLabTHU/MLLA/blob/master/models/mlla.py

import torch
import torch.nn as nn

from flash_bla.ops.linear_attn.decoupled import decoupled_la
from flash_bla.utils import assert_close


class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class MambaLikeLinearAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

    def forward(self, x, mode='triton'):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

        if mode == 'torch':
            z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
            kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
            x = q_rope @ kv
            x = x * z
        elif mode == 'triton':
            x = decoupled_la(q, k, q_rope, k_rope, v, scale=n ** -1.0)
        else:
            raise NotImplementedError
        
        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'
    
    
if __name__ == "__main__":
    B, H, L, D = 4, 16, 256, 32
    dtype = torch.float32

    x = torch.randn((B, L, H*D), dtype=dtype, device="cuda", requires_grad=True)
    do = torch.randn_like(x).cuda()
    
    model = MambaLikeLinearAttention(dim=H*D, input_resolution=(16, 16), num_heads=H).cuda()
    
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
    
    assert_close("Output", ref, tri, 0.06)
    assert_close("Grad", ref_dx, tri_dx, 0.06)
    print("Triton and Torch match")