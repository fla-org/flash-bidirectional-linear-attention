# https://github.com/LeapLabTHU/FLatten-Transformer/blob/master/models/flatten_pvt.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from einops import rearrange

from flash_bla.ops.linear_attn.fused import linear_attention


class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, focusing_factor, kernel_size))

    def forward(self, x, H, W, mode='triton'):
        B, N, C = x.shape
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n = k.shape[1]

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        # Important! qkv must be contiguous.
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        if mode == 'torch':
            z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
            kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
            x = q @ kv * z
        elif mode == 'triton':
            x = linear_attention(q, k, v)
        else:
            raise NotImplementedError

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)
        x = x.transpose(1, 2).reshape(B, N, C)
        v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
    
if __name__ == "__main__":
    B, H, L, D = 4, 16, 256, 64
    dtype = torch.float32

    x = torch.randn((B, L, H*D), dtype=dtype, device="cuda", requires_grad=True)
    do = torch.randn_like(x).cuda()
    
    model = FocusedLinearAttention(dim=H*D, num_patches=1, num_heads=H).cuda()
    
    # naive
    ref = model(x, H=int(L**0.5), W=int(L**0.5), mode='torch')
    
    x.retain_grad()
    ref.backward(do, retain_graph=True)
    ref_dx, x.grad = x.grad.clone(), None
    
    # triton
    tri = model(x,  H=int(L**0.5), W=int(L**0.5), mode='triton')
    
    x.retain_grad()
    tri.backward(do, retain_graph=True)
    tri_dx, x.grad = x.grad.clone(), None
    
    assert torch.allclose(ref, tri, rtol=0, atol=1e-4)
    assert torch.allclose(ref_dx, tri_dx, rtol=0, atol=1e-4)
    print("Triton and Torch match")