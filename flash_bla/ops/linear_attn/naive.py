import torch
import torch.nn as nn
import torch.nn.functional as F


def naive_linear_attn(q, k, v, scale = None):
    if scale is None:
        scale = k.shape[-2] ** -1.0
        
    z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1)
    s = k.transpose(-2, -1) @ (v * scale)
    o = q @ s / (z + 1e-6)
    
    return o


def naive_decoupled_la(q, k, q_rope, k_rope, v, scale = None):
    if scale is None:
        scale = k.shape[-2] ** -1.0
        
    z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1)
    s = k_rope.transpose(-2, -1) @ (v * scale)
    o = q_rope @ s / (z + 1e-6)
    
    return o