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


def naive_varlen_linear_attn(q, k, v, cu_seqlens, scale=None, eps=1e-6):
    outputs = []
    for i in range(cu_seqlens.numel() - 1):
        bos = cu_seqlens[i].item()
        eos = cu_seqlens[i + 1].item()
        length = eos - bos
        seq_scale = length ** -1.0 if scale is None else scale

        q_i = q[bos:eos].transpose(0, 1)
        k_i = k[bos:eos].transpose(0, 1)
        v_i = v[bos:eos].transpose(0, 1)
        z = (q_i * k_i.mean(dim=1)[:, None, :]).sum(dim=-1, keepdim=True)
        s = k_i.transpose(-2, -1) @ (v_i * seq_scale)
        outputs.append(((q_i @ s) / (z + eps)).transpose(0, 1))

    return torch.cat(outputs, dim=0)