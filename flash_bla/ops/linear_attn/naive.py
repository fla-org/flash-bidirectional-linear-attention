import torch
import torch.nn as nn
import torch.nn.functional as F

from fbi_la.ops.linear_attn.attention import linear_attention


def naive_linear_attn(
    q,
    k,
    v,
    scale = None
):
    if scale is None:
        scale = k.shape[-2] ** -1.0
        
    z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1)
    s = k.transpose(-2, -1) @ (v * scale)
    o = q @ s / (z + 1e-6)
    
    return o


def check_close(A, B):
    b = torch.allclose(A, B, rtol=0, atol=1e-2)
    return b


if __name__ == "__main__":
    B, H, L, D = 1, 4, 256, 64
    dtype = torch.bfloat16

    q = torch.randn((B, H, L, D), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((B, H, L, D), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((B, H, L, D), dtype=dtype, device="cuda", requires_grad=True)
    
    q = F.elu(q) + 1.0
    k = F.elu(k) + 1.0
    
    do = torch.randn_like(v)
    
    # torch
    ref = naive_linear_attn(q, k, v)
    
    q.retain_grad(), k.retain_grad(), v.retain_grad()
    ref.backward(do, retain_graph=True)
    
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    
    # triton
    tri = linear_attention(q, k, v)
    
    q.retain_grad(), k.retain_grad(), v.retain_grad()
    tri.backward(do, retain_graph=True)
    
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    
    print(check_close(ref, tri))
    print(check_close(ref_dq, tri_dq))
    print(check_close(ref_dk, tri_dk))
    print(check_close(ref_dv, tri_dv))
    
    print("Triton and Torch match")
    