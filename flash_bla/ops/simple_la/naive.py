import torch
import torch.nn as nn
import torch.nn.functional as F

from fbi_la.ops.simple_la.attention import simple_la


def naive_simple_la(
    q,
    k,
    v,
    scale = None
):
    if scale is None:
        scale = q.shape[-1] ** -0.5
        
    s = k.transpose(-2, -1) @ (v * scale)
    o = q @ s
    
    return o


def check_close(A, B):
    b = torch.allclose(A, B, rtol=0, atol=1e-2)
    return b


if __name__ == "__main__":
    B, H, L, D = 2, 4, 256, 32
    dtype = torch.float32

    q = torch.randn((B, H, L, D), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((B, H, L, D), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((B, H, L, D), dtype=dtype, device="cuda", requires_grad=True)
    
    q = F.elu(q) + 1.0
    k = F.elu(k) + 1.0
    
    do = torch.randn_like(v)
    ds = torch.randn_like(k.transpose(-2, -1) @ v)
    
    # naive
    ref = naive_simple_la(q, k, v)
    
    q.retain_grad(), k.retain_grad(), v.retain_grad()
    ref.backward(do, retain_graph=True)
    
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    
    # triton
    # tri = simple_la(q, k, v)
    tri, s = simple_la(q, k, v, retuen_states=True)
    
    q.retain_grad(), k.retain_grad(), v.retain_grad()
    # tri.backward(do, retain_graph=True)
    ((tri * do).sum() + (ds * s).sum()).backward()
    
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    
    assert check_close(ref, tri)
    assert check_close(ref_dq, tri_dq)
    assert check_close(ref_dk, tri_dk)
    assert check_close(ref_dv, tri_dv)
    
    print("Triton and Torch match")