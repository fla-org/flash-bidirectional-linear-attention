# Copyright (c) 2026 FLA contributors

import pytest
import torch
import torch.nn.functional as F

from flash_bla.ops.linear_attn.naive import naive_varlen_linear_attn
from flash_bla.ops.linear_attn.varlen import varlen_linear_attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale", [None, 0.02])
@pytest.mark.parametrize("heads", [1, 8])
@pytest.mark.parametrize(
    ("key_dim", "value_dim"),
    [(8, 8), (17, 13), (64, 48), (128, 128)],
)
def test_varlen_linear_attention(dtype, scale, heads, key_dim, value_dim):
    torch.manual_seed(42)
    lengths = torch.tensor([8, 17, 32], dtype=torch.int32, device="cuda")
    total_tokens = int(lengths.sum().item())
    cu_seqlens = torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, dim=0, dtype=torch.int32)])

    q = (F.elu(torch.randn(total_tokens, heads, key_dim, dtype=dtype, device="cuda")) + 1).requires_grad_()
    k = (F.elu(torch.randn(total_tokens, heads, key_dim, dtype=dtype, device="cuda")) + 1).requires_grad_()
    v = torch.randn(total_tokens, heads, value_dim, dtype=dtype, device="cuda", requires_grad=True)
    do = torch.randn_like(v)

    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()

    out = varlen_linear_attention(q, k, v, cu_seqlens, int(lengths.max()), scale=scale)
    out_ref = naive_varlen_linear_attn(q_ref, k_ref, v_ref, cu_seqlens, scale=scale)
    out.backward(do)
    out_ref.backward(do)

    atol, rtol = (1e-5, 1e-4) if dtype == torch.float32 else (2e-2, 8e-2)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_varlen_linear_attention_oversized_max_seqlen():
    lengths = torch.tensor([7, 11], dtype=torch.int32, device="cuda")
    cu_seqlens = torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, dim=0, dtype=torch.int32)])
    q = torch.rand(int(lengths.sum().item()), 2, 16, dtype=torch.float32, device="cuda")
    k = torch.rand_like(q)
    v = torch.randn_like(q)

    expected = varlen_linear_attention(q, k, v, cu_seqlens, int(lengths.max()))
    oversized = varlen_linear_attention(q, k, v, cu_seqlens, q.shape[0])
    torch.testing.assert_close(oversized, expected)
