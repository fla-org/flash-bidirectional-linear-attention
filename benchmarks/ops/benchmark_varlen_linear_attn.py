# -*- coding: utf-8 -*-

import torch
import triton
from torch.nn import functional as F

from flash_bla.ops.linear_attn.fused import linear_attention
from flash_bla.ops.linear_attn.varlen import varlen_linear_attention


def _bench(fn):
    # Finish Triton autotuning outside the timed region.
    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["max_seqlen"],
        x_vals=[256, 1024, 4096],
        line_arg="provider",
        line_vals=[
            "flash_padded_fwd",
            "varlen_fwd",
            "flash_padded_fwdbwd",
            "varlen_fwdbwd",
        ],
        line_names=[
            "flash padded fwd",
            "packed varlen fwd",
            "flash padded fwd+bwd",
            "packed varlen fwd+bwd",
        ],
        styles=[
            ("red", "-"),
            ("blue", "-"),
            ("red", "--"),
            ("blue", "--"),
        ],
        ylabel="Execution Time (ms)",
        plot_name="varlen-equal-B8-H16-D64",
        args={},
    )
)
def benchmark(max_seqlen, provider):
    device = "cuda"
    dtype = torch.bfloat16
    num_sequences, heads, dim = 8, 16, 64
    lengths = torch.full((num_sequences,), max_seqlen, dtype=torch.int32, device=device)
    total_tokens = int(lengths.sum().item())
    cu_seqlens = torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, dim=0, dtype=torch.int32)])

    q = (F.elu(torch.randn(total_tokens, heads, dim, device=device, dtype=dtype)) + 1).requires_grad_()
    k = (F.elu(torch.randn(total_tokens, heads, dim, device=device, dtype=dtype)) + 1).requires_grad_()
    v = torch.randn(total_tokens, heads, dim, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn_like(v)

    q_padded = (
        F.elu(torch.randn(num_sequences, heads, max_seqlen, dim, device=device, dtype=dtype)) + 1
    ).requires_grad_()
    k_padded = (F.elu(torch.randn_like(q_padded)) + 1).requires_grad_()
    v_padded = torch.randn_like(q_padded).requires_grad_()
    do_padded = torch.randn_like(v_padded)

    def flash_padded_forward():
        return linear_attention(q_padded, k_padded, v_padded)

    def varlen_forward():
        return varlen_linear_attention(q, k, v, cu_seqlens, max_seqlen)

    if provider == "flash_padded_fwd":
        return _bench(flash_padded_forward)
    if provider == "varlen_fwd":
        return _bench(varlen_forward)
    if provider == "flash_padded_fwdbwd":
        return _bench(lambda: flash_padded_forward().backward(do_padded, retain_graph=True))
    return _bench(lambda: varlen_forward().backward(do, retain_graph=True))


if __name__ == "__main__":
    benchmark.run(print_data=True)
