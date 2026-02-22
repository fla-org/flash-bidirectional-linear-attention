# -*- coding: utf-8 -*-

import torch
import triton
from torch.nn import functional as F

from flash_bla.ops.linear_attn.decoupled import decoupled_la
from flash_bla.ops.linear_attn.naive import naive_decoupled_la


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[1024, 4096, 16384, 32768],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['torch_la_fwd', 'flash_bla_fwd', 'torch_sdpa_fwd', 'torch_la_bwd', 'flash_bla_bwd', 'torch_sdpa_bwd'],
        # label name for the lines
        line_names=['torch_la_fwd', 'flash_bla_fwd', 'torch_sdpa_fwd', 'torch_la_bwd', 'flash_bla_bwd', 'torch_sdpa_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-.'), ('green', '--'), ('blue', '--'), ('red', '--')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="B2-H16-D64",
        args={},
    )
)
def benchmark(T, provider):
    device = 'cuda'
    dtype = torch.bfloat16
    B, H, D = 2, 16, 64

    q = torch.randn(B, H, T, D, device=device, requires_grad=True, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, requires_grad=True, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, requires_grad=True, dtype=dtype)

    q = F.elu(q) + 1.0
    k = F.elu(k) + 1.0

    q_rope = torch.randn(B, H, T, D, device=device, requires_grad=True, dtype=dtype)
    k_rope = torch.randn(B, H, T, D, device=device, requires_grad=True, dtype=dtype)

    q_rope = F.elu(q_rope) + 1.0
    k_rope = F.elu(k_rope) + 1.0

    do = torch.ones_like(q, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == 'torch_la_fwd':
        results = triton.testing.do_bench(lambda: naive_decoupled_la(q, k, q_rope, k_rope, v), quantiles=quantiles)
    elif provider == 'flash_bla_fwd':
        results = triton.testing.do_bench(lambda: decoupled_la(q, k, q_rope, k_rope, v), quantiles=quantiles)
    elif provider == 'torch_sdpa_fwd':
        results = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False), quantiles=quantiles)
    elif provider == 'torch_la_bwd':
        results = triton.testing.do_bench(lambda: naive_decoupled_la(q, k, q_rope, k_rope, v).backward(do, retain_graph=True), quantiles=quantiles)
    elif provider == 'flash_bla_bwd':
        results = triton.testing.do_bench(lambda: decoupled_la(q, k, q_rope, k_rope, v).backward(do, retain_graph=True), quantiles=quantiles)
    elif provider == 'torch_sdpa_bwd':
        results = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False).backward(do, retain_graph=True), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)