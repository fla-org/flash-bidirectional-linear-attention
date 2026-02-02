# -*- coding: utf-8 -*-

import torch
import triton
from torch.nn import functional as F

from flash_bla.ops.linear_attn.fused import linear_attention
from flash_bla.ops.linear_attn.naive import naive_linear_attn


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 3)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['torch_fwd', 'triton_fwd', 'fa_fwd', 'torch_bwd', 'triton_bwd', 'fa_bwd'],
        # label name for the lines
        line_names=['torch_fwd', 'triton_fwd', 'fa_fwd', 'torch_bwd', 'triton_bwd', 'fa_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':'), ('cyan', ':'), ('cyan', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="B8-H16-D64",
        args={},
    )
)
def benchmark(T, provider):
    device = 'cuda'
    dtype = torch.bfloat16
    requires_grad = True
    B, H, D = 8, 16, 64

    q = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad, dtype=dtype)

    do = torch.ones_like(q, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == 'torch_fwd':
        # pass
        results = triton.testing.do_bench(lambda: naive_linear_attn(q, k, v), quantiles=quantiles)
    elif provider == 'triton_fwd':
        # pass
        results = triton.testing.do_bench(lambda: linear_attention(q, k, v), quantiles=quantiles)
    elif provider == 'fa_fwd':
        # pass
        results = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False), quantiles=quantiles)
    elif provider == 'torch_bwd':
        # pass
        results = triton.testing.do_bench(lambda: naive_linear_attn(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'triton_bwd':
        # pass
        results = triton.testing.do_bench(lambda: linear_attention(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'fa_bwd':
        # pass
        results = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False).backward(do), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)