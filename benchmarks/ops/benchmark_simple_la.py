# -*- coding: utf-8 -*-

import torch
import triton
from torch.nn import functional as F

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/public/liguoqi/ssl/wds/flash-bidirectional-linear-attention/')))


from fbi_la.ops.simple_la.attention import simple_la
from fbi_la.ops.simple_la.naive import naive_simple_la


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 4)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['torch_fwd', 'triton_fwd', 'torch_bwd', 'triton_bwd'],
        # label name for the lines
        line_names=['torch_fwd', 'triton_fwd', 'torch_bwd', 'triton_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':')],
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
        results = triton.testing.do_bench(lambda: naive_simple_la(q, k, v), quantiles=quantiles)
    elif provider == 'triton_fwd':
        # pass
        results = triton.testing.do_bench(lambda: simple_la(q, k, v), quantiles=quantiles)
    elif provider == 'torch_bwd':
        # pass
        results = triton.testing.do_bench(lambda: naive_simple_la(q, k, v).backward(do), quantiles=quantiles)
    elif provider == 'triton_bwd':
        # pass
        results = triton.testing.do_bench(lambda: simple_la(q, k, v).backward(do), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)