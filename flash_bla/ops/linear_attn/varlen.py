# Copyright (c) 2026 FLA contributors

from typing import Optional

import torch
import triton
import triton.language as tl

from flash_bla.utils import contiguous


_FWD_AUTOTUNE_CONFIGS = [
    triton.Config({"BL": block_l}, num_warps=num_warps)
    for block_l in (32, 64, 128)
    for num_warps in (4, 8)
]

_BWD_AUTOTUNE_CONFIGS = [
    triton.Config({"BL": block_l}, num_warps=num_warps)
    for block_l in (32, 64)
    for num_warps in (4, 8)
]


@triton.autotune(
    configs=_FWD_AUTOTUNE_CONFIGS,
    key=["MAX_SEQLEN_BUCKET", "DK", "DV"],
)
@triton.jit
def _varlen_fwd_state_kernel(
    K,
    V,
    CU_SEQLENS,
    S,
    KM,
    stride_kt,
    stride_kh,
    stride_kk,
    stride_vt,
    stride_vh,
    stride_vv,
    scale,
    MAX_SEQLEN_BUCKET: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BL: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_DEFAULT_SCALE: tl.constexpr,
):
    off_nh = tl.program_id(0)
    off_n = off_nh // H
    off_h = off_nh % H

    bos = tl.load(CU_SEQLENS + off_n)
    eos = tl.load(CU_SEQLENS + off_n + 1)
    length = eos - bos

    offs_l = tl.arange(0, BL)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)

    s = tl.zeros((BK, BV), dtype=tl.float32)
    km = tl.zeros((BK,), dtype=tl.float32)
    seq_scale = 1.0 / length if USE_DEFAULT_SCALE else scale

    start = 0
    while start < length:
        offs_t = bos + start + offs_l
        mask_l = start + offs_l < length
        k = tl.load(
            K + offs_t[:, None] * stride_kt + off_h * stride_kh + offs_k[None, :] * stride_kk,
            mask=mask_l[:, None] & (offs_k[None, :] < DK),
            other=0.0,
        )
        v = tl.load(
            V + offs_t[:, None] * stride_vt + off_h * stride_vh + offs_v[None, :] * stride_vv,
            mask=mask_l[:, None] & (offs_v[None, :] < DV),
            other=0.0,
        )
        s += tl.dot(tl.trans(k), (v * seq_scale).to(v.dtype), allow_tf32=False)
        km += tl.sum(k, axis=0)
        start += BL

    s_ptrs = S + off_nh * DK * DV + offs_k[:, None] * DV + offs_v[None, :]
    tl.store(s_ptrs, s, mask=(offs_k[:, None] < DK) & (offs_v[None, :] < DV))
    tl.store(KM + off_nh * DK + offs_k, km / length, mask=offs_k < DK)


@triton.autotune(
    configs=_FWD_AUTOTUNE_CONFIGS,
    key=["MAX_SEQLEN_BUCKET", "DK", "DV"],
)
@triton.jit
def _varlen_fwd_output_kernel(
    Q,
    CU_SEQLENS,
    S,
    KM,
    OUT,
    stride_qt,
    stride_qh,
    stride_qk,
    stride_ot,
    stride_oh,
    stride_ov,
    eps,
    MAX_SEQLEN_BUCKET: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BL: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    off_l = tl.program_id(0)
    off_nh = tl.program_id(1)
    off_n = off_nh // H
    off_h = off_nh % H

    bos = tl.load(CU_SEQLENS + off_n)
    eos = tl.load(CU_SEQLENS + off_n + 1)
    if off_l * BL < eos - bos:
        offs_t = bos + off_l * BL + tl.arange(0, BL)
        offs_k = tl.arange(0, BK)
        offs_v = tl.arange(0, BV)
        mask_l = offs_t < eos

        q = tl.load(
            Q + offs_t[:, None] * stride_qt + off_h * stride_qh + offs_k[None, :] * stride_qk,
            mask=mask_l[:, None] & (offs_k[None, :] < DK),
            other=0.0,
        )
        s = tl.load(
            S + off_nh * DK * DV + offs_k[:, None] * DV + offs_v[None, :],
            mask=(offs_k[:, None] < DK) & (offs_v[None, :] < DV),
            other=0.0,
        )
        km = tl.load(KM + off_nh * DK + offs_k, mask=offs_k < DK, other=0.0)

        z = tl.sum(q * km[None, :], axis=1) + eps
        o = tl.dot(q, s, allow_tf32=False) / z[:, None]

        tl.store(
            OUT + offs_t[:, None] * stride_ot + off_h * stride_oh + offs_v[None, :] * stride_ov,
            o,
            mask=mask_l[:, None] & (offs_v[None, :] < DV),
        )


@triton.autotune(
    configs=_BWD_AUTOTUNE_CONFIGS,
    key=["MAX_SEQLEN_BUCKET", "DK", "DV"],
)
@triton.jit
def _varlen_bwd_state_kernel(
    Q,
    OUT,
    DO,
    CU_SEQLENS,
    KM,
    DS,
    DKM,
    stride_qt,
    stride_qh,
    stride_qk,
    stride_ot,
    stride_oh,
    stride_ov,
    eps,
    MAX_SEQLEN_BUCKET: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BL: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    off_nh = tl.program_id(0)
    off_n = off_nh // H
    off_h = off_nh % H

    bos = tl.load(CU_SEQLENS + off_n)
    eos = tl.load(CU_SEQLENS + off_n + 1)
    length = eos - bos

    offs_l = tl.arange(0, BL)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    km = tl.load(KM + off_nh * DK + offs_k, mask=offs_k < DK, other=0.0)

    ds = tl.zeros((BK, BV), dtype=tl.float32)
    dkm = tl.zeros((BK,), dtype=tl.float32)

    start = 0
    while start < length:
        offs_t = bos + start + offs_l
        mask_l = start + offs_l < length
        q = tl.load(
            Q + offs_t[:, None] * stride_qt + off_h * stride_qh + offs_k[None, :] * stride_qk,
            mask=mask_l[:, None] & (offs_k[None, :] < DK),
            other=0.0,
        )
        o = tl.load(
            OUT + offs_t[:, None] * stride_ot + off_h * stride_oh + offs_v[None, :] * stride_ov,
            mask=mask_l[:, None] & (offs_v[None, :] < DV),
            other=0.0,
        )
        do = tl.load(
            DO + offs_t[:, None] * stride_ot + off_h * stride_oh + offs_v[None, :] * stride_ov,
            mask=mask_l[:, None] & (offs_v[None, :] < DV),
            other=0.0,
        )

        z = tl.sum(q * km[None, :], axis=1) + eps
        ds += tl.dot(tl.trans((q / z[:, None]).to(do.dtype)), do, allow_tf32=False)
        dz = -tl.sum(o * do, axis=1) / z
        dkm += tl.sum(dz[:, None] * q, axis=0)
        start += BL

    ds_ptrs = DS + off_nh * DK * DV + offs_k[:, None] * DV + offs_v[None, :]
    tl.store(ds_ptrs, ds, mask=(offs_k[:, None] < DK) & (offs_v[None, :] < DV))
    tl.store(DKM + off_nh * DK + offs_k, dkm, mask=offs_k < DK)


@triton.autotune(
    configs=_BWD_AUTOTUNE_CONFIGS,
    key=["MAX_SEQLEN_BUCKET", "DK", "DV"],
)
@triton.jit
def _varlen_bwd_input_kernel(
    Q,
    K,
    V,
    OUT,
    DO,
    CU_SEQLENS,
    S,
    KM,
    DS,
    DKM,
    DQ,
    DK_OUT,
    DV_OUT,
    stride_qt,
    stride_qh,
    stride_qk,
    stride_vt,
    stride_vh,
    stride_vv,
    eps,
    scale,
    MAX_SEQLEN_BUCKET: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BL: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_DEFAULT_SCALE: tl.constexpr,
):
    off_l = tl.program_id(0)
    off_nh = tl.program_id(1)
    off_n = off_nh // H
    off_h = off_nh % H

    bos = tl.load(CU_SEQLENS + off_n)
    eos = tl.load(CU_SEQLENS + off_n + 1)
    length = eos - bos
    if off_l * BL < length:
        offs_t = bos + off_l * BL + tl.arange(0, BL)
        offs_k = tl.arange(0, BK)
        offs_v = tl.arange(0, BV)
        mask_l = offs_t < eos
        seq_scale = 1.0 / length if USE_DEFAULT_SCALE else scale

        q_ptrs = Q + offs_t[:, None] * stride_qt + off_h * stride_qh + offs_k[None, :] * stride_qk
        v_ptrs = V + offs_t[:, None] * stride_vt + off_h * stride_vh + offs_v[None, :] * stride_vv
        q = tl.load(q_ptrs, mask=mask_l[:, None] & (offs_k[None, :] < DK), other=0.0)
        k = tl.load(
            K + offs_t[:, None] * stride_qt + off_h * stride_qh + offs_k[None, :] * stride_qk,
            mask=mask_l[:, None] & (offs_k[None, :] < DK),
            other=0.0,
        )
        v = tl.load(v_ptrs, mask=mask_l[:, None] & (offs_v[None, :] < DV), other=0.0)
        o = tl.load(
            OUT + offs_t[:, None] * stride_vt + off_h * stride_vh + offs_v[None, :] * stride_vv,
            mask=mask_l[:, None] & (offs_v[None, :] < DV),
            other=0.0,
        )
        do = tl.load(
            DO + offs_t[:, None] * stride_vt + off_h * stride_vh + offs_v[None, :] * stride_vv,
            mask=mask_l[:, None] & (offs_v[None, :] < DV),
            other=0.0,
        )
        s = tl.load(
            S + off_nh * DK * DV + offs_k[:, None] * DV + offs_v[None, :],
            mask=(offs_k[:, None] < DK) & (offs_v[None, :] < DV),
            other=0.0,
        )
        km = tl.load(KM + off_nh * DK + offs_k, mask=offs_k < DK, other=0.0)
        ds = tl.load(
            DS + off_nh * DK * DV + offs_k[:, None] * DV + offs_v[None, :],
            mask=(offs_k[:, None] < DK) & (offs_v[None, :] < DV),
            other=0.0,
        )
        dkm = tl.load(DKM + off_nh * DK + offs_k, mask=offs_k < DK, other=0.0)

        z = tl.sum(q * km[None, :], axis=1) + eps
        dz = -tl.sum(o * do, axis=1) / z
        dq = tl.dot(do, tl.trans(s), allow_tf32=False) / z[:, None] + dz[:, None] * km[None, :]
        dk = tl.dot((v * seq_scale).to(v.dtype), tl.trans(ds), allow_tf32=False) + dkm[None, :] / length
        dv = tl.dot(k, ds, allow_tf32=False) * seq_scale

        mask_qk = mask_l[:, None] & (offs_k[None, :] < DK)
        mask_v = mask_l[:, None] & (offs_v[None, :] < DV)
        tl.store(
            DQ + offs_t[:, None] * stride_qt + off_h * stride_qh + offs_k[None, :] * stride_qk,
            dq,
            mask=mask_qk,
        )
        tl.store(
            DK_OUT + offs_t[:, None] * stride_qt + off_h * stride_qh + offs_k[None, :] * stride_qk,
            dk,
            mask=mask_qk,
        )
        tl.store(
            DV_OUT + offs_t[:, None] * stride_vt + off_h * stride_vh + offs_v[None, :] * stride_vv,
            dv,
            mask=mask_v,
        )


class VarlenLinearAttnFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, scale, eps):
        total_tokens, heads, key_dim = q.shape
        value_dim = v.shape[-1]
        num_sequences = cu_seqlens.numel() - 1
        block_k = max(16, triton.next_power_of_2(key_dim))
        block_v = max(16, triton.next_power_of_2(value_dim))
        max_seqlen_bucket = triton.next_power_of_2(max_seqlen)
        use_default_scale = scale is None
        kernel_scale = 0.0 if scale is None else scale

        s = torch.empty(
            num_sequences,
            heads,
            key_dim,
            value_dim,
            dtype=q.dtype,
            device=q.device,
        )
        km = torch.empty(
            num_sequences,
            heads,
            key_dim,
            dtype=q.dtype,
            device=q.device,
        )
        _varlen_fwd_state_kernel[(num_sequences * heads,)](
            k,
            v,
            cu_seqlens,
            s,
            km,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            kernel_scale,
            MAX_SEQLEN_BUCKET=max_seqlen_bucket,
            H=heads,
            DK=key_dim,
            DV=value_dim,
            BK=block_k,
            BV=block_v,
            USE_DEFAULT_SCALE=use_default_scale,
        )

        o = torch.empty(
            total_tokens,
            heads,
            value_dim,
            dtype=q.dtype,
            device=q.device,
        )
        grid = lambda meta: (
            triton.cdiv(max_seqlen, meta["BL"]),
            num_sequences * heads,
        )
        _varlen_fwd_output_kernel[grid](
            q,
            cu_seqlens,
            s,
            km,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            eps,
            MAX_SEQLEN_BUCKET=max_seqlen_bucket,
            H=heads,
            DK=key_dim,
            DV=value_dim,
            BK=block_k,
            BV=block_v,
        )

        ctx.save_for_backward(q, k, v, cu_seqlens, o, s, km)
        ctx.max_seqlen = max_seqlen
        ctx.scale = scale
        ctx.eps = eps
        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, v, cu_seqlens, o, s, km = ctx.saved_tensors
        _, heads, key_dim = q.shape
        value_dim = v.shape[-1]
        num_sequences = cu_seqlens.numel() - 1
        block_k = max(16, triton.next_power_of_2(key_dim))
        block_v = max(16, triton.next_power_of_2(value_dim))
        max_seqlen_bucket = triton.next_power_of_2(ctx.max_seqlen)
        use_default_scale = ctx.scale is None
        kernel_scale = 0.0 if ctx.scale is None else ctx.scale

        ds = torch.empty_like(s)
        dkm = torch.empty(
            num_sequences,
            heads,
            key_dim,
            dtype=torch.float32,
            device=q.device,
        )
        _varlen_bwd_state_kernel[(num_sequences * heads,)](
            q,
            o,
            do,
            cu_seqlens,
            km,
            ds,
            dkm,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            ctx.eps,
            MAX_SEQLEN_BUCKET=max_seqlen_bucket,
            H=heads,
            DK=key_dim,
            DV=value_dim,
            BK=block_k,
            BV=block_v,
        )

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        grid = lambda meta: (
            triton.cdiv(ctx.max_seqlen, meta["BL"]),
            num_sequences * heads,
        )
        _varlen_bwd_input_kernel[grid](
            q,
            k,
            v,
            o,
            do,
            cu_seqlens,
            s,
            km,
            ds,
            dkm,
            dq,
            dk,
            dv,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            ctx.eps,
            kernel_scale,
            MAX_SEQLEN_BUCKET=max_seqlen_bucket,
            H=heads,
            DK=key_dim,
            DV=value_dim,
            BK=block_k,
            BV=block_v,
            USE_DEFAULT_SCALE=use_default_scale,
        )
        return dq, dk, dv, None, None, None, None


def varlen_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: Optional[float | int] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute bidirectional linear attention over packed variable-length sequences.

    Args:
        q (torch.Tensor):
            Packed queries of shape ``(total_tokens, heads, key_dim)``.
        k (torch.Tensor):
            Packed keys of shape ``(total_tokens, heads, key_dim)``.
        v (torch.Tensor):
            Packed values of shape ``(total_tokens, heads, value_dim)``.
        cu_seqlens (torch.Tensor):
            Cumulative sequence lengths of shape ``(num_sequences + 1,)``.
            Entries must start at zero, end at ``total_tokens``, and be
            strictly increasing. The tensor must have dtype ``torch.int32``.
        max_seqlen (int):
            An upper bound on the sequence lengths represented by
            ``cu_seqlens``.
        scale (Optional[float]):
            Scale applied to values when accumulating the key-value state.
            If omitted, each sequence uses the reciprocal of its own length.
        eps (float):
            A small constant added to the denominator.

    Returns:
        torch.Tensor:
            Packed outputs of shape ``(total_tokens, heads, value_dim)``.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"q, k, and v must have shape (total_tokens, heads, dim), got q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have identical shapes, got {q.shape} and {k.shape}")
    if q.shape[:2] != v.shape[:2]:
        raise ValueError("q, k, and v must have matching token and head dimensions")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must have the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if not q.is_cuda:
        raise ValueError("q, k, and v must be CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("q, k, and v must have dtype float16, bfloat16, or float32")
    if not (0 < q.shape[-1] <= 128 and 0 < v.shape[-1] <= 128):
        raise ValueError("key_dim and value_dim must be between 1 and 128")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must have shape (num_sequences + 1,)")
    if cu_seqlens.device != q.device or cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be an int32 tensor on the same device as q")
    if not isinstance(max_seqlen, int) or max_seqlen <= 0:
        raise ValueError("max_seqlen must be a positive integer")

    scale = None if scale is None else float(scale)
    return VarlenLinearAttnFunction.apply(q, k, v, cu_seqlens, max_seqlen, scale, eps)
