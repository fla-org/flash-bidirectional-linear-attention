# Copyright (c) 2024 - 2026 Haopeng Li

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

from flash_bla.utils import contiguous


@triton.autotune(
    configs=[
        triton.Config({'BL': BL, 'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BL in [32, 64, 128]
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['L']
)
@triton.jit
def fused_fwd_kernel_s_km(
    K, K_ROPE, V, S, KM,
    stride_qk_bh, stride_qk_l, stride_qk_d,
    stride_vo_bh, stride_vo_l, stride_vo_d,
    stride_s_bh, stride_s_dk, stride_s_dv,
    stride_km_bh,
    scale,
    L: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BL: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    The fused kernel computes both S = K_ROPE^T @ V and the mean of K (along the L dimension).
    """
    start_v, start_k, off_bs_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    K_block_ptr = tl.make_block_ptr(
        base=K + off_bs_head * stride_qk_bh,
        shape=(DK, L),
        strides=(stride_qk_d, stride_qk_l),
        offsets=(start_k * BK, 0),
        block_shape=(BK, BL),
        order=(0, 1),
    )
    K_ROPE_block_ptr = tl.make_block_ptr(
        base=K_ROPE + off_bs_head * stride_qk_bh,
        shape=(DK, L),
        strides=(stride_qk_d, stride_qk_l),
        offsets=(start_k * BK, 0),
        block_shape=(BK, BL),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + off_bs_head * stride_vo_bh,
        shape=(L, DV),
        strides=(stride_vo_l, stride_vo_d),
        offsets=(0, start_v * BV),
        block_shape=(BL, BV),
        order=(1, 0),
    )
    
    s = tl.zeros([BK, BV], dtype=tl.float32)
    km = tl.zeros([BK], dtype=tl.float32)
    
    for _ in range(0, L, BL):
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        k_rope = tl.load(K_ROPE_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))
    
        v = (v * scale).to(v.dtype)
        # S is calculated using K_ROPE
        s += tl.dot(k_rope, v, allow_tf32=False)
        # KM is calculated using normal K
        km += tl.sum(k, axis=1) / L
        
        K_block_ptr = tl.advance(K_block_ptr, (0, BL))
        K_ROPE_block_ptr = tl.advance(K_ROPE_block_ptr, (0, BL))
        V_block_ptr = tl.advance(V_block_ptr, (BL, 0))

    S_block_ptr = tl.make_block_ptr(
        base=S + off_bs_head * stride_s_bh,
        shape=(DK, DV),
        strides=(stride_s_dk, stride_s_dv),
        offsets=(start_k * BK, start_v * BV),
        block_shape=(BK, BV),
        order=(1, 0),
    )
    tl.store(S_block_ptr, s.to(S.dtype.element_ty), boundary_check=(0, 1))
    
    KM_block_ptr = KM + off_bs_head * stride_km_bh + start_k * BK + tl.arange(0, BK)
    tl.store(KM_block_ptr, km.to(KM.dtype.element_ty), mask=((start_k * BK + tl.arange(0, BK)) < DK))


@triton.autotune(
    configs=[
        triton.Config({'BL': BL, 'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BL in [32, 64, 128]
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['L']
)
@triton.jit
def fused_fwd_kernel_o(
    Q, Q_ROPE, S, O, KM,
    stride_qk_bh, stride_qk_l, stride_qk_d,
    stride_vo_bh, stride_vo_l, stride_vo_d,
    stride_s_bh, stride_s_dk, stride_s_dv,
    stride_km_bh,
    eps,
    L: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BL: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    The fused kernel computes O = (Q_ROPE @ S) / Z
    where Z = Q @ KM
    """
    start_v, start_l, off_bs_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    Q_block_ptr = tl.make_block_ptr(
        base=Q + off_bs_head * stride_qk_bh,
        shape=(L, DK),
        strides=(stride_qk_l, stride_qk_d),
        offsets=(start_l * BL, 0),
        block_shape=(BL, BK),
        order=(1, 0),
    )
    Q_ROPE_block_ptr = tl.make_block_ptr(
        base=Q_ROPE + off_bs_head * stride_qk_bh,
        shape=(L, DK),
        strides=(stride_qk_l, stride_qk_d),
        offsets=(start_l * BL, 0),
        block_shape=(BL, BK),
        order=(1, 0),
    )
    S_block_ptr = tl.make_block_ptr(
        base=S + off_bs_head * stride_s_bh,
        shape=(DK, DV),
        strides=(stride_s_dk, stride_s_dv),
        offsets=(0, start_v * BV),
        block_shape=(BK, BV),
        order=(1, 0),
    )
    KM_block_ptr = KM + off_bs_head * stride_km_bh + tl.arange(0, BK)
    
    o = tl.zeros([BL, BV], dtype=tl.float32)
    z = tl.zeros([BL], dtype=tl.float32)
    
    for offset_k in range(0, DK, BK):
        q = tl.load(Q_block_ptr, boundary_check=(0, 1))
        q_rope = tl.load(Q_ROPE_block_ptr, boundary_check=(0, 1))
        s = tl.load(S_block_ptr, boundary_check=(0, 1))
        km = tl.load(KM_block_ptr, mask=((offset_k + tl.arange(0, BK)) < DK))
    
        # Denominator Z uses pure Q
        z += tl.sum(q * km[None, :], axis=1, keep_dims=False)
        # Numerator uses Q_ROPE
        o += tl.dot(q_rope, s, allow_tf32=False)
        
        Q_block_ptr = tl.advance(Q_block_ptr, (0, BK))
        Q_ROPE_block_ptr = tl.advance(Q_ROPE_block_ptr, (0, BK))
        S_block_ptr = tl.advance(S_block_ptr, (BK, 0))
        KM_block_ptr = KM_block_ptr + tl.arange(0, BK)
        
    o = o / (z[:, None] + eps)

    O_block_ptr = tl.make_block_ptr(
        base=O + off_bs_head * stride_vo_bh,
        shape=(L, DV),
        strides=(stride_vo_l, stride_vo_d),
        offsets=(start_l * BL, start_v * BV),
        block_shape=(BL, BV),
        order=(1, 0),
    )
    tl.store(O_block_ptr, o.to(O.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def fused_bwd_kernel_ds_dq(
    O, Q, Q_ROPE, S, KM,
    dO, dQ, dQ_ROPE, dS, dKZ, 
    stride_qk_bh, stride_qk_l, stride_qk_d,
    stride_vo_bh, stride_vo_l, stride_vo_d,
    stride_ds_bh, stride_ds_k, stride_ds_v,
    stride_km_bh,
    stride_dkz_bh,
    eps,
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BL: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    start_kv, start_l, off_bs_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    NV = tl.cdiv(DV, BV)
    start_k = start_kv // (NV)
    start_v = start_kv % (NV)
    
    O_block_ptr = tl.make_block_ptr(
        base=O + off_bs_head * stride_vo_bh,
        shape=(L, DV),
        strides=(stride_vo_l, stride_vo_d),
        offsets=(start_l * BL, start_v * BV),
        block_shape=(BL, BV),
        order=(1, 0),
    )
    Q_block_ptr = tl.make_block_ptr(
        base=Q + off_bs_head * stride_qk_bh,
        shape=(L, DK),
        strides=(stride_qk_l, stride_qk_d),
        offsets=(start_l * BL, start_k * BK),
        block_shape=(BL, BK),
        order=(1, 0),
    )
    Q_ROPE_block_ptr = tl.make_block_ptr(
        base=Q_ROPE + off_bs_head * stride_qk_bh,
        shape=(L, DK),
        strides=(stride_qk_l, stride_qk_d),
        offsets=(start_l * BL, start_k * BK),
        block_shape=(BL, BK),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO + off_bs_head * stride_vo_bh,
        shape=(L, DV),
        strides=(stride_vo_l, stride_vo_d),
        offsets=(start_l * BL, start_v * BV),
        block_shape=(BL, BV),
        order=(1, 0),
    )
    S_block_ptr = tl.make_block_ptr(
        base=S + off_bs_head * stride_ds_bh,
        shape=(DK, DV),
        strides=(stride_ds_k, stride_ds_v),
        offsets=(start_k * BK, start_v * BV),
        block_shape=(BK, BV),
        order=(1, 0),
    )
    KM_block_ptr = KM + off_bs_head * stride_km_bh + start_k * BK + tl.arange(0, BK)
    
    ds = tl.zeros([BK, BV], dtype=tl.float32)
    dq = tl.zeros([BL, BK], dtype=tl.float32)
    dq_rope = tl.zeros([BL, BK], dtype=tl.float32)
    dz = tl.zeros([BL], dtype=tl.float32)
    dkz = tl.zeros([BK], dtype=tl.float32)
    
    q = tl.load(Q_block_ptr, boundary_check=(0, 1))
    q_rope = tl.load(Q_ROPE_block_ptr, boundary_check=(0, 1))
    o = tl.load(O_block_ptr, boundary_check=(0, 1))
    s = tl.load(S_block_ptr, boundary_check=(0, 1))
    do = tl.load(dO_block_ptr, boundary_check=(0, 1))
    km = tl.load(KM_block_ptr, mask=((start_k * BK + tl.arange(0, BK)) < DK))
    
    # Forward Z
    z = tl.sum(q * km[None, :], axis=1, keep_dims=True) + eps
    
    # Gradients
    ds += tl.dot(tl.trans(q_rope / z).to(do.dtype), do, allow_tf32=False)
    dz -= tl.sum(o * do / z, axis=1)
    
    dq_rope += tl.dot(do, tl.trans(s), allow_tf32=False) / z 
    dq += dz[:, None] * km[None, :]
    
    dkz += tl.sum(dz[:, None] * q, axis=0) / L

    dS_block_ptr = tl.make_block_ptr(
        base=dS + (off_bs_head + B * H * start_l) * stride_ds_bh,
        shape=(DK, DV),
        strides=(stride_ds_k, stride_ds_v),
        offsets=(start_k * BK, start_v * BV),
        block_shape=(BK, BV),
        order=(1, 0),
    )
    tl.store(dS_block_ptr, ds.to(dS.dtype.element_ty), boundary_check=(0, 1))
    
    dQ_block_ptr = tl.make_block_ptr(
        base=dQ + off_bs_head * stride_qk_bh,
        shape=(L, DK),
        strides=(stride_qk_l, stride_qk_d),
        offsets=(start_l * BL, start_k * BK),
        block_shape=(BL, BK),
        order=(1, 0),
    )
    tl.store(dQ_block_ptr, dq.to(dQ.dtype.element_ty), boundary_check=(0, 1))

    dQ_ROPE_block_ptr = tl.make_block_ptr(
        base=dQ_ROPE + off_bs_head * stride_qk_bh,
        shape=(L, DK),
        strides=(stride_qk_l, stride_qk_d),
        offsets=(start_l * BL, start_k * BK),
        block_shape=(BL, BK),
        order=(1, 0),
    )
    tl.store(dQ_ROPE_block_ptr, dq_rope.to(dQ_ROPE.dtype.element_ty), boundary_check=(0, 1))
    
    dKZ_block_ptr = dKZ + (off_bs_head + B * H * start_l) * stride_dkz_bh + start_k * BK + tl.arange(0, BK)
    tl.store(dKZ_block_ptr, dkz.to(dKZ.dtype.element_ty), mask=((start_k * BK + tl.arange(0, BK)) < DK))


@triton.jit
def fused_bwd_kernel_dk_dv(
    K_ROPE, V,
    dK, dK_ROPE, dV, dS, dKZ,
    stride_qk_bh, stride_qk_l, stride_qk_d,
    stride_vo_bh, stride_vo_l, stride_vo_d,
    stride_s_bh, stride_s_dk, stride_s_dv,
    stride_dkz_bh,
    scale,
    L: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BL: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    start_kv, start_l, off_bs_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    NV = tl.cdiv(DV, BV)
    start_k = start_kv // NV
    start_v = start_kv % NV
    
    K_ROPE_block_ptr = tl.make_block_ptr(
        base=K_ROPE + off_bs_head * stride_qk_bh,
        shape=(L, DK),
        strides=(stride_qk_l, stride_qk_d),
        offsets=(start_l * BL, start_k * BK),
        block_shape=(BL, BK),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + off_bs_head * stride_vo_bh,
        shape=(L, DV),
        strides=(stride_vo_l, stride_vo_d),
        offsets=(start_l * BL, start_v * BV),
        block_shape=(BL, BV),
        order=(1, 0),
    )
    dS_block_ptr = tl.make_block_ptr(
        base=dS + off_bs_head * stride_s_bh,
        shape=(DK, DV),
        strides=(stride_s_dk, stride_s_dv),
        offsets=(start_k * BK, start_v * BV),
        block_shape=(BK, BV),
        order=(1, 0),
    )
    dKZ_block_ptr = dKZ + off_bs_head * stride_dkz_bh + start_k * BK + tl.arange(0, BK)

    dk_rope = tl.zeros([BL, BK], dtype=tl.float32)
    dv = tl.zeros([BL, BV], dtype=tl.float32)
    
    k_rope = tl.load(K_ROPE_block_ptr, boundary_check=(0, 1))
    v = tl.load(V_block_ptr, boundary_check=(0, 1))
    ds = tl.load(dS_block_ptr, boundary_check=(0, 1))
    dkz = tl.load(dKZ_block_ptr, mask=((start_k * BK + tl.arange(0, BK)) < DK))

    v = (v * scale).to(v.dtype)
    
    dk_rope += tl.dot(v, tl.trans(ds), allow_tf32=False)
    dv += tl.dot(k_rope, ds, allow_tf32=False) * scale
    # dK completely derived from dKZ (broadcasted along L)
    dk = tl.zeros([BL, BK], dtype=tl.float32) + dkz[None, :]
    
    dK_block_ptr = tl.make_block_ptr(
        base=dK + off_bs_head * stride_qk_bh,
        shape=(L, DK),
        strides=(stride_qk_l, stride_qk_d),
        offsets=(start_l * BL, start_k * BK),
        block_shape=(BL, BK),
        order=(1, 0),
    )
    dK_ROPE_block_ptr = tl.make_block_ptr(
        base=dK_ROPE + off_bs_head * stride_qk_bh,
        shape=(L, DK),
        strides=(stride_qk_l, stride_qk_d),
        offsets=(start_l * BL, start_k * BK),
        block_shape=(BL, BK),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        base=dV + off_bs_head * stride_vo_bh,
        shape=(L, DV),
        strides=(stride_vo_l, stride_vo_d),
        offsets=(start_l * BL, start_v * BV),
        block_shape=(BL, BV),
        order=(1, 0),
    )
    tl.store(dK_block_ptr, dk.to(dK.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dK_ROPE_block_ptr, dk_rope.to(dK_ROPE.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dv.to(dV.dtype.element_ty), boundary_check=(0, 1))


class DecoupledLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, q_rope, k_rope, v, scale, eps):
        B, H, L, K_dim, V_dim = *q.shape, v.shape[-1]
        
        grid = lambda meta: (
            triton.cdiv(V_dim, meta['BV']),
            triton.cdiv(K_dim, meta['BK']),
            B * H
        )
        s = torch.empty(B, H, K_dim, V_dim, device=k.device)
        km = torch.empty(B, H, K_dim, device=k.device)
        fused_fwd_kernel_s_km[grid](
            k, k_rope, v, s, km,
            k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            s.stride(1), s.stride(2), s.stride(3),
            km.stride(1),
            scale,
            L=L, DK=K_dim, DV=V_dim,
        )
        s = s.to(k.dtype)
        km = km.to(k.dtype)
        
        grid = lambda meta: (
            triton.cdiv(V_dim, meta['BV']),
            triton.cdiv(L, meta['BL']),
            B * H
        )
        o = torch.empty_like(v)
        fused_fwd_kernel_o[grid](
            q, q_rope, s, o, km,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            s.stride(1), s.stride(2), s.stride(3),
            km.stride(1),
            eps=eps,
            L=L, DK=K_dim, DV=V_dim,
        )
        ctx.save_for_backward(q, k, q_rope, k_rope, v, o, s, km)
        ctx.scale = scale
        ctx.eps = eps
        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, q_rope, k_rope, v, o, s, km = ctx.saved_tensors
        B, H, L, K_dim, V_dim = *q.shape, v.shape[-1]
        
        BL = 32
        NL = triton.cdiv(L, BL)

        BK, BV = K_dim, V_dim
        NK, NV = triton.cdiv(K_dim, BK), triton.cdiv(V_dim, BV)
        assert NK == NV == 1, "The fused kernel currently only supports NK = NV = 1"

        grid = (NK * NV, NL, B * H)
        
        dq = torch.empty_like(q)
        dq_rope = torch.empty_like(q_rope)
        ds = torch.empty(NL, B, H, K_dim, V_dim, device=k.device)
        dkz = torch.empty(NL, B, H, K_dim, device=k.device)
        
        fused_bwd_kernel_ds_dq[grid](
            o, q, q_rope, s, km,
            do, dq, dq_rope, ds, dkz,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            ds.stride(2), ds.stride(3), ds.stride(4),
            km.stride(1),
            dkz.stride(2),
            eps=ctx.eps,
            B=B, H=H, L=L, DK=K_dim, DV=V_dim,
            BL=BL, BK=BK, BV=BV,
        )
        ds = ds.sum(0).to(k.dtype)
        dkz = dkz.sum(0)
        
        dk = torch.empty_like(k)
        dk_rope = torch.empty_like(k_rope)
        dv = torch.empty_like(v)
        
        fused_bwd_kernel_dk_dv[grid](
            k_rope, v,
            dk, dk_rope, dv, ds, dkz,
            k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            ds.stride(1), ds.stride(2), ds.stride(3),
            dkz.stride(1),
            scale=ctx.scale,
            L=L, DK=K_dim, DV=V_dim,
            BL=BL, BK=BK, BV=BV,
        )
        
        # return matching the number of inputs in forward: q, k, q_rope, k_rope, v, scale, eps
        return dq, dk, dq_rope, dk_rope, dv, None, None


def decoupled_la(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    eps: float = 1e-6
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `(B, H, L, K)`
        k (torch.Tensor):
            keys of shape `(B, H, L, K)`
        q_rope (torch.Tensor):
            queries with rotary position embeddings `(B, H, L, K)`
        k_rope (torch.Tensor):
            keys with rotary position embeddings `(B, H, L, K)`
        v (torch.Tensor):
            values of shape `(B, H, L, V)`
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / n**0.5`. Default: `None`.
        eps (float):
            A small constant added to the denominator to prevent division by zero.
    """
    if scale is None:
        scale = k.shape[-2] ** -1.0
    o = DecoupledLAFunction.apply(q, k, q_rope, k_rope, v, scale, eps)
    return o