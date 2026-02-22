# Copyright (c) 2025-2026, Haopeng Li

import functools
import warnings

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["T"]
)
@triton.jit
def chunk_reduce_kernel(
    q, k, v,
    qc, kc, vc,
    T,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    i_k = i_kv // tl.cdiv(V, BV)
    i_v = i_kv % tl.cdiv(V, BV)

    BLOCK_SIZE = tl.minimum(BT, T - i_t * BT)

    p_q = tl.make_tensor_descriptor(q + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    p_k = tl.make_tensor_descriptor(k + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    p_v = tl.make_tensor_descriptor(v + i_bh * T * V, (T, V), (V, 1), (BT, BV))

    b_q = p_q.load([i_t * BT, i_k * BK])
    b_k = p_k.load([i_t * BT, i_k * BK])
    b_v = p_v.load([i_t * BT, i_v * BV])

    b_qc = tl.sum(b_q, axis=0) / BLOCK_SIZE
    b_kc = tl.sum(b_k, axis=0) / BLOCK_SIZE
    b_vc = tl.sum(b_v, axis=0)

    p_qc = tl.make_block_ptr(qc + i_bh * N * K + i_t * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    p_kc = tl.make_block_ptr(kc + i_bh * N * K + i_t * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    p_vc = tl.make_block_ptr(vc + i_bh * N * V + i_t * V, (V,), (1,), (i_v * BV,), (BV,), (0,))
    
    tl.store(p_qc, b_qc.to(p_qc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_kc, b_kc.to(p_kc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_vc, b_vc.to(p_vc.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["T"]
)
@triton.jit
def fused_chunk_reduce_kernel(
    q, k, v,
    qc, kc, vc, hc,
    T,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    i_k = i_kv // tl.cdiv(V, BV)
    i_v = i_kv % tl.cdiv(V, BV)

    BLOCK_SIZE = tl.minimum(BT, T - i_t * BT)

    p_q = tl.make_tensor_descriptor(q + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    p_k = tl.make_tensor_descriptor(k + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    p_v = tl.make_tensor_descriptor(v + i_bh * T * V, (T, V), (V, 1), (BT, BV))
    
    b_q = p_q.load([i_t * BT, i_k * BK])
    b_k = p_k.load([i_t * BT, i_k * BK])
    b_v = p_v.load([i_t * BT, i_v * BV])
    
    b_qc = tl.sum(b_q, axis=0) / BLOCK_SIZE
    b_kc = tl.sum(b_k, axis=0) / BLOCK_SIZE
    b_vc = tl.sum(b_v, axis=0)
    b_hc = tl.dot(tl.trans(b_k - b_kc[None, :]).to(b_v.dtype), b_v)

    p_qc = tl.make_block_ptr(qc + i_bh * N * K + i_t * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    p_kc = tl.make_block_ptr(kc + i_bh * N * K + i_t * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    p_vc = tl.make_block_ptr(vc + i_bh * N * V + i_t * V, (V,), (1,), (i_v * BV,), (BV,), (0,))
    p_hc = tl.make_block_ptr(hc + i_bh * N * K * V + i_t * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

    tl.store(p_qc, b_qc.to(p_qc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_kc, b_kc.to(p_kc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_vc, b_vc.to(p_vc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_hc, b_hc.to(p_hc.dtype.element_ty), boundary_check=(0, 1))


def chunk_reduce(q, k, v, chunk_size):
    B, H, T, K, V = *k.shape, v.shape[-1]
    N = triton.cdiv(T, chunk_size)
    
    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V)) 
    
    qc = torch.empty(B, H, N, K, device=k.device, dtype=k.dtype)
    kc = torch.empty(B, H, N, K, device=k.device, dtype=k.dtype)
    vc = torch.empty(B, H, N, V, device=v.device, dtype=v.dtype)
    
    grid = (triton.cdiv(K, BK) * triton.cdiv(V, BV), N, B * H)
    chunk_reduce_kernel[grid](
        q=q, k=k, v=v,
        qc=qc, kc=kc, vc=vc,
        T=T, N=N, K=K, V=V,
        BT=chunk_size, BK=BK, BV=BV
    )
    return qc, kc, vc


def fused_chunk_reduce(q, k, v, chunk_size, use_bias=False, p=2):
    B, H, T, K, V = *k.shape, v.shape[-1]
    N = triton.cdiv(T, chunk_size)
    
    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V)) 
    
    qc = torch.empty(B, H, N, K, device=k.device, dtype=k.dtype)
    kc = torch.empty(B, H, N, K, device=k.device, dtype=k.dtype)
    vc = torch.empty(B, H, N, V, device=v.device, dtype=v.dtype)
    hc = torch.empty(B, H, N, K, V, device=v.device, dtype=v.dtype)
    
    grid = (triton.cdiv(K, BK) * triton.cdiv(V, BV), N, B * H)
    fused_chunk_reduce_kernel[grid](
        q=q, k=k, v=v,
        qc=qc, kc=kc, vc=vc, hc=hc,
        T=T, N=N, K=K, V=V,
        BT=chunk_size, BK=BK, BV=BV
    )

    if use_bias:
        h_norm = torch.norm(hc.flatten(-2, -1), p=p, dim=-1)
        return qc, kc, vc, hc.sum(dim=2), h_norm[..., None, :]
    else:
        return qc, kc, vc, hc.sum(dim=2), None


@triton.autotune(
    configs=[
        triton.Config({'GROUP_SIZE': GROUP_SIZE}, num_warps=num_warps, num_stages=num_stages)
        for GROUP_SIZE in [32, 64, 128]
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["T"]
)
@triton.jit
def piecewise_sparse_attention_fwd_kernel(
    q, k, v,
    kc, vc,
    h, o,
    indices,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    NS: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    p_q = tl.make_tensor_descriptor(q + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    b_q = p_q.load([i_t * BT, 0])
    
    acc = tl.zeros([BT, BV], dtype=tl.float32)
    l_i = tl.zeros((BT,), dtype=tl.float32)
    m_i = tl.zeros((BT,), dtype=tl.float32) - float('inf')
    
    sm_scale = scale * 1.44269504

    # Phase 1: Exact Attention
    for i in range(NS):
        i_n = tl.load(indices + i_bh * NT * NS + i_t * NS + i).to(tl.int32)
        bos = i_n * BT
        
        p_k = tl.make_tensor_descriptor(k + i_bh * T * K, (T, K), (K, 1), (BT, BK))
        p_v = tl.make_tensor_descriptor(v + i_bh * T * V, (T, V), (V, 1), (BT, BV))
        b_k = p_k.load([bos, 0])
        b_v = p_v.load([bos, i_v * BV])
        
        b_s = tl.dot(b_q, tl.trans(b_k))
        b_s *= sm_scale
        b_s += tl.where((bos + tl.arange(0, BT))[None, :] < T, 0, float("-inf"))
        
        new_m_i = tl.maximum(m_i, tl.max(b_s, -1))
        alpha = tl.math.exp2(m_i - new_m_i)
        score = tl.math.exp2(b_s - new_m_i[:, None])
        
        l_i = l_i * alpha + tl.sum(score, -1)
        acc = acc * alpha[:, None] + tl.dot(score.to(b_v.dtype), b_v)
        m_i = new_m_i

    # Phase 2: Approx Attention (Zeroth-Order)
    last_chunk_len = T - (NT - 1) * BT

    offs_n_idx = tl.arange(0, triton.next_power_of_2(NS))
    loaded_indices = tl.load(indices + i_bh * NT * NS + i_t * NS + offs_n_idx, mask=offs_n_idx < NS, other=-1)
    
    g_l = tl.zeros([BT], dtype=tl.float32)

    for start_n in range(0, NT, GROUP_SIZE):
        p_kc = tl.make_tensor_descriptor(kc + i_bh * NT * K, (NT, K), (K, 1), (GROUP_SIZE, BK))
        b_kc = p_kc.load([start_n, 0])
        
        b_s_mean = tl.dot(b_q, tl.trans(b_kc))
        b_s_mean = b_s_mean * sm_scale
        
        chunk_indices = start_n + tl.arange(0, GROUP_SIZE)
        current_lens = tl.where(chunk_indices == NT - 1, last_chunk_len, BT)
        current_lens = current_lens.to(tl.float32)
        
        is_in_indices = chunk_indices[:, None] == loaded_indices[None, :]
        mask_is_selected = tl.max(is_in_indices, axis=1)
        valid_mask = (chunk_indices < NT) & (mask_is_selected == 0)
        
        b_s_mean = tl.where(valid_mask[None, :], b_s_mean, float("-inf"))

        new_m_i = tl.maximum(m_i, tl.max(b_s_mean, 1))
        alpha = tl.math.exp2(m_i - new_m_i)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        m_i = new_m_i
        
        prob_chunk = tl.math.exp2(b_s_mean - m_i[:, None]) 
        
        p_vc = tl.make_tensor_descriptor(vc + i_bh * NT * V, (NT, V), (V, 1), (GROUP_SIZE, BV))
        b_vc = p_vc.load([start_n, i_v * BV])
        
        acc += tl.dot(prob_chunk.to(b_vc.dtype), b_vc)
        weighted_prob = prob_chunk * current_lens[None, :]
        g_l += tl.sum(weighted_prob, axis=1)

    # Phase 3: Approx Attention (First-Order)
    p_h = tl.make_tensor_descriptor(h + i_bh * K * V , (K, V), (V, 1), (BK, BV))
    b_h = p_h.load([0, i_v * BV])

    b_r = tl.dot(b_q, b_h.to(b_q.dtype)) 
    correction_scale = g_l * (1.0 / T) * scale

    acc += b_r * correction_scale[:, None]

    # Final
    l_i += g_l
    acc /= l_i[:, None]
    
    p_o = tl.make_tensor_descriptor(o + i_bh * T * V, (T, V), (V, 1), (BT, BV))
    p_o.store([i_t * BT, i_v * BV], acc.to(b_q.dtype))


@torch.compile
def piecewise_sparse_attention(q, k, v, density=0.1, block_size=64, scale=None, use_bias=False):
    if not supports_host_descriptor():
        warnings.warn(
            "Optimization Note: Best performance is achieved on the Hopper platform (e.g., H100). "
            "Current platform may experience sub-optimal execution speeds.",
            UserWarning
        )
        
    B, H, T, K, V = *k.shape, v.shape[-1]
    scale = K ** -0.5 if scale is None else scale

    NT = triton.cdiv(T, block_size)
    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V))

    qc, kc, vc, h, bias = fused_chunk_reduce(q, k, v, block_size, use_bias=use_bias)
    
    score = torch.einsum('bhid, bhjd -> bhij', qc, kc * scale)
    if bias is not None:
        score = torch.softmax(score + torch.log(bias + 1e-5), dim=-1)

    top_k = max(1, int(density * NT))
    indices = torch.topk(score, k=top_k, dim=-1).indices 
    
    o = torch.empty_like(v)

    grid = (triton.cdiv(V, BV), NT, B * H)
    piecewise_sparse_attention_fwd_kernel[grid](
        q=q, k=k, v=v,
        kc=kc, vc=vc,
        h=h, o=o,
        indices=indices,
        scale=scale,
        T=T, K=K, V=V,
        BT=block_size, BK=BK, BV=BV,
        NT=NT, NS=top_k
    )
    
    return o