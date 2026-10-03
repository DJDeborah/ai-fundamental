"""Educational causal FlashAttention-style *forward only* Triton kernel.

Training uses PyTorch SDPA because this file has no custom backward. Compare this
kernel with SDPA on the same Q/K/V; it is not expected to beat all SDPA backends.
"""
import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _causal_fwd(Q, K, V, O, T: tl.constexpr, D: tl.constexpr,
                    SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        block_m = tl.program_id(0)
        head = tl.program_id(1)
        qpos = block_m * BM + tl.arange(0, BM)
        kpos = tl.arange(0, BN)
        dim = tl.arange(0, D)
        base = head * T * D
        q = tl.load(Q + base + qpos[:, None]*D + dim[None, :], qpos[:, None] < T, 0)
        m = tl.full((BM,), -float("inf"), tl.float32)
        l = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D), tl.float32)
        # Causal attention only needs key blocks up to this query block.
        for block_n in range(tl.cdiv((block_m + 1) * BM, BN)):
            npos = block_n * BN + kpos
            k = tl.load(K + base + npos[:, None]*D + dim[None, :], npos[:, None] < T, 0)
            v = tl.load(V + base + npos[:, None]*D + dim[None, :], npos[:, None] < T, 0)
            scores = tl.dot(q, tl.trans(k)) * SCALE
            valid = (npos[None, :] <= qpos[:, None]) & (npos[None, :] < T)
            scores = tl.where(valid, scores, -1.0e6)
            new_m = tl.maximum(m, tl.max(scores, axis=1))
            alpha = tl.exp(m - new_m)
            p = tl.exp(scores - new_m[:, None])
            acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
            l = l * alpha + tl.sum(p, axis=1)
            m = new_m
        out = acc / l[:, None]
        tl.store(O + base + qpos[:, None]*D + dim[None, :], out, qpos[:, None] < T)


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Install Triton in a compatible CUDA/Linux environment")
    if not (q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)):
        raise ValueError("CUDA fp16/bf16 inputs required")
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("Q/K/V need matching [batch, heads, time, dim]")
    if q.shape[-1] not in (32, 64, 128):
        raise ValueError("head dimension must be 32, 64, or 128")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    batch, heads, time, dim = q.shape
    out = torch.empty_like(q)
    _causal_fwd[(triton.cdiv(time, 32), batch * heads)](
        q, k, v, out, time, dim, 1.0 / math.sqrt(dim), 32, 64, num_warps=4)
    return out
