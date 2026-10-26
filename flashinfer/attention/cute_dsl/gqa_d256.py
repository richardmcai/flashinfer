# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rubin paged FP8 sigmoid-gated GQA-D256, loaded from offline exports."""

import functools
import math
import os

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.trace.templates.gqa_d256 import gated_gqa_d256_trace

from .fmha import _get_gpu_arch, _load_from_artifact, _load_from_local


def _get_variant_name(causal: bool, use_softmax_scale: bool) -> str:
    mask = "causal" if causal else "nocausal"
    scale = "scale" if use_softmax_scale else "noscale"
    return (
        "cute_dsl_gqa_d256_e4m3_bf16_hq16_hkv2_d256_paged32_"
        f"fp16sm_sigmoid_{mask}_{scale}_g8_q16_k128_tvmffi"
    )


@functools.cache
def _load_kernel(causal: bool, use_softmax_scale: bool, local_dir: str):
    name = _get_variant_name(causal, use_softmax_scale)
    if local_dir:
        return _load_from_local(name, local_dir, enable_tvm_ffi=True)
    return _load_from_artifact(name, "sm_107a", enable_tvm_ffi=True)


@flashinfer_api(trace=gated_gqa_d256_trace)
def cute_dsl_gqa_d256_paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    causal: bool = False,
    use_softmax_scale: bool = True,
    sm_scale: float | None = None,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """Run paged attention followed by a fused elementwise sigmoid gate.

    Supports only Rubin SM107, FP8 E4M3 Q/K/V, Q heads=16, KV heads=2,
    head dimension=256, page size=32, and BF16 gate/output. Softmax uses
    FP16 arithmetic with FP32 MMA accumulation. Split-KV is disabled.

    Parameters
    ----------
    q : torch.Tensor
        Contiguous query, shape ``[B, S_q, 16, 256]``.
    k, v : torch.Tensor
        Contiguous NHD page pools, shape ``[num_pages, 32, 2, 256]``.
        Unused tokens in partial final pages must be zero padded.
    gate : torch.Tensor
        Contiguous BF16 logits, same shape as Q. The kernel computes
        ``attention * sigmoid(gate)`` before converting to BF16.
    seq_lens : torch.Tensor
        Contiguous Int32 logical KV lengths, shape ``[B]``. Every length
        must be positive and, for causal attention, at least ``S_q``.
    kv_indptr : torch.Tensor
        Contiguous Int32 offsets into ``kv_indices``, shape ``[B + 1]``.
        Request b needs at least ``ceil(seq_lens[b] / 32)`` page entries.
        Offsets may include gaps and a nonzero initial offset.
    kv_indices : torch.Tensor
        Contiguous flattened Int32 physical page indices. Every referenced
        index must be in range of the K/V pools.
    out : torch.Tensor, optional
        Contiguous BF16 output, same shape as Q; allocated if omitted.
        Must not share storage with any input.
    causal : bool
        Apply bottom-right-aligned causal masking.
    use_softmax_scale : bool
        If True, apply ``sm_scale * log2(e)`` inside the kernel. If False,
        Q must already be prescaled by that factor before FP8 quantization;
        ``sm_scale`` is ignored. This mode does not prescale Q internally.
    sm_scale : float, optional
        Positive finite score scale, default ``1 / sqrt(256)``. FP8 values
        are interpreted directly; separate Q/K/V dequantization factors
        are not supported.
    enable_pdl : bool
        Enable Programmatic Dependent Launch.

    Returns
    -------
    torch.Tensor
        BF16 gated attention output.

    Notes
    -----
    All tensors must be on the same CUDA device and 16-byte aligned.
    Only metadata shapes/dtypes are checked; callers must validate metadata
    values and zero padding before use. No device-to-host synchronization
    or JIT compilation is performed. Before CUDA graph capture, invoke once
    eagerly to load the artifact and pass a preallocated ``out`` at capture.
    Set ``FLASHINFER_DSL_FMHA_LOCAL_DIR`` to the export's ``sm_107a`` directory
    to use locally built artifacts before publication.
    """
    if not q.is_cuda or _get_gpu_arch(q.device) != "sm_107a":
        raise ValueError("gated GQA-D256 requires a Rubin SM107 CUDA device")
    if q.ndim != 4 or q.shape[2:] != (16, 256):
        raise ValueError("q must have shape [B, S_q, 16, 256]")
    b, sq = q.shape[:2]
    if b <= 0 or sq <= 0:
        raise ValueError("batch size and query length must be positive")
    if k.ndim != 4 or k.shape[1:] != (32, 2, 256) or k.shape[0] <= 0:
        raise ValueError("k must have shape [num_pages, 32, 2, 256]")
    if v.shape != k.shape:
        raise ValueError("v must have the same shape as k")
    if gate.shape != q.shape:
        raise ValueError("gate must have the same shape as q")
    if seq_lens.shape != (b,) or kv_indptr.shape != (b + 1,):
        raise ValueError("seq_lens and kv_indptr must have shapes [B] and [B + 1]")
    if kv_indices.ndim != 1 or kv_indices.numel() == 0:
        raise ValueError("kv_indices must be a nonempty flattened page table")

    inputs = (
        ("q", q, torch.float8_e4m3fn),
        ("k", k, torch.float8_e4m3fn),
        ("v", v, torch.float8_e4m3fn),
        ("gate", gate, torch.bfloat16),
        ("seq_lens", seq_lens, torch.int32),
        ("kv_indptr", kv_indptr, torch.int32),
        ("kv_indices", kv_indices, torch.int32),
    )
    for name, tensor, dtype in inputs:
        _check_tensor(name, tensor, dtype, q.device)
    if not all(
        isinstance(flag, bool) for flag in (causal, use_softmax_scale, enable_pdl)
    ):
        raise ValueError("causal, use_softmax_scale and enable_pdl must be bools")
    scale = 1.0 / 16 if sm_scale is None or not use_softmax_scale else float(sm_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("sm_scale must be finite and positive")
    if out is None:
        out = torch.empty_like(gate)
    if out.shape != q.shape:
        raise ValueError("out must have the same shape as q")
    _check_tensor("out", out, torch.bfloat16, q.device)
    out_storage = out.untyped_storage().data_ptr()
    if any(t.untyped_storage().data_ptr() == out_storage for _, t, _ in inputs):
        raise ValueError("out must not share storage with an input")

    with torch.cuda.device(q.device):
        kernel = _load_kernel(
            causal,
            use_softmax_scale,
            os.environ.get("FLASHINFER_DSL_FMHA_LOCAL_DIR", ""),
        )
        kernel(
            q, k, v, gate, out, seq_lens, kv_indptr[:b], kv_indices, scale, enable_pdl
        )
    return out


def _check_tensor(name, tensor, dtype, device):
    if tensor.device != device or tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype} on {device}")
    if not tensor.is_contiguous() or tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be contiguous and 16-byte aligned")
