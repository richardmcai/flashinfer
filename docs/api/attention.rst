.. _apiattention:

FlashInfer Attention Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rubin Paged Sigmoid-Gated GQA-D256
=================================

``flashinfer.attention.cute_dsl.cute_dsl_gqa_d256_paged`` loads precompiled
TVM-FFI artifacts from the CuTe DSL FMHA cubin distribution. It supports
Rubin SM107 with FP8 E4M3 Q/K/V, 16 Q heads, 2 KV heads, QK/VO dimension 256,
NHD KV pages of 32 tokens, FP16 softmax, and BF16 output/gate logits. Both
noncausal and bottom-right causal attention are available. The sigmoid gate
is applied elementwise to the normalized attention output before conversion
to BF16. Gate fusion uses one KV split and no reduction.

The scale-enabled specialization applies ``sm_scale * log2(e)`` to logits,
with ``sm_scale=1/16`` by default. For ``use_softmax_scale=False``, the caller
must multiply Q by that factor **before FP8 quantization**. The kernel then
consumes base-2 logits and ignores the runtime ``sm_scale`` argument. Separate
FP8 dequantization scales are not supported.

All tensors must be contiguous, on the same CUDA device, and 16-byte aligned.
Q and BF16 gate/output have shape ``[B, S_q, 16, 256]``; K/V pools have shape
``[num_pages, 32, 2, 256]``. Supply Int32 ``seq_lens[B]``,
``kv_indptr[B+1]`` and flattened ``kv_indices``. Each request needs at least
``ceil(seq_lens[b]/32)`` valid page entries starting at ``kv_indptr[b]``.
Offsets may include gaps; referenced page IDs must be in range. KV lengths
must be positive and at least ``S_q`` for causal attention. Unused tokens in
partial final pages must be zero padded. Metadata values and padding are
the caller's responsibility; the API checks shapes, dtypes and alignment
without copying device data to the host.

For local verification, set ``FLASHINFER_DSL_FMHA_LOCAL_DIR`` to the export's
``sm_107a`` directory. The installed runtime needs CuTe DSL and TVM-FFI,
with no kernel compilation required. Warm up the exact API invocation before
CUDA graph capture and pass a preallocated ``out`` that does not share input
storage. Paging metadata must also be prepared before capture.

For callers that already have FlashInfer paging metadata, prepare lengths and
warm up before capture::

    from flashinfer.attention.cute_dsl import cute_dsl_gqa_d256_paged

    # All metadata is contiguous Int32 CUDA storage; Q is [B, S_q, 16, 256].
    seq_lens = (kv_indptr[1:] - kv_indptr[:-1] - 1) * 32 + last_page_len
    out = torch.empty_like(gate)  # gate and output are BF16
    cute_dsl_gqa_d256_paged(
        q, k_pages, v_pages, gate, seq_lens, kv_indptr, kv_indices,
        out=out, causal=True,
    )

.. autofunction:: flashinfer.attention.cute_dsl.gqa_d256.cute_dsl_gqa_d256_paged


Experimental Task-Scheduled Attention
=====================================

The experimental Blackwell task-scheduled FMHA context, FMHA decode,
block-sparse FMHA, and MLA decode APIs are imported from
``flashinfer.attention.prims_ts``. Scheduling, tile selection, and split-KV
reduction are automatic implementation details; there are no public tuning
knobs.

See the `PrimTS guide index <https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/attention/prims_ts/README.md>`_
for the public entry points, supported contracts, and examples. Current accuracy
and performance signoff is on SM100a/B200; SM103a/B300 is architecture-gated
but not yet signoff-qualified.

Calling these APIs is an explicit opt-in and emits an
``ExperimentalWarning`` once per decorated function. They provide no API
compatibility guarantee; generated stable API reference entries are deferred
until graduation. Logging and existing ``fi_trace`` bindings remain available.

QToken-KvBlock-Sparse-Attention
--------------------------------

QToken-KvBlock-Sparse-Attention consumes per-query
``indexer_block_ids[total_q, block_topk]`` and a dense physical
``block_table``. Packed prefill uses ``[total_q, Hq, D]`` with
``qo_indptr``; fixed MTP decode uses ``[B, Nq, G, Hq, D]``.
``kv_block_size`` is the semantic sparse K/V atom and currently supports
only four tokens. The wrapper plans capacity outside CUDA Graph capture and
runs live route metadata on the hot path.


flashinfer.decode
=================

.. currentmodule:: flashinfer.decode

Single Request Decoding
-----------------------

.. autosummary::
    :toctree: ../generated

    single_decode_with_kv_cache
    single_decode_with_kv_cache_with_jit_module

Batch Decoding
--------------

.. autosummary::
    :toctree: ../generated

    cudnn_batch_decode_with_kv_cache
    trtllm_batch_decode_with_kv_cache
    xqa_batch_decode_with_kv_cache

DCP Speculative Decode Workspace
--------------------------------

The native Cake FMHA DCP speculative route of
:func:`flashinfer.decode.trtllm_batch_decode_with_kv_cache` uses caller-owned
scratch buffers so a prewarmed invocation can be captured in a CUDA Graph.
It is also reachable through
:func:`flashinfer.cake_fmha.cake_batch_decode_with_kv_cache`; the non-null
``causal_seqlens_kv_global`` argument is the explicit add-on selection key.
On SM103, the same Cake entrypoint accepts a device ``request_order`` tensor
for BF16-query, FP8-E4M3 paged decode with head dimension 256.  Precompute an
optional immutable length-aware schedule with
:func:`flashinfer.plan_cake_fmha_request_ordered_paged_decode` before graph
capture.  Page-table rows must be padded to
``4 * ceil(max_seq_len / 256)`` entries, and the exact tensor/workspace binding
must be invoked once eagerly to initialize its TMA descriptors before capture.
After that prewarm, changing only the order tensor contents does not require
recapture.

.. currentmodule:: flashinfer

.. autosummary::
    :toctree: ../generated

    get_dcp_spec_workspace_size_bytes
    get_dcp_spec_counter_bytes
    plan_cake_fmha_request_ordered_paged_decode
    CakeFmhaRequestOrderedDecodePlan

.. currentmodule:: flashinfer.decode

.. autoclass:: BatchDecodeWithPagedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, forward, forward_return_lse

    .. automethod:: __init__

.. autoclass:: CUDAGraphBatchDecodeWithPagedKVCacheWrapper
    :members:

    .. automethod:: __init__


XQA
---

.. currentmodule:: flashinfer.xqa

.. autosummary::
    :toctree: ../generated

    xqa
    xqa_mla

flashinfer.prefill
==================

Attention kernels for prefill & append attention in both single request and batch serving setting.

.. currentmodule:: flashinfer.prefill

Single Request Prefill/Append Attention
---------------------------------------

.. autosummary::
    :toctree: ../generated

    single_prefill_with_kv_cache
    single_prefill_with_kv_cache_return_lse
    single_prefill_with_kv_cache_with_jit_module

Batch Prefill/Append Attention
------------------------------

.. autosummary::
    :toctree: ../generated

    cudnn_batch_prefill_with_kv_cache
    trtllm_batch_context_with_kv_cache
    trtllm_ragged_attention_deepseek
    fmha_v2_prefill_deepseek
    trtllm_fmha_v2_prefill
    fmha_v2_prefill_sm120

.. autoclass:: BatchPrefillWithPagedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, forward, forward_return_lse

    .. automethod:: __init__

.. autoclass:: BatchPrefillWithRaggedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, forward, forward_return_lse

    .. automethod:: __init__


Causal + Bidirectional Ranges Prefill
-------------------------------------

.. currentmodule:: flashinfer.attention

A batch-prefill wrapper whose fa2 attention variant owns the whole mask:
causal, plus an inclusive per-query key span attended in both directions. The
spans are handed to :meth:`BatchPrefillWithCausalBidirectionalRangesWrapper.run`
as a compact ``int32 [total_q, 2]`` tensor and no mask is materialized, so
nothing scales with ``qo_len * kv_len``. The JIT module is specialized in the
constructor, and the inherited options the variant makes meaningless are
rejected rather than ignored.

.. autoclass:: BatchPrefillWithCausalBidirectionalRangesWrapper
    :members:

    .. automethod:: __init__


Unified BatchAttention
----------------------

.. currentmodule:: flashinfer.attention

The ``BatchAttention`` class provides a holistic attention wrapper that automatically dispatches
between paged-prefill and paged-decode based on per-request sequence lengths. It is the
recommended entry point for serving stacks that batch mixed prefill/decode requests in a
single kernel launch.

.. autoclass:: BatchAttention
    :members:

    .. automethod:: __init__

.. autoclass:: BatchAttentionWithAttentionSinkWrapper
    :members:

    .. automethod:: __init__


SM120 NVFP4 Attention
---------------------

.. currentmodule:: flashinfer.nvfp4_attention_sm120

.. autosummary::
    :toctree: ../generated

    nvfp4_attention_sm120_quantize_qkv
    nvfp4_attention_sm120_fwd


flashinfer.mla
==============

MLA (Multi-head Latent Attention) is an attention mechanism proposed in DeepSeek series of models (
`DeepSeek-V2 <https://arxiv.org/abs/2405.04434>`_, `DeepSeek-V3 <https://arxiv.org/abs/2412.19437>`_,
and `DeepSeek-R1 <https://arxiv.org/abs/2501.12948>`_).

.. currentmodule:: flashinfer.mla

PageAttention for MLA
---------------------

.. autosummary::
    :toctree: ../generated

    trtllm_batch_decode_with_kv_cache_mla
    trtllm_prefill_with_kv_cache_mla
    trtllm_batch_decode_sparse_mla_dsv4
    nvfp4_quantize_pack_sparse_mla_cache
    nvfp4_quantize_append_sparse_mla_cache
    dsv41_fp4_quantize_pack_sparse_mla_cache
    dsv41_fp4_quantize_append_sparse_mla_cache
    dsv41_fp8_quantize_pack_sparse_mla_cache
    dsv41_fp8_quantize_append_sparse_mla_cache
    convert_compressed_page_aligned_sparse_indices_to_hca_metadata
    DSV4HCAMetadata
    xqa_batch_decode_with_kv_cache_mla
    supported_sparse_mla_sm120_configs
    SparseMLASm120DecodeConfig
    SparseMLASm120Wrapper
    cake_sparse_mla_sm120_dsv4_nvfp4_decode
    cake_sparse_mla_sm120_dsv4_nvfp4_prefill
    cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel
    cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill
    cake_sparse_mla_sm120_dsv4_nvfp4_plan_head_tiles
    cake_sparse_mla_sm120_dsv4_nvfp4_plan_splits
    cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes
    cake_dsv4_nvfp4_rope_quantize_insert
    cake_dsv4_nvfp4_kv_rope_quantize_insert

.. note::

    With ``backend="cake"`` on SM120/SM121 (``kv_cache_format="nvfp4"``), every
    call runs the Cake split decode kernel or the single-launch prefill kernel
    as chosen by ``cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel`` from the
    token, head and candidate counts and the SM count (DeepSeek-V4 sparse MLA
    SM120 tracker: flashinfer#4254).

.. note::

    The NVFP4 cache that route reads is written per layer by the fused
    SM120/SM121 writers ``cake_dsv4_nvfp4_rope_quantize_insert`` (sliding-window
    pool: GPT-J RoPE of the query and latent KV, head-padded ``q_out`` or, with
    ``q_inplace=True`` and no head padding, the query rotated in place, NVFP4
    quantization and paged insert in one launch) and
    ``cake_dsv4_nvfp4_kv_rope_quantize_insert`` (compressed pool with
    ``compress_ratio`` 1 or 2, speculative context with ratio 1); both produce
    the bytes of ``nvfp4_quantize_append_sparse_mla_cache`` applied to the
    BF16-rounded roped rows.

.. note::

    With ``backend="cute-dsl"``, pass ``hca_swa_indices`` as absolute rows into
    the flattened SWA cache and ``hca_compressed_block_tables`` as physical
    compressed-cache page IDs. The SWA table has shape ``[B * Q, 128]`` and may
    express ring rotation or wraparound. Combined tables whose compressed
    segment is a canonical page expansion can opt into compatibility conversion
    with ``hca_sparse_indices_format="compressed-page-aligned"``. SWA entries
    remain arbitrary absolute rows. Precompute that conversion before a CUDA
    Graph or a latency-sensitive loop.

.. note::

    ``kv_cache_format="nvfp4"`` (the 384-byte-per-token DeepSeek-V4 NVFP4 sparse
    cache: 448 NoPE values as E2M1 with one E4M3 scale per 16 values, 64 BF16
    RoPE values, ``page_size * 352`` data bytes followed by ``page_size * 32``
    scale bytes per page) is consumed by ``backend="sparse"`` on SM120 / SM121 and
    by ``backend="cake"`` on SM100 / SM103 (B200 / GB300). The CAKE route takes a
    BF16 query, two independent tables (``sparse_indices`` over ``swa_kv_cache``
    and ``extra_sparse_indices`` over ``compressed_kv_cache`` with their own
    ``*_topk_lens``; ``-1`` entries are masked), ``sinks``, a caller-owned
    ``workspace_buffer`` sized by
    :func:`flashinfer.mla.cake_dsv4.get_cake_dsv4_workspace_bytes` and is
    CUDA-Graph safe. Build the cache with
    :func:`nvfp4_quantize_pack_sparse_mla_cache` /
    :func:`nvfp4_quantize_append_sparse_mla_cache` (one implementation for all
    four architectures). ``backend="auto"`` keeps selecting TRTLLM-GEN on
    SM100 / SM103; pass ``backend="cake"`` explicitly.

.. autoclass:: BatchMLAPagedAttentionWrapper
    :members:

    .. automethod:: __init__
