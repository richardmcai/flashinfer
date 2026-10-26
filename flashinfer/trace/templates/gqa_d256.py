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
"""Trace and independent mathematical reference for paged gated GQA-D256."""

import math

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


def _gated_gqa_d256_reference(
    q,
    k,
    v,
    gate,
    seq_lens,
    kv_indptr,
    kv_indices,
    *,
    out=None,
    causal=False,
    use_softmax_scale=True,
    sm_scale=None,
    enable_pdl=True,
):
    # Read actual stored FP8 operands, including quantization of prescaled Q.
    # The noscale kernel consumes base-2 logits, whereas torch.softmax uses e.
    scale = (
        (1.0 / 16 if sm_scale is None else sm_scale)
        if use_softmax_scale
        else 1.0 / math.log2(math.e)
    )
    pieces = []
    for batch in range(q.shape[0]):
        length = int(seq_lens[batch])
        start = int(kv_indptr[batch])
        count = (length + 31) // 32
        pages = kv_indices[start : start + count].long()
        keys = k.float()[pages].reshape(-1, 2, 256)[:length].repeat_interleave(8, dim=1)
        values = (
            v.float()[pages].reshape(-1, 2, 256)[:length].repeat_interleave(8, dim=1)
        )
        scores = (q[batch].float().transpose(0, 1) @ keys.permute(1, 2, 0)) * scale
        if causal:
            rows = torch.arange(q.shape[1], device=q.device)[:, None]
            cols = torch.arange(length, device=q.device)[None, :]
            scores.masked_fill_(cols > rows + length - q.shape[1], float("-inf"))
        attention = (scores.softmax(dim=-1) @ values.transpose(0, 1)).transpose(0, 1)
        pieces.append(attention * torch.sigmoid(gate[batch].float()))
    result = torch.stack(pieces).to(gate.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _gated_gqa_d256_init(
    *,
    batch_size=2,
    qo_len=16,
    device="cuda",
    seed=0,
    causal=False,
    use_softmax_scale=True,
    sm_scale=None,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    lengths = [max(qo_len, 33) + batch * 32 for batch in range(batch_size)]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + (length + 31) // 32)
    num_pages = offsets[-1]
    indices = torch.randperm(num_pages, generator=generator, device=device)
    pools = []
    for _ in range(2):
        pool = torch.randn(num_pages, 32, 2, 256, device=device, generator=generator)
        for batch, length in enumerate(lengths):
            if length % 32:
                page = indices[offsets[batch + 1] - 1]
                pool[page, length % 32 :] = 0
        pools.append(pool.to(torch.float8_e4m3fn))
    shape = (batch_size, qo_len, 16, 256)
    query = torch.randn(shape, device=device, generator=generator)
    if not use_softmax_scale:
        query *= (1.0 / 16 if sm_scale is None else sm_scale) * math.log2(math.e)
    return {
        "q": query.to(torch.float8_e4m3fn),
        "k": pools[0],
        "v": pools[1],
        "gate": torch.randn(shape, device=device, generator=generator).to(
            torch.bfloat16
        ),
        "seq_lens": torch.tensor(lengths, device=device, dtype=torch.int32),
        "kv_indptr": torch.tensor(offsets, device=device, dtype=torch.int32),
        "kv_indices": indices.to(torch.int32),
        "causal": causal,
        "use_softmax_scale": use_softmax_scale,
        "sm_scale": sm_scale,
    }


gated_gqa_d256_trace = TraceTemplate(
    op_type="gated_gqa_paged",
    name_prefix="cute_dsl_gqa_d256",
    description="Rubin paged FP8 GQA with FP16 softmax and fused sigmoid output gate.",
    axes={
        "batch_size": Var(),
        "qo_len": Var(),
        "num_pages": Var(),
        "indptr_len": Var(),
        "indices_len": Var(),
        "num_q_heads": Const(abbrev="hq"),
        "num_kv_heads": Const(abbrev="hkv"),
        "head_dim": Const(abbrev="d"),
        "page_size": Const(abbrev="ps"),
        "causal_mode": Const(
            abbrev="causal", extract=lambda kw: int(kw.get("causal", False))
        ),
        "scale_mode": Const(
            abbrev="scale", extract=lambda kw: int(kw.get("use_softmax_scale", True))
        ),
    },
    inputs={
        "q": Tensor(["batch_size", "qo_len", "num_q_heads", "head_dim"]),
        "k": Tensor(["num_pages", "page_size", "num_kv_heads", "head_dim"]),
        "v": Tensor(["num_pages", "page_size", "num_kv_heads", "head_dim"]),
        "gate": Tensor(["batch_size", "qo_len", "num_q_heads", "head_dim"]),
        "seq_lens": Tensor(["batch_size"], dtype="int32"),
        "kv_indptr": Tensor(["indptr_len"], dtype="int32"),
        "kv_indices": Tensor(["indices_len"], dtype="int32"),
        "causal": Scalar("bool", optional=True),
        "use_softmax_scale": Scalar("bool", optional=True),
        "sm_scale": Scalar("float32", optional=True),
        "enable_pdl": Scalar("bool", optional=True),
    },
    outputs={
        "output": Tensor(
            ["batch_size", "qo_len", "num_q_heads", "head_dim"], dtype_from="gate"
        ),
    },
    tags=["stage:attention", "backend:cute-dsl", "gated", "paged"],
    reference=_gated_gqa_d256_reference,
    init=_gated_gqa_d256_init,
)
