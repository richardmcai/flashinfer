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
"""Artifact integration tests; require Rubin and locally built/published exports."""

import math

import pytest
import torch

from flashinfer.cute_dsl.availability import is_cute_dsl_available
from flashinfer.trace.templates.gqa_d256 import _gated_gqa_d256_reference

if not is_cute_dsl_available():
    pytest.skip("CuTe DSL is unavailable", allow_module_level=True)

from flashinfer.attention.cute_dsl.gqa_d256 import cute_dsl_gqa_d256_paged

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Gated GQA-D256 requires Rubin SM107",
)


def _inputs(sq, lengths, use_scale, sm_scale=1.0 / 16):
    torch.manual_seed(42)
    shape = (len(lengths), sq, 16, 256)
    q = torch.randn(shape, device="cuda")
    if not use_scale:
        q *= sm_scale * math.log2(math.e)
    counts = [(length + 31) // 32 for length in lengths]
    pages = sum(counts)
    permutation = torch.randperm(pages).tolist()
    # Deliberately use a nonzero initial offset and gaps between table rows.
    table = [0, 0]
    offsets = [2]
    pool_k = torch.zeros(pages, 32, 2, 256, device="cuda")
    pool_v = torch.zeros_like(pool_k)
    cursor = 0
    for length, count in zip(lengths, counts, strict=True):
        row = permutation[cursor : cursor + count]
        cursor += count
        table.extend(row)
        table.append(0)
        offsets.append(len(table))
        for logical, physical in enumerate(row):
            valid = min(32, length - logical * 32)
            pool_k[physical, :valid] = torch.randn(valid, 2, 256, device="cuda")
            pool_v[physical, :valid] = torch.randn(valid, 2, 256, device="cuda")
    return dict(
        q=q.to(torch.float8_e4m3fn),
        k=pool_k.to(torch.float8_e4m3fn),
        v=pool_v.to(torch.float8_e4m3fn),
        gate=torch.randn(shape, device="cuda").to(torch.bfloat16),
        seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        kv_indptr=torch.tensor(offsets, device="cuda", dtype=torch.int32),
        kv_indices=torch.tensor(table, device="cuda", dtype=torch.int32),
        use_softmax_scale=use_scale,
        sm_scale=sm_scale,
    )


@pytest.mark.parametrize("sq", [1, 7, 8, 15, 16, 17, 32, 33, 257])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("use_scale", [False, True])
def test_paged_gated_gqa(sq, causal, use_scale):
    lengths = [max(sq, length) for length in (1, 31, 32, 33, 127, 128, 129)]
    inputs = _inputs(sq, lengths, use_scale)
    out = torch.empty_like(inputs["gate"])
    actual = cute_dsl_gqa_d256_paged(**inputs, causal=causal, out=out)
    assert actual is out
    expected = _gated_gqa_d256_reference(**inputs, causal=causal)
    # Budget includes FP16 softmax, FP8 P conversion and BF16 output rounding.
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2e-2)


@pytest.mark.parametrize("use_scale", [False, True])
def test_scale_and_gate_contract(use_scale):
    inputs = _inputs(17, [33, 129], use_scale, sm_scale=0.125)
    # One call exercises negative saturation, half gating and positive saturation.
    inputs["gate"][..., :85] = -80
    inputs["gate"][..., 85:170] = 0
    inputs["gate"][..., 170:] = 80
    actual = cute_dsl_gqa_d256_paged(**inputs, causal=True)
    expected = _gated_gqa_d256_reference(**inputs, causal=True)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2e-2)
    assert actual[..., :85].abs().max().item() < 1e-25
    if not use_scale:
        # sm_scale is unused by this specialization; never apply it twice.
        inputs["sm_scale"] = 0.5
        repeated = cute_dsl_gqa_d256_paged(**inputs, causal=True)
        torch.testing.assert_close(repeated, actual, rtol=0, atol=0)


@pytest.mark.parametrize("use_scale", [False, True])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_stream_and_cuda_graph(use_scale, enable_pdl):
    inputs = _inputs(16, [33, 65], use_scale)
    inputs["enable_pdl"] = enable_pdl
    out = torch.empty_like(inputs["gate"])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            cute_dsl_gqa_d256_paged(**inputs, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    expected = _gated_gqa_d256_reference(**inputs)
    torch.testing.assert_close(out, expected, rtol=5e-2, atol=2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        cute_dsl_gqa_d256_paged(**inputs, out=out)
    inputs["gate"].zero_()
    graph.replay()
    expected = _gated_gqa_d256_reference(**inputs)
    torch.testing.assert_close(out, expected, rtol=5e-2, atol=2e-2)


@pytest.mark.parametrize(
    "bad_input",
    [
        "q_dtype",
        "page_size",
        "heads",
        "gate_dtype",
        "metadata",
        "alias",
        "alignment",
        "strides",
    ],
)
def test_invalid_contract(monkeypatch, bad_input):
    import flashinfer.attention.cute_dsl.gqa_d256 as module

    inputs = _inputs(1, [33], True)
    if bad_input == "q_dtype":
        inputs["q"] = inputs["q"].to(torch.bfloat16)
    elif bad_input == "page_size":
        inputs["k"] = inputs["k"][:, :16].contiguous()
        inputs["v"] = inputs["v"][:, :16].contiguous()
    elif bad_input == "heads":
        inputs["q"] = inputs["q"][:, :, :8].contiguous()
    elif bad_input == "gate_dtype":
        inputs["gate"] = inputs["gate"].float()
    elif bad_input == "metadata":
        inputs["seq_lens"] = inputs["seq_lens"].long()
    elif bad_input == "alias":
        inputs["out"] = inputs["gate"]
    elif bad_input == "alignment":
        indices = torch.zeros(
            inputs["kv_indices"].numel() + 1, device="cuda", dtype=torch.int32
        )
        inputs["kv_indices"] = indices[1:]
    elif bad_input == "strides":
        inputs["q"] = torch.empty(
            1, 1, 16, 512, device="cuda", dtype=torch.float8_e4m3fn
        )[..., ::2]

    def fail_load(*args):
        pytest.fail("invalid arguments reached the artifact loader")

    monkeypatch.setattr(module, "_load_kernel", fail_load)
    with pytest.raises(ValueError):
        cute_dsl_gqa_d256_paged(**inputs)
