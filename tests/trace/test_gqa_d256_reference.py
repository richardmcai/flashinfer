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
"""CPU checks of the standalone trace's scale convention and causal alignment."""

import math

import pytest
import torch

from flashinfer.trace.template import _render_reference_source
from flashinfer.trace.templates.gqa_d256 import _gated_gqa_d256_reference


@pytest.mark.parametrize("use_scale", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_standalone_reference(use_scale, causal):
    namespace = {}
    exec(_render_reference_source(_gated_gqa_d256_reference), namespace)
    reference = namespace["_gated_gqa_d256_reference"]
    q = torch.zeros(1, 2, 16, 256)
    q[..., 0] = 1
    k = torch.zeros(2, 32, 2, 256)
    v = torch.zeros_like(k)
    # Both logical tokens live in physical page 1, not page 0.
    k[1, 0, :, 0] = 1
    k[1, 1, :, 0] = 3
    v[1, 0] = 2
    v[1, 1] = 6
    actual = reference(
        q.to(torch.float8_e4m3fn),
        k.to(torch.float8_e4m3fn),
        v.to(torch.float8_e4m3fn),
        torch.zeros_like(q, dtype=torch.bfloat16),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        causal=causal,
        use_softmax_scale=use_scale,
        sm_scale=0.5,
    )
    # Scale mode: logits differ by 1 in base e. Noscale: logits differ by
    # 2 in base 2, so the second token is four times as likely.
    ratio = math.e if use_scale else 4.0
    expected = torch.full_like(actual, 1 + 2 * ratio / (1 + ratio))
    if causal:
        expected[:, 0] = 1  # first row sees just V=2, then sigmoid(0)=1/2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
