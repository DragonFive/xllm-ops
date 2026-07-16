#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


INVALID_LOGPROB = -1.0e20


def _build_tables():
    vocab_size = 16
    first_token_ids = torch.tensor([1, 7, 11, 12], dtype=torch.int32)

    prefix1 = {
        1: [2, 5],
        7: [8],
        11: [0, 4, 13, 14],
        12: [15],
    }
    prefix2 = {
        (1, 2): [3, 4],
        (1, 5): [6],
        (7, 8): [9, 10],
        (11, 0): [2, 3, 5, 6],
        (11, 4): [7],
        (11, 13): [8, 9],
        (11, 14): [10],
        (12, 15): [1, 2, 3],
    }

    prefix1_offsets = [0]
    prefix1_values = []
    prefix1_pair_keys = []
    prefix2_value_offsets = [0]
    prefix2_values = []
    for t0 in range(vocab_size):
        values = prefix1.get(t0, [])
        for t1 in values:
            prefix1_values.append(t1)
            prefix1_pair_keys.append(t0 * vocab_size + t1)
            next_values = prefix2[(t0, t1)]
            prefix2_values.extend(next_values)
            prefix2_value_offsets.append(len(prefix2_values))
        prefix1_offsets.append(len(prefix1_values))

    return {
        "vocab_size": vocab_size,
        "first_token_ids": first_token_ids,
        "prefix1_offsets": torch.tensor(prefix1_offsets, dtype=torch.int32),
        "prefix1_values": torch.tensor(prefix1_values, dtype=torch.int32),
        "prefix1_pair_keys": torch.tensor(prefix1_pair_keys, dtype=torch.int64),
        "prefix2_value_offsets": torch.tensor(prefix2_value_offsets, dtype=torch.int32),
        "prefix2_values": torch.tensor(prefix2_values, dtype=torch.int32),
        "max_prefix1_degree": max(len(v) for v in prefix1.values()),
        "max_prefix2_degree": max(len(v) for v in prefix2.values()),
    }


def _build_wide_step0_tables(vocab_size=768, candidate_count=640):
    first_token_ids = torch.arange(candidate_count, dtype=torch.int32)
    return {
        "vocab_size": vocab_size,
        "first_token_ids": first_token_ids,
        "prefix1_offsets": torch.zeros([vocab_size + 1], dtype=torch.int32),
        "prefix1_values": torch.tensor([0], dtype=torch.int32),
        "prefix1_pair_keys": torch.tensor([0], dtype=torch.int64),
        "prefix2_value_offsets": torch.tensor([0, 0], dtype=torch.int32),
        "prefix2_values": torch.tensor([0], dtype=torch.int32),
        "max_prefix1_degree": 1,
        "max_prefix2_degree": 1,
    }


def _allowed_tokens(row, current_step, sequence_group, tables):
    if current_step == 0:
        return tables["first_token_ids"].to(torch.long)
    sequence_flat = sequence_group.reshape(-1, sequence_group.shape[-1])
    t0 = int(sequence_flat[row, 0].item())
    if current_step == 1:
        if t0 < 0 or t0 + 1 >= tables["prefix1_offsets"].numel():
            return torch.empty([0], dtype=torch.long)
        begin = int(tables["prefix1_offsets"][t0].item())
        end = int(tables["prefix1_offsets"][t0 + 1].item())
        return tables["prefix1_values"][begin:end].to(torch.long)

    t1 = int(sequence_flat[row, 1].item())
    query_key = t0 * tables["vocab_size"] + t1
    pair_keys = tables["prefix1_pair_keys"]
    pair_index = torch.searchsorted(pair_keys, torch.tensor(query_key, dtype=torch.int64))
    pair_index_value = int(pair_index.item())
    if pair_index_value >= pair_keys.numel() or int(pair_keys[pair_index_value].item()) != query_key:
        return torch.empty([0], dtype=torch.long)
    begin = int(tables["prefix2_value_offsets"][pair_index_value].item())
    end = int(tables["prefix2_value_offsets"][pair_index_value + 1].item())
    return tables["prefix2_values"][begin:end].to(torch.long)


def _default_token(current_step, tables):
    if current_step == 0:
        return int(tables["first_token_ids"][0].item())
    if current_step == 1:
        return int(tables["prefix1_values"][0].item())
    return int(tables["prefix2_values"][0].item())


def _reference_topk(logits, sequence_group, temperatures, current_step, top_k, tables):
    logits_fp32 = logits.float().cpu()
    if temperatures.numel() == 0:
        temps = torch.ones([1], dtype=torch.float32)
    else:
        temps = temperatures.float().cpu()

    out_tokens = torch.empty([logits.shape[0], top_k], dtype=torch.int32)
    out_logprobs = torch.empty([logits.shape[0], top_k], dtype=torch.float32)
    default_token = _default_token(current_step, tables)
    for row in range(logits.shape[0]):
        allowed = _allowed_tokens(row, current_step, sequence_group.cpu(), tables)
        if allowed.numel() == 0:
            out_tokens[row].fill_(default_token)
            out_logprobs[row].fill_(INVALID_LOGPROB)
            continue

        temp = temps[0] if temps.numel() == 1 else temps[row]
        if float(temp.item()) == 0.0:
            temp = torch.tensor(1.0, dtype=torch.float32)
        candidate_logits = logits_fp32[row, allowed] / temp
        candidate_logprobs = torch.log_softmax(candidate_logits, dim=0)
        k = min(top_k, allowed.numel())
        values, indices = torch.topk(candidate_logprobs, k=k, largest=True, sorted=True)
        out_tokens[row, :k] = allowed[indices].to(torch.int32)
        out_logprobs[row, :k] = values
        if k < top_k:
            out_tokens[row, k:].fill_(default_token)
            out_logprobs[row, k:].fill_(INVALID_LOGPROB)
    return out_tokens, out_logprobs


def _to_npu_tables(tables):
    return {
        key: value.npu() if isinstance(value, torch.Tensor) else value
        for key, value in tables.items()
    }


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "current_step,top_k,temperature_mode",
    [
        (0, 2, "scalar"),
        (0, 4, "row"),
        (1, 3, "none"),
        (1, 4, "row"),
        (2, 3, "scalar"),
        (2, 4, "row"),
    ],
)
def test_rec_constrained_topk_npu(dtype, current_step, top_k, temperature_mode):
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    torch.manual_seed(2026 + current_step + top_k)
    tables = _build_tables()
    rows = 6
    logits = torch.randn([rows, tables["vocab_size"]], dtype=torch.float32)
    logits[:, 0] += 0.25
    logits[:, 7] += 0.5
    sequence_group = torch.tensor(
        [
            [[1, 2, 0], [1, 5, 0], [7, 8, 0]],
            [[11, 0, 0], [11, 13, 0], [3, 9, 0]],
        ],
        dtype=torch.int32,
    )

    if temperature_mode == "none":
        temperatures = torch.ones([1], dtype=torch.float32)
    elif temperature_mode == "scalar":
        temperatures = torch.tensor([0.0], dtype=torch.float32)
    else:
        temperatures = torch.tensor([1.0, 0.7, 1.3, 1.0, 0.9, 1.1], dtype=torch.float32)

    oracle_logits = logits.to(dtype).float()
    expected_tokens, expected_logprobs = _reference_topk(
        oracle_logits, sequence_group, temperatures, current_step, top_k, tables
    )
    npu_tables = _to_npu_tables(tables)
    out_tokens, out_logprobs = custom_ops.rec_constrained_topk_npu(
        logits.to(dtype).npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        current_step,
        top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )

    assert torch.equal(out_tokens.cpu(), expected_tokens)
    assert torch.allclose(out_logprobs.cpu(), expected_logprobs, atol=1e-4, rtol=1e-4)


def test_rec_constrained_topk_invalid_prefix_npu():
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    tables = _build_tables()
    top_k = 3
    logits = torch.randn([2, tables["vocab_size"]], dtype=torch.float32)
    sequence_group = torch.tensor([[[3, 9, 0], [12, 4, 0]]], dtype=torch.int32)
    temperatures = torch.ones([1], dtype=torch.float32)
    expected_tokens, expected_logprobs = _reference_topk(
        logits, sequence_group, temperatures, 2, top_k, tables
    )
    npu_tables = _to_npu_tables(tables)
    out_tokens, out_logprobs = custom_ops.rec_constrained_topk_npu(
        logits.npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        2,
        top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )

    assert torch.equal(out_tokens.cpu(), expected_tokens)
    assert torch.allclose(out_logprobs.cpu(), expected_logprobs, atol=1e-4, rtol=1e-4)


def test_rec_constrained_topk_aligned_large_topk_npu():
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    tables = _build_tables()
    top_k = 256
    rows = 6
    torch.manual_seed(20260508)
    logits = torch.randn([rows, tables["vocab_size"]], dtype=torch.float32)
    sequence_group = torch.tensor(
        [
            [[1, 2, 0], [1, 5, 0], [7, 8, 0]],
            [[11, 0, 0], [11, 13, 0], [3, 9, 0]],
        ],
        dtype=torch.int32,
    )
    temperatures = torch.ones([1], dtype=torch.float32)
    expected_tokens, expected_logprobs = _reference_topk(
        logits, sequence_group, temperatures, 2, top_k, tables
    )
    npu_tables = _to_npu_tables(tables)
    out_tokens, out_logprobs = custom_ops.rec_constrained_topk_npu(
        logits.npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        2,
        top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )

    assert torch.equal(out_tokens.cpu(), expected_tokens)
    assert torch.allclose(out_logprobs.cpu(), expected_logprobs, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("top_k", [384, 512])
@pytest.mark.parametrize("temperature_mode", ["scalar", "row"])
def test_rec_constrained_topk_maxk512_step0_blocksort_npu(top_k, temperature_mode):
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    tables = _build_wide_step0_tables()
    rows = 2
    torch.manual_seed(20260710 + top_k)
    logits = torch.randn([rows, tables["vocab_size"]], dtype=torch.float32)
    sequence_group = torch.zeros([rows, 3], dtype=torch.int32)
    if temperature_mode == "scalar":
        temperatures = torch.tensor([0.75], dtype=torch.float32)
    else:
        temperatures = torch.tensor([0.7, 1.3], dtype=torch.float32)
    expected_tokens, expected_logprobs = _reference_topk(
        logits, sequence_group, temperatures, 0, top_k, tables
    )
    npu_tables = _to_npu_tables(tables)
    out_tokens, out_logprobs = custom_ops.rec_constrained_topk_npu(
        logits.npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        0,
        top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )

    assert torch.equal(out_tokens.cpu(), expected_tokens)
    assert torch.allclose(out_logprobs.cpu(), expected_logprobs, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("top_k", [128, 256, 384, 512])
@pytest.mark.parametrize("streaming_mode", [1, 2])
def test_rec_constrained_topk_streaming_blocksort_bf16_npu(
    monkeypatch, top_k, streaming_mode
):
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    monkeypatch.setenv(
        "XLLM_REC_CONSTRAINED_TOPK_STREAMING_MODE", str(streaming_mode)
    )
    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_P11_MODE", "4")

    candidate_count = 2113 + top_k + streaming_mode
    tables = _build_wide_step0_tables(
        vocab_size=candidate_count + 257, candidate_count=candidate_count
    )
    rows = 2
    torch.manual_seed(20260714 + top_k)
    logits = torch.randn([rows, tables["vocab_size"]], dtype=torch.float32)
    logits_bf16 = logits.to(torch.bfloat16)
    sequence_group = torch.zeros([rows, 3], dtype=torch.int32)
    temperatures = torch.tensor([0.7, 1.3], dtype=torch.float32)
    expected_tokens, expected_logprobs = _reference_topk(
        logits_bf16.float(), sequence_group, temperatures, 0, top_k, tables
    )
    npu_tables = _to_npu_tables(tables)
    out_tokens, out_logprobs = custom_ops.rec_constrained_topk_npu(
        logits_bf16.npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        0,
        top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )

    actual_tokens = out_tokens.cpu()
    actual_logprobs = out_logprobs.cpu()
    assert torch.allclose(
        torch.sort(actual_logprobs, dim=1, descending=True).values,
        torch.sort(expected_logprobs, dim=1, descending=True).values,
        atol=1e-4,
        rtol=1e-4,
    )
    allowed = tables["first_token_ids"].to(torch.long)
    for row in range(rows):
        candidate_logits = logits_bf16.float()[row, allowed] / temperatures[row]
        candidate_logprobs = torch.log_softmax(candidate_logits, dim=0)
        expected_by_token = {
            int(token.item()): float(logprob.item())
            for token, logprob in zip(allowed, candidate_logprobs)
        }
        for idx in range(top_k):
            token = int(actual_tokens[row, idx].item())
            assert token in expected_by_token
            assert actual_logprobs[row, idx].item() == pytest.approx(
                expected_by_token[token], abs=1e-4
            )


def test_rec_constrained_topk_p19_pair_merge_noncontiguous_tokens_npu(monkeypatch):
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_STREAMING_MODE", "2")
    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_P11_MODE", "4")
    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_P19_MODE", "1")
    top_k = 384
    candidate_count = 2603
    vocab_size = 4096
    tables = _build_wide_step0_tables(vocab_size, candidate_count)
    torch.manual_seed(20260719)
    tables["first_token_ids"] = torch.randperm(vocab_size, dtype=torch.int64)[
        :candidate_count
    ].to(torch.int32)
    logits = torch.randn([1, vocab_size], dtype=torch.bfloat16)
    sequence_group = torch.zeros([1, 3], dtype=torch.int32)
    temperatures = torch.tensor([0.73], dtype=torch.float32)
    expected_tokens, expected_logprobs = _reference_topk(
        logits.float(), sequence_group, temperatures, 0, top_k, tables
    )
    npu_tables = _to_npu_tables(tables)
    out_tokens, out_logprobs = custom_ops.rec_constrained_topk_npu(
        logits.npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        0,
        top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )

    actual_tokens = out_tokens.cpu()
    actual_logprobs = out_logprobs.cpu()
    assert torch.allclose(
        torch.sort(actual_logprobs, dim=1, descending=True).values,
        torch.sort(expected_logprobs, dim=1, descending=True).values,
        atol=1e-4,
        rtol=1e-4,
    )
    allowed = tables["first_token_ids"].to(torch.long)
    candidate_logits = logits.float()[0, allowed] / temperatures[0]
    candidate_logprobs = torch.log_softmax(candidate_logits, dim=0)
    expected_by_token = {
        int(token.item()): float(logprob.item())
        for token, logprob in zip(allowed, candidate_logprobs)
    }
    for idx in range(top_k):
        token = int(actual_tokens[0, idx].item())
        assert token in expected_by_token
        assert actual_logprobs[0, idx].item() == pytest.approx(
            expected_by_token[token], abs=1e-4
        )


def test_rec_constrained_topk_streaming_single_block_bf16_npu(monkeypatch):
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_STREAMING_MODE", "2")
    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_P11_MODE", "4")

    candidate_count = 1051
    tables = _build_wide_step0_tables(
        vocab_size=candidate_count + 257, candidate_count=candidate_count
    )
    rows = 2
    top_k = 384
    torch.manual_seed(20260716)
    logits = torch.randn([rows, tables["vocab_size"]], dtype=torch.float32)
    logits_bf16 = logits.to(torch.bfloat16)
    sequence_group = torch.zeros([rows, 3], dtype=torch.int32)
    temperatures = torch.tensor([0.7, 1.3], dtype=torch.float32)
    expected_tokens, expected_logprobs = _reference_topk(
        logits_bf16.float(), sequence_group, temperatures, 0, top_k, tables
    )
    npu_tables = _to_npu_tables(tables)
    out_tokens, out_logprobs = custom_ops.rec_constrained_topk_npu(
        logits_bf16.npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        0,
        top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )

    actual_tokens = out_tokens.cpu()
    actual_logprobs = out_logprobs.cpu()
    assert torch.allclose(
        torch.sort(actual_logprobs, dim=1, descending=True).values,
        torch.sort(expected_logprobs, dim=1, descending=True).values,
        atol=1e-4,
        rtol=1e-4,
    )
    allowed = tables["first_token_ids"].to(torch.long)
    for row in range(rows):
        candidate_logits = logits_bf16.float()[row, allowed] / temperatures[row]
        candidate_logprobs = torch.log_softmax(candidate_logits, dim=0)
        expected_by_token = {
            int(token.item()): float(logprob.item())
            for token, logprob in zip(allowed, candidate_logprobs)
        }
        for idx in range(top_k):
            token = int(actual_tokens[row, idx].item())
            assert token in expected_by_token
            assert actual_logprobs[row, idx].item() == pytest.approx(
                expected_by_token[token], abs=1e-4
            )


@pytest.mark.parametrize("contiguous_scan_mode", [1, 2])
def test_rec_constrained_topk_contiguous_scan_env_npu(monkeypatch, contiguous_scan_mode):
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    monkeypatch.setenv(
        "XLLM_REC_CONSTRAINED_TOPK_CONTIGUOUS_SCAN", str(contiguous_scan_mode)
    )

    tables = _build_wide_step0_tables(vocab_size=1024, candidate_count=641)
    rows = 2
    top_k = 512
    torch.manual_seed(20260712)
    logits = torch.randn([rows, tables["vocab_size"]], dtype=torch.float32)
    logits += torch.arange(tables["vocab_size"], dtype=torch.float32).view(1, -1) * 0.02
    logits_bf16 = logits.to(torch.bfloat16)
    sequence_group = torch.zeros([rows, 3], dtype=torch.int32)
    temperatures = torch.tensor([0.75, 1.25], dtype=torch.float32)
    expected_tokens, expected_logprobs = _reference_topk(
        logits_bf16.float(), sequence_group, temperatures, 0, top_k, tables
    )
    npu_tables = _to_npu_tables(tables)
    out_tokens, out_logprobs = custom_ops.rec_constrained_topk_npu(
        logits_bf16.npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        0,
        top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )

    actual_tokens = out_tokens.cpu()
    actual_logprobs = out_logprobs.cpu()
    assert torch.allclose(
        torch.sort(actual_logprobs, dim=1, descending=True).values,
        torch.sort(expected_logprobs, dim=1, descending=True).values,
        atol=1e-4,
        rtol=1e-4,
    )
    allowed = tables["first_token_ids"].to(torch.long)
    for row in range(rows):
        temp = temperatures[row]
        candidate_logits = logits_bf16.float()[row, allowed] / temp
        candidate_logprobs = torch.log_softmax(candidate_logits, dim=0)
        expected_by_token = {
            int(token.item()): float(logprob.item())
            for token, logprob in zip(allowed, candidate_logprobs)
        }
        for idx in range(top_k):
            token = int(actual_tokens[row, idx].item())
            assert token in expected_by_token
            assert actual_logprobs[row, idx].item() == pytest.approx(
                expected_by_token[token], abs=1e-4
            )


def test_rec_constrained_topk_vector_final_sort_env_npu(monkeypatch):
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_VECTOR_FINAL_SORT", "1")

    tables = _build_wide_step0_tables(vocab_size=1024, candidate_count=641)
    rows = 2
    top_k = 512
    torch.manual_seed(20260713)
    logits = torch.randn([rows, tables["vocab_size"]], dtype=torch.float32)
    logits += torch.arange(tables["vocab_size"], dtype=torch.float32).view(1, -1) * 0.01
    sequence_group = torch.zeros([rows, 3], dtype=torch.int32)
    temperatures = torch.tensor([0.75, 1.25], dtype=torch.float32)
    expected_tokens, expected_logprobs = _reference_topk(
        logits, sequence_group, temperatures, 0, top_k, tables
    )
    npu_tables = _to_npu_tables(tables)
    out_tokens, out_logprobs = custom_ops.rec_constrained_topk_npu(
        logits.npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        0,
        top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )

    assert torch.equal(out_tokens.cpu(), expected_tokens)
    assert torch.allclose(out_logprobs.cpu(), expected_logprobs, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("p11_mode", [2, 4])
def test_rec_constrained_topk_p11_token_staging_env_npu(monkeypatch, p11_mode):
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_P11_MODE", str(p11_mode))

    tables = _build_wide_step0_tables(vocab_size=1024, candidate_count=641)
    rows = 2
    top_k = 512
    torch.manual_seed(20260714)
    logits = torch.randn([rows, tables["vocab_size"]], dtype=torch.float32)
    logits += torch.arange(tables["vocab_size"], dtype=torch.float32).view(1, -1) * 0.01
    logits_bf16 = logits.to(torch.bfloat16)
    sequence_group = torch.zeros([rows, 3], dtype=torch.int32)
    temperatures = torch.tensor([0.75, 1.25], dtype=torch.float32)
    expected_tokens, expected_logprobs = _reference_topk(
        logits_bf16.float(), sequence_group, temperatures, 0, top_k, tables
    )
    npu_tables = _to_npu_tables(tables)
    out_tokens, out_logprobs = custom_ops.rec_constrained_topk_npu(
        logits_bf16.npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        0,
        top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )

    actual_tokens = out_tokens.cpu()
    actual_logprobs = out_logprobs.cpu()
    assert torch.allclose(
        torch.sort(actual_logprobs, dim=1, descending=True).values,
        torch.sort(expected_logprobs, dim=1, descending=True).values,
        atol=1e-4,
        rtol=1e-4,
    )
    allowed = tables["first_token_ids"].to(torch.long)
    for row in range(rows):
        temp = temperatures[row]
        candidate_logits = logits_bf16.float()[row, allowed] / temp
        candidate_logprobs = torch.log_softmax(candidate_logits, dim=0)
        expected_by_token = {
            int(token.item()): float(logprob.item())
            for token, logprob in zip(allowed, candidate_logprobs)
        }
        for idx in range(top_k):
            token = int(actual_tokens[row, idx].item())
            assert token in expected_by_token
            assert actual_logprobs[row, idx].item() == pytest.approx(
                expected_by_token[token], abs=1e-4
            )


def test_rec_constrained_topk_p11_ordered_topk_env_npu(monkeypatch):
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_P11_MODE", "3")

    tables = _build_wide_step0_tables(vocab_size=1024, candidate_count=641)
    rows = 2
    top_k = 512
    torch.manual_seed(20260715)
    logits = torch.randn([rows, tables["vocab_size"]], dtype=torch.float32)
    logits += torch.arange(tables["vocab_size"], dtype=torch.float32).view(1, -1) * 0.01
    sequence_group = torch.zeros([rows, 3], dtype=torch.int32)
    temperatures = torch.tensor([0.75, 1.25], dtype=torch.float32)
    expected_tokens, expected_logprobs = _reference_topk(
        logits, sequence_group, temperatures, 0, top_k, tables
    )
    npu_tables = _to_npu_tables(tables)
    out_tokens, out_logprobs = custom_ops.rec_constrained_topk_npu(
        logits.npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        0,
        top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )

    assert torch.equal(out_tokens.cpu(), expected_tokens)
    assert torch.allclose(out_logprobs.cpu(), expected_logprobs, atol=1e-4, rtol=1e-4)


def test_rec_constrained_topk_small_topk_uses_single_core_debug_npu(monkeypatch):
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_DEBUG_MODE", "1")

    tables = _build_tables()
    rows = 6
    top_k = 3
    logits = torch.randn([rows, tables["vocab_size"]], dtype=torch.float32)
    sequence_group = torch.zeros([2, 3, 3], dtype=torch.int32)
    temperatures = torch.ones([1], dtype=torch.float32)
    npu_tables = _to_npu_tables(tables)
    out_tokens, out_logprobs = custom_ops.rec_constrained_topk_npu(
        logits.npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        0,
        top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )

    expected_tokens = (torch.arange(rows, dtype=torch.int32) + 100).view(rows, 1).repeat(1, top_k)
    expected_logprobs = torch.ones([rows, top_k], dtype=torch.float32)
    assert torch.equal(out_tokens.cpu(), expected_tokens)
    assert torch.equal(out_logprobs.cpu(), expected_logprobs)


def test_rec_constrained_topk_aligned_topk_multicore_debug_npu(monkeypatch):
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_DEBUG_MODE", "1")

    tables = _build_tables()
    rows = 6
    top_k = 8
    logits = torch.randn([rows, tables["vocab_size"]], dtype=torch.float32)
    sequence_group = torch.zeros([2, 3, 3], dtype=torch.int32)
    temperatures = torch.ones([1], dtype=torch.float32)
    npu_tables = _to_npu_tables(tables)
    out_tokens, out_logprobs = custom_ops.rec_constrained_topk_npu(
        logits.npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        0,
        top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )

    expected_tokens = (torch.arange(rows, dtype=torch.int32) + 100).view(rows, 1).repeat(1, top_k)
    expected_logprobs = torch.tensor(
        [6.0, 1006.0, 2006.0, 3006.0, 4006.0, 5006.0], dtype=torch.float32
    ).view(rows, 1).repeat(1, top_k)
    assert torch.equal(out_tokens.cpu(), expected_tokens)
    assert torch.equal(out_logprobs.cpu(), expected_logprobs)


def test_rec_constrained_topk_stage_attribution_debug_npu(monkeypatch):
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_DEBUG_MODE", "2")

    tables = _build_tables()
    rows = 2
    top_k = 32
    torch.manual_seed(20260711)
    logits = torch.randn([rows, tables["vocab_size"]], dtype=torch.float32)
    sequence_group = torch.zeros([rows, 3], dtype=torch.int32)
    temperatures = torch.ones([1], dtype=torch.float32)
    npu_tables = _to_npu_tables(tables)
    out_tokens, out_logprobs = custom_ops.rec_constrained_topk_npu(
        logits.npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        0,
        top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )

    diag_tokens = out_tokens.cpu()
    diag_timings = out_logprobs.cpu()
    assert torch.equal(diag_tokens[:, 0], torch.full([rows], 20260711, dtype=torch.int32))
    assert torch.equal(diag_tokens[:, 1], torch.arange(rows, dtype=torch.int32))
    assert torch.equal(diag_tokens[:, 2], torch.zeros([rows], dtype=torch.int32))
    assert torch.equal(diag_tokens[:, 3], torch.zeros([rows], dtype=torch.int32))
    assert torch.equal(
        diag_tokens[:, 4],
        torch.full([rows], tables["first_token_ids"].numel(), dtype=torch.int32),
    )
    assert torch.equal(diag_tokens[:, 5], torch.full([rows], top_k, dtype=torch.int32))
    assert torch.equal(diag_tokens[:, 6], torch.zeros([rows], dtype=torch.int32))
    assert torch.all(diag_tokens[:, 7] >= 0)
    assert torch.all(diag_tokens[:, 8] >= 1)
    assert torch.equal(
        diag_tokens[:, 9],
        torch.full([rows], tables["first_token_ids"].numel(), dtype=torch.int32),
    )
    assert torch.equal(
        diag_tokens[:, 15], torch.full([rows], 4, dtype=torch.int32)
    )
    assert torch.equal(diag_tokens[:, 16], torch.ones([rows], dtype=torch.int32))
    assert torch.equal(
        diag_tokens[:, 17], torch.full([rows], 2, dtype=torch.int32)
    )
    assert torch.equal(diag_timings[:, 10], torch.zeros([rows], dtype=torch.float32))
    assert torch.all(diag_timings[:, :29] >= 0)

    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_P11_MODE", "0")
    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_P19_MODE", "0")
    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_STREAMING_MODE", "0")
    rollback_top_k = top_k - 1
    rollback_tokens, _ = custom_ops.rec_constrained_topk_npu(
        logits.npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        0,
        rollback_top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )
    rollback_diag_tokens = rollback_tokens.cpu()
    assert torch.equal(
        rollback_diag_tokens[:, 15], torch.zeros([rows], dtype=torch.int32)
    )
    assert torch.equal(
        rollback_diag_tokens[:, 16], torch.zeros([rows], dtype=torch.int32)
    )
    assert torch.equal(
        rollback_diag_tokens[:, 17], torch.zeros([rows], dtype=torch.int32)
    )


@pytest.mark.parametrize("streaming_mode", [1, 2])
def test_rec_constrained_topk_streaming_stage_attribution_npu(
    monkeypatch, streaming_mode
):
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_DEBUG_MODE", "2")
    monkeypatch.setenv(
        "XLLM_REC_CONSTRAINED_TOPK_STREAMING_MODE", str(streaming_mode)
    )
    monkeypatch.setenv("XLLM_REC_CONSTRAINED_TOPK_P11_MODE", "4")

    tables = _build_wide_step0_tables(
        vocab_size=2753 + streaming_mode, candidate_count=2497 + streaming_mode
    )
    rows = 1
    top_k = 384
    torch.manual_seed(20260715)
    logits = torch.randn([rows, tables["vocab_size"]], dtype=torch.bfloat16)
    sequence_group = torch.zeros([rows, 3], dtype=torch.int32)
    temperatures = torch.tensor([0.8], dtype=torch.float32)
    npu_tables = _to_npu_tables(tables)
    out_tokens, out_logprobs = custom_ops.rec_constrained_topk_npu(
        logits.npu(),
        sequence_group.npu(),
        npu_tables["first_token_ids"],
        npu_tables["prefix1_offsets"],
        npu_tables["prefix1_values"],
        npu_tables["prefix1_pair_keys"],
        npu_tables["prefix2_value_offsets"],
        npu_tables["prefix2_values"],
        temperatures.npu(),
        0,
        top_k,
        tables["max_prefix1_degree"],
        tables["max_prefix2_degree"],
    )

    diag_tokens = out_tokens.cpu()
    diag_timings = out_logprobs.cpu()
    assert diag_tokens[0, 6].item() == 1
    assert diag_tokens[0, 17].item() == streaming_mode
    assert torch.all(diag_timings[:, 25:27] > 0)
    assert torch.all(diag_timings[:, 27:29] >= 0)
    if streaming_mode == 1:
        assert torch.all(diag_timings[:, 28] > 0)
    else:
        assert diag_tokens[0, 16].item() == 1
        assert torch.all(diag_timings[:, 23] >= 0)
    if streaming_mode == 2:
        assert torch.all(diag_timings[:, 29:31] >= 0)
    else:
        assert torch.equal(
            diag_timings[:, 29:31], torch.zeros_like(diag_timings[:, 29:31])
        )
