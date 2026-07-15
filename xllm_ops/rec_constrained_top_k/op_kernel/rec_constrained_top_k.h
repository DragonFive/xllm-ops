/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "kernel_operator.h"

namespace kernels {

constexpr int32_t kRecConstrainedTopKMaxK = 512;
constexpr float kRecConstrainedTopKInvalidLogProb = -1.0e20f;
constexpr int32_t kRecConstrainedTopKMathBufferBytes = 32;
constexpr int32_t kRecConstrainedTopKExpBlockElements = 128;
constexpr int32_t kRecConstrainedTopKExpBufferBytes =
    kRecConstrainedTopKExpBlockElements * static_cast<int32_t>(sizeof(float));
constexpr int32_t kRecConstrainedTopKSortBlockElements = 2048;
constexpr int32_t kRecConstrainedTopKSortThreshold = 512;
constexpr int32_t kRecConstrainedTopKSortPairBufferElements =
    kRecConstrainedTopKSortBlockElements * 4;
constexpr int32_t kRecConstrainedTopKSortPairBufferBytes =
    kRecConstrainedTopKSortPairBufferElements *
    static_cast<int32_t>(sizeof(float));
constexpr int32_t kRecConstrainedTopKGlobalPairBufferElements =
    kRecConstrainedTopKMaxK * 4;
constexpr int32_t kRecConstrainedTopKGlobalPairBufferBytes =
    kRecConstrainedTopKGlobalPairBufferElements *
    static_cast<int32_t>(sizeof(float));
constexpr int32_t kRecConstrainedTopKSortIndexBufferBytes =
    kRecConstrainedTopKSortBlockElements *
    static_cast<int32_t>(sizeof(int32_t));
constexpr int32_t kRecConstrainedTopKSortTmpBufferBytes =
    kRecConstrainedTopKSortBlockElements * 3 *
    static_cast<int32_t>(sizeof(float));
constexpr int32_t kRecConstrainedTopKFloatElementsPerBlock =
    32 / static_cast<int32_t>(sizeof(float));
constexpr int32_t kRecConstrainedTopKFloatElementsPerRepeat =
    256 / static_cast<int32_t>(sizeof(float));
constexpr int32_t kRecConstrainedTopKReduceTmpElements =
    (((kRecConstrainedTopKExpBlockElements +
       kRecConstrainedTopKFloatElementsPerRepeat - 1) /
          kRecConstrainedTopKFloatElementsPerRepeat +
      kRecConstrainedTopKFloatElementsPerBlock - 1) /
     kRecConstrainedTopKFloatElementsPerBlock) *
    kRecConstrainedTopKFloatElementsPerBlock;
constexpr int32_t kRecConstrainedTopKReduceTmpBufferBytes =
    kRecConstrainedTopKReduceTmpElements * static_cast<int32_t>(sizeof(float));
constexpr int32_t kRecConstrainedTopKLoadBf16Elements = 64;
constexpr int32_t kRecConstrainedTopKMaxLoadBf16Elements = 128;
constexpr int32_t kRecConstrainedTopKContiguousLoadElements = 512;
constexpr int32_t kRecConstrainedTopKContiguousBf16BufferBytes =
    kRecConstrainedTopKContiguousLoadElements *
    static_cast<int32_t>(sizeof(bfloat16_t));
constexpr int32_t kRecConstrainedTopKCandidateTokenChunkElements = 512;
constexpr int32_t kRecConstrainedTopKCandidateTokenChunkBufferBytes =
    kRecConstrainedTopKCandidateTokenChunkElements *
    static_cast<int32_t>(sizeof(int32_t));
constexpr int32_t kRecConstrainedTopKBlockTokenBufferCount = 2;
constexpr int32_t kRecConstrainedTopKBlockTokenBufferBytes =
    kRecConstrainedTopKBlockTokenBufferCount *
    kRecConstrainedTopKSortBlockElements *
    static_cast<int32_t>(sizeof(int32_t));
constexpr int32_t kRecConstrainedTopKTokenStagingThreshold =
    kRecConstrainedTopKCandidateTokenChunkElements;
constexpr int32_t kRecConstrainedTopKOutputTokenBufferBytes =
    kRecConstrainedTopKMaxK * static_cast<int32_t>(sizeof(int32_t));
constexpr int32_t kRecConstrainedTopKOutputLogprobBufferBytes =
    kRecConstrainedTopKMaxK * static_cast<int32_t>(sizeof(float));
constexpr int64_t kRecConstrainedTopKCyclesPerUs = 50;
constexpr int32_t kRecConstrainedTopKAttributionMagic = 20260711;

__aicore__ inline int64_t RecConstrainedTopKTimestampUs() {
  return static_cast<int64_t>(AscendC::GetSystemCycle() /
                              kRecConstrainedTopKCyclesPerUs);
}

__aicore__ inline int32_t RecConstrainedTopKAlign32(int32_t value) {
  return ((value + 31) / 32) * 32;
}

__aicore__ inline int32_t RecConstrainedTopKReduceTmpElementsForCount(
    int32_t count) {
  return (((count + kRecConstrainedTopKFloatElementsPerRepeat - 1) /
               kRecConstrainedTopKFloatElementsPerRepeat +
           kRecConstrainedTopKFloatElementsPerBlock - 1) /
          kRecConstrainedTopKFloatElementsPerBlock) *
         kRecConstrainedTopKFloatElementsPerBlock;
}

template <typename LogitType>
class RecConstrainedTopKKernel {
 public:
  __aicore__ inline RecConstrainedTopKKernel() {}

  __aicore__ inline void Init(GM_ADDR logits,
                              GM_ADDR sequence_group,
                              GM_ADDR first_token_ids,
                              GM_ADDR prefix1_offsets,
                              GM_ADDR prefix1_values,
                              GM_ADDR prefix1_pair_keys,
                              GM_ADDR prefix2_value_offsets,
                              GM_ADDR prefix2_values,
                              GM_ADDR temperatures,
                              GM_ADDR out_tokens,
                              GM_ADDR out_logprobs,
                              const RecConstrainedTopKTilingData* tiling_data,
                              AscendC::TPipe* pipe) {
    tiling_data_ = *tiling_data;
    pipe_ = pipe;
    pipe_->InitBuffer(math_buf_, kRecConstrainedTopKMathBufferBytes);
    pipe_->InitBuffer(exp_buf_, kRecConstrainedTopKExpBufferBytes);
    pipe_->InitBuffer(reduce_tmp_buf_, kRecConstrainedTopKReduceTmpBufferBytes);
    const int32_t logit_cache_elements = LogitCacheBufferElements();
    pipe_->InitBuffer(
        logit_bf16_buf_,
        logit_cache_elements * static_cast<int32_t>(sizeof(bfloat16_t)));
    pipe_->InitBuffer(
        logit_float_buf_,
        logit_cache_elements * static_cast<int32_t>(sizeof(float)));
    if (tiling_data_.enable_contiguous_scan != 0) {
      pipe_->InitBuffer(contiguous_logit_bf16_buf_,
                        kRecConstrainedTopKContiguousBf16BufferBytes);
    }
    pipe_->InitBuffer(candidate_token_chunk_buf_,
                      kRecConstrainedTopKCandidateTokenChunkBufferBytes);
    if (tiling_data_.streaming_mode == 2) {
      pipe_->InitBuffer(block_token_buf_,
                        kRecConstrainedTopKBlockTokenBufferBytes);
    }
    pipe_->InitBuffer(
        candidate_logit_buf_,
        static_cast<uint32_t>(CandidateLogitBufferElements() *
                              static_cast<int32_t>(sizeof(float))));
    pipe_->InitBuffer(sort_pair_buf_, kRecConstrainedTopKSortPairBufferBytes);
    if (tiling_data_.p19_mode == 1) {
      pipe_->InitBuffer(global_pair_buf_,
                        kRecConstrainedTopKGlobalPairBufferBytes);
    }
    pipe_->InitBuffer(sort_index_buf_, kRecConstrainedTopKSortIndexBufferBytes);
    pipe_->InitBuffer(sort_tmp_buf_, kRecConstrainedTopKSortTmpBufferBytes);
    pipe_->InitBuffer(output_token_buf_,
                      kRecConstrainedTopKOutputTokenBufferBytes);
    pipe_->InitBuffer(output_logprob_buf_,
                      kRecConstrainedTopKOutputLogprobBufferBytes);
    logits_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ LogitType*>(logits));
    sequence_group_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(sequence_group));
    first_token_ids_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(first_token_ids));
    prefix1_offsets_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(prefix1_offsets));
    prefix1_values_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(prefix1_values));
    prefix1_pair_keys_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int64_t*>(prefix1_pair_keys));
    prefix2_value_offsets_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(prefix2_value_offsets));
    prefix2_values_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(prefix2_values));
    temperatures_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(temperatures));
    out_tokens_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(out_tokens));
    out_logprobs_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(out_logprobs));
  }

  __aicore__ inline void Process() {
    const int32_t core_id = static_cast<int32_t>(AscendC::GetBlockIdx());
    const int32_t core_num = static_cast<int32_t>(AscendC::GetBlockNum());
    const bool use_block_token_pipeline =
        tiling_data_.streaming_mode == 2 &&
        tiling_data_.max_candidate_count >
            kRecConstrainedTopKSortBlockElements &&
        (tiling_data_.p11_mode == 3 || tiling_data_.p11_mode == 4);
    if (use_block_token_pipeline) {
      for (int32_t index = 0; index < kRecConstrainedTopKBlockTokenBufferCount;
           ++index) {
        block_token_events_[index] = static_cast<event_t>(
            GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_S>());
      }
    }
    for (int32_t row = core_id; row < tiling_data_.num_rows; row += core_num) {
      if (tiling_data_.debug_mode == 1) {
        WriteDebugRow(row, core_id, core_num);
        continue;
      }
      ProcessRow(row, core_id, core_num);
    }
    if (use_block_token_pipeline) {
      for (int32_t index = 0; index < kRecConstrainedTopKBlockTokenBufferCount;
           ++index) {
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_S>(
            block_token_events_[index]);
      }
    }
  }

 private:
  struct StageAttribution {
    int64_t total_us = 0;
    int64_t resolve_us = 0;
    int64_t temperature_us = 0;
    int64_t candidate_scan_us = 0;
    int64_t candidate_token_read_us = 0;
    int64_t candidate_token_copy_us = 0;
    int64_t candidate_token_ub_read_us = 0;
    int64_t logit_read_us = 0;
    int64_t score_store_us = 0;
    int64_t scan_topk_update_us = 0;
    int64_t logsumexp_us = 0;
    int64_t block_sort_us = 0;
    int64_t merge_us = 0;
    int64_t final_heap_ln_us = 0;
    int64_t final_order_us = 0;
    int64_t final_ln_us = 0;
    int64_t output_fill_us = 0;
    int64_t flush_us = 0;
    int64_t token_materialize_us = 0;
    int64_t compact_copy_us = 0;
    int64_t global_sort_us = 0;
    int64_t merge_token_lookup_us = 0;
    int64_t stream_block_scan_us = 0;
    int64_t stream_block_lse_us = 0;
    int64_t stream_block_sort_us = 0;
    int64_t stream_block_merge_us = 0;
    int64_t stream_token_prefetch_issue_us = 0;
    int64_t stream_token_prefetch_wait_us = 0;
    int64_t pair_payload_prepare_us = 0;
    int64_t vector_pair_merge_us = 0;
    int64_t final_pair_extract_us = 0;
  };

  __aicore__ inline float ReadLogit(
      int32_t row,
      int32_t token,
      int32_t& cached_block_offset,
      AscendC::LocalTensor<bfloat16_t>& cached_logit_bf16,
      AscendC::LocalTensor<float>& cached_logit_float) {
    if constexpr (AscendC::IsSameType<LogitType, bfloat16_t>::value) {
      if (tiling_data_.load_bf16_mode == 0) {
        return ReadLogitBf16Fixed<kRecConstrainedTopKLoadBf16Elements>(
            row,
            token,
            cached_block_offset,
            cached_logit_bf16,
            cached_logit_float);
      }
      if (tiling_data_.load_bf16_mode == 4) {
        if (tiling_data_.top_k == 384) {
          return ReadLogitBf16Fixed<32>(row,
                                        token,
                                        cached_block_offset,
                                        cached_logit_bf16,
                                        cached_logit_float);
        }
        return ReadLogitBf16Fixed<kRecConstrainedTopKLoadBf16Elements>(
            row,
            token,
            cached_block_offset,
            cached_logit_bf16,
            cached_logit_float);
      }
      return ReadLogitBf16Dynamic(row,
                                  token,
                                  cached_block_offset,
                                  cached_logit_bf16,
                                  cached_logit_float,
                                  ActiveLogitLoadElements());
    }
    return static_cast<float>(logits_gm_.GetValue(
        static_cast<int64_t>(row) * tiling_data_.vocab_size + token));
  }

  template <int32_t LoadElements>
  __aicore__ inline float ReadLogitBf16Fixed(
      int32_t row,
      int32_t token,
      int32_t& cached_block_offset,
      AscendC::LocalTensor<bfloat16_t>& cached_logit_bf16,
      AscendC::LocalTensor<float>& cached_logit_float) {
    const int32_t block_offset = token / LoadElements * LoadElements;
    const int32_t max_block_offset = tiling_data_.vocab_size - LoadElements;
    const int32_t safe_block_offset =
        max_block_offset > 0 && block_offset > max_block_offset
            ? max_block_offset
            : block_offset;
    const int32_t local_offset = token - safe_block_offset;
    const int32_t copy_len =
        tiling_data_.vocab_size - safe_block_offset < LoadElements
            ? tiling_data_.vocab_size - safe_block_offset
            : LoadElements;
    if (cached_block_offset != safe_block_offset) {
      cached_block_offset = safe_block_offset;
      AscendC::DataCopyExtParams copy_params;
      copy_params.blockCount = 1;
      copy_params.blockLen = static_cast<uint32_t>(
          copy_len * static_cast<int32_t>(sizeof(bfloat16_t)));
      copy_params.srcStride = 0;
      copy_params.dstStride = 0;
      copy_params.rsv = 0;
      AscendC::DataCopyPadExtParams<bfloat16_t> pad_params{false, 0, 0, 0};
      AscendC::DataCopyPad(
          cached_logit_bf16,
          logits_gm_[static_cast<int64_t>(row) * tiling_data_.vocab_size +
                     safe_block_offset],
          copy_params,
          pad_params);
      event_t event_mte2_v = static_cast<event_t>(
          GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_mte2_v);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_mte2_v);
      AscendC::Cast(cached_logit_float,
                    cached_logit_bf16,
                    AscendC::RoundMode::CAST_NONE,
                    copy_len);
      AscendC::PipeBarrier<PIPE_V>();
    }
    return cached_logit_float.GetValue(local_offset);
  }

  __aicore__ inline float ReadLogitBf16Dynamic(
      int32_t row,
      int32_t token,
      int32_t& cached_block_offset,
      AscendC::LocalTensor<bfloat16_t>& cached_logit_bf16,
      AscendC::LocalTensor<float>& cached_logit_float,
      int32_t load_elements) {
    const int32_t block_offset = token / load_elements * load_elements;
    const int32_t max_block_offset = tiling_data_.vocab_size - load_elements;
    const int32_t safe_block_offset =
        max_block_offset > 0 && block_offset > max_block_offset
            ? max_block_offset
            : block_offset;
    const int32_t local_offset = token - safe_block_offset;
    const int32_t copy_len =
        tiling_data_.vocab_size - safe_block_offset < load_elements
            ? tiling_data_.vocab_size - safe_block_offset
            : load_elements;
    if (cached_block_offset != safe_block_offset) {
      cached_block_offset = safe_block_offset;
      AscendC::DataCopyExtParams copy_params;
      copy_params.blockCount = 1;
      copy_params.blockLen = static_cast<uint32_t>(
          copy_len * static_cast<int32_t>(sizeof(bfloat16_t)));
      copy_params.srcStride = 0;
      copy_params.dstStride = 0;
      copy_params.rsv = 0;
      AscendC::DataCopyPadExtParams<bfloat16_t> pad_params{false, 0, 0, 0};
      AscendC::DataCopyPad(
          cached_logit_bf16,
          logits_gm_[static_cast<int64_t>(row) * tiling_data_.vocab_size +
                     safe_block_offset],
          copy_params,
          pad_params);
      event_t event_mte2_v = static_cast<event_t>(
          GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_mte2_v);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_mte2_v);
      AscendC::Cast(cached_logit_float,
                    cached_logit_bf16,
                    AscendC::RoundMode::CAST_NONE,
                    copy_len);
      AscendC::PipeBarrier<PIPE_V>();
    }
    return cached_logit_float.GetValue(local_offset);
  }

  __aicore__ inline float ReadTemperature(int32_t row) const {
    if (tiling_data_.temperature_count == tiling_data_.num_rows) {
      return temperatures_gm_.GetValue(row);
    }
    return temperatures_gm_.GetValue(0);
  }

  __aicore__ inline int32_t ActiveLogitLoadElements() const {
    if (tiling_data_.load_bf16_mode == 1) {
      return tiling_data_.current_step >= 2
                 ? 32
                 : kRecConstrainedTopKMaxLoadBf16Elements;
    }
    if (tiling_data_.load_bf16_mode == 2) {
      return 32;
    }
    if (tiling_data_.load_bf16_mode == 3) {
      return kRecConstrainedTopKMaxLoadBf16Elements;
    }
    return kRecConstrainedTopKLoadBf16Elements;
  }

  __aicore__ inline int32_t LogitCacheBufferElements() const {
    return tiling_data_.load_bf16_mode == 1 || tiling_data_.load_bf16_mode == 3
               ? kRecConstrainedTopKMaxLoadBf16Elements
               : kRecConstrainedTopKLoadBf16Elements;
  }

  __aicore__ inline int32_t CandidateLogitBufferElements() const {
    int32_t elements =
        RecConstrainedTopKAlign32(tiling_data_.max_candidate_count);
    if (tiling_data_.streaming_mode != 0 &&
        elements > kRecConstrainedTopKSortBlockElements) {
      elements = kRecConstrainedTopKSortBlockElements;
    }
    return elements;
  }

  __aicore__ inline float ExpScalar(float value) {
    AscendC::LocalTensor<float> math_tensor = math_buf_.Get<float>();
    // Scalar SetValue writes to UBUF. Before the vector Exp reads it, insert an
    // S_V sync so the vector unit sees the up-to-date data (otherwise it may
    // read stale/uninitialized UBUF and return 0 on some SoC, e.g. ascend950).
    math_tensor.SetValue(0, value);
    event_t ev_s_v_e = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(ev_s_v_e);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(ev_s_v_e);
    AscendC::Exp(math_tensor, math_tensor, 1);
    // Vector Exp wrote to UBUF. Before the scalar GetValue reads it, insert a
    // V_S sync so the scalar unit observes the vector result.
    event_t ev_v_s_e = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(ev_v_s_e);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(ev_v_s_e);
    return math_tensor.GetValue(0);
  }

  __aicore__ inline float FlushExpBlock(AscendC::LocalTensor<float>& exp_values,
                                        int32_t exp_count) {
    if (exp_count <= 0) {
      return 0.0f;
    }
    // exp_values were filled via scalar SetValue in UBUF. Add an S_V sync so the
    // vector Exp reads the freshly written data instead of stale UBUF.
    event_t ev_s_v_f = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(ev_s_v_f);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(ev_s_v_f);
    AscendC::Exp(exp_values, exp_values, exp_count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::LocalTensor<float> reduce_sum = math_buf_.Get<float>();
    AscendC::LocalTensor<float> reduce_tmp = reduce_tmp_buf_.Get<float>();
    AscendC::ReduceSum(reduce_sum, exp_values, reduce_tmp, exp_count);
    AscendC::PipeBarrier<PIPE_V>();
    // ReduceSum writes UBUF through the vector unit. Synchronize before the
    // scalar unit reads the reduced value.
    event_t ev_v_s_f = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(ev_v_s_f);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(ev_v_s_f);
    return reduce_sum.GetValue(0);
  }

  __aicore__ inline float SumExpFromCandidateLogits(
      AscendC::LocalTensor<float>& candidate_logits,
      int32_t degree,
      float max_value) {
    float sum_exp = 0.0f;
    AscendC::LocalTensor<float> exp_values =
        exp_buf_.Get<float>(kRecConstrainedTopKExpBlockElements);
    for (int32_t offset = 0; offset < degree;
         offset += kRecConstrainedTopKExpBlockElements) {
      const int32_t remaining = degree - offset;
      const int32_t exp_count = remaining < kRecConstrainedTopKExpBlockElements
                                    ? remaining
                                    : kRecConstrainedTopKExpBlockElements;
      AscendC::Adds(
          exp_values, candidate_logits[offset], -max_value, exp_count);
      AscendC::PipeBarrier<PIPE_V>();
      sum_exp += FlushExpBlock(exp_values, exp_count);
    }
    return sum_exp;
  }

  __aicore__ inline float LnScalar(float value) {
    AscendC::LocalTensor<float> math_tensor = math_buf_.Get<float>();
    // Same scalar-write/vector-read hazard as ExpScalar: add an S_V sync before
    // the vector Ln so it reads the value just written by SetValue.
    math_tensor.SetValue(0, value);
    event_t ev_s_v_l = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(ev_s_v_l);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(ev_s_v_l);
    AscendC::Ln(math_tensor, math_tensor, 1);
    // Add a V_S sync so the scalar GetValue observes the vector Ln result.
    event_t ev_v_s_l = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(ev_v_s_l);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(ev_v_s_l);
    return math_tensor.GetValue(0);
  }

  __aicore__ inline void ResolveRange(int32_t row,
                                      int32_t& begin,
                                      int32_t& degree,
                                      int32_t& table_kind) const {
    begin = 0;
    degree = 0;
    table_kind = 0;
    if (tiling_data_.current_step == 0) {
      degree = tiling_data_.first_token_count;
      table_kind = 0;
      return;
    }

    table_kind = tiling_data_.current_step == 1 ? 1 : 2;
    const int32_t sequence_base = row * tiling_data_.sequence_stride;
    const int32_t t0 = sequence_group_gm_.GetValue(sequence_base);
    if (t0 < 0 || t0 + 1 >= tiling_data_.vocab_size) {
      return;
    }

    if (tiling_data_.current_step == 1) {
      begin = prefix1_offsets_gm_.GetValue(t0);
      const int32_t end = prefix1_offsets_gm_.GetValue(t0 + 1);
      degree = end - begin;
      return;
    }

    const int32_t t1 = sequence_group_gm_.GetValue(sequence_base + 1);
    if (t1 < 0 || t1 >= tiling_data_.vocab_size) {
      return;
    }
    const int64_t query_key =
        static_cast<int64_t>(t0) * tiling_data_.vocab_size +
        static_cast<int64_t>(t1);
    int32_t left = 0;
    int32_t right = tiling_data_.prefix1_pair_count;
    while (left < right) {
      const int32_t mid = left + (right - left) / 2;
      const int64_t key = prefix1_pair_keys_gm_.GetValue(mid);
      if (key < query_key) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    if (left >= tiling_data_.prefix1_pair_count ||
        prefix1_pair_keys_gm_.GetValue(left) != query_key) {
      return;
    }
    begin = prefix2_value_offsets_gm_.GetValue(left);
    const int32_t end = prefix2_value_offsets_gm_.GetValue(left + 1);
    degree = end - begin;
  }

  __aicore__ inline int32_t ReadCandidateToken(int32_t table_kind,
                                               int32_t index) const {
    if (table_kind == 0) {
      return first_token_ids_gm_.GetValue(index);
    }
    if (table_kind == 1) {
      return prefix1_values_gm_.GetValue(index);
    }
    return prefix2_values_gm_.GetValue(index);
  }

  __aicore__ inline void WaitMte2ToVector() {
    event_t event_mte2_v = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_mte2_v);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_mte2_v);
  }

  __aicore__ inline void WaitMte2ToScalar() const {
    event_t event_mte2_s = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event_mte2_s);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event_mte2_s);
  }

  __aicore__ inline void WaitVectorToMte2() {
    event_t event_v_mte2 = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_v_mte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_v_mte2);
  }

  __aicore__ inline void CopyCandidateTokenChunk(
      AscendC::LocalTensor<int32_t>& candidate_tokens,
      int32_t table_kind,
      int32_t begin,
      int32_t count) const {
    AscendC::DataCopyExtParams copy_params;
    copy_params.blockCount = 1;
    copy_params.blockLen =
        static_cast<uint32_t>(count * static_cast<int32_t>(sizeof(int32_t)));
    copy_params.srcStride = 0;
    copy_params.dstStride = 0;
    copy_params.rsv = 0;
    AscendC::DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
    if (table_kind == 0) {
      AscendC::DataCopyPad(candidate_tokens,
                           first_token_ids_gm_[begin],
                           copy_params,
                           pad_params);
    } else if (table_kind == 1) {
      AscendC::DataCopyPad(
          candidate_tokens, prefix1_values_gm_[begin], copy_params, pad_params);
    } else {
      AscendC::DataCopyPad(
          candidate_tokens, prefix2_values_gm_[begin], copy_params, pad_params);
    }
  }

  __aicore__ inline void PrefetchCandidateTokenBlock(
      AscendC::LocalTensor<int32_t>& candidate_tokens,
      int32_t table_kind,
      int32_t begin,
      int32_t count,
      event_t event_id,
      bool attribution_mode,
      StageAttribution& attribution) const {
    const int64_t issue_start_us =
        attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
    CopyCandidateTokenChunk(candidate_tokens, table_kind, begin, count);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event_id);
    if (attribution_mode) {
      const int64_t elapsed_us =
          RecConstrainedTopKTimestampUs() - issue_start_us;
      attribution.candidate_token_copy_us += elapsed_us;
      attribution.stream_token_prefetch_issue_us += elapsed_us;
    }
  }

  __aicore__ inline void WaitCandidateTokenBlock(
      event_t event_id,
      bool attribution_mode,
      StageAttribution& attribution) const {
    const int64_t wait_start_us =
        attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event_id);
    if (attribution_mode) {
      const int64_t elapsed_us =
          RecConstrainedTopKTimestampUs() - wait_start_us;
      attribution.candidate_token_copy_us += elapsed_us;
      attribution.stream_token_prefetch_wait_us += elapsed_us;
    }
  }

  __aicore__ inline int32_t ReadMergeCandidateToken(
      int32_t table_kind,
      int32_t begin,
      int32_t candidate_index,
      bool attribution_mode,
      StageAttribution& attribution) const {
    const int64_t lookup_start_us =
        attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
    const int32_t token =
        ReadCandidateToken(table_kind, begin + candidate_index);
    if (attribution_mode) {
      attribution.merge_token_lookup_us +=
          RecConstrainedTopKTimestampUs() - lookup_start_us;
    }
    return token;
  }

  __aicore__ inline int32_t ReadPrefetchedMergeCandidateToken(
      int32_t candidate_index,
      int32_t block_offset,
      int32_t block_count,
      AscendC::LocalTensor<int32_t>& block_tokens,
      bool attribution_mode,
      StageAttribution& attribution) const {
    const int64_t lookup_start_us =
        attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
    const int32_t local_index = candidate_index - block_offset;
    const int32_t token = local_index >= 0 && local_index < block_count
                              ? block_tokens.GetValue(local_index)
                              : -1;
    if (attribution_mode) {
      attribution.merge_token_lookup_us +=
          RecConstrainedTopKTimestampUs() - lookup_start_us;
    }
    return token;
  }

  __aicore__ inline bool TryResolveContiguousCandidateRange(
      int32_t begin,
      int32_t degree,
      int32_t table_kind,
      int32_t& first_token,
      bool attribution_mode,
      StageAttribution& attribution) {
    if (degree <= 0) {
      return false;
    }
    int64_t stage_start_us =
        attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
    first_token = ReadCandidateToken(table_kind, begin);
    const int32_t last_token =
        degree == 1 ? first_token
                    : ReadCandidateToken(table_kind, begin + degree - 1);
    if (first_token < 0 || first_token >= tiling_data_.vocab_size ||
        first_token + degree > tiling_data_.vocab_size ||
        last_token != first_token + degree - 1) {
      if (attribution_mode) {
        attribution.candidate_token_read_us +=
            RecConstrainedTopKTimestampUs() - stage_start_us;
      }
      return false;
    }
    if (tiling_data_.enable_contiguous_scan == 1 && degree > 2) {
      AscendC::LocalTensor<int32_t> candidate_tokens =
          candidate_token_chunk_buf_.Get<int32_t>(
              kRecConstrainedTopKCandidateTokenChunkElements);
      for (int32_t offset = 1; offset < degree - 1;
           offset += kRecConstrainedTopKCandidateTokenChunkElements) {
        const int32_t remaining = degree - 1 - offset;
        const int32_t copy_count =
            remaining < kRecConstrainedTopKCandidateTokenChunkElements
                ? remaining
                : kRecConstrainedTopKCandidateTokenChunkElements;
        CopyCandidateTokenChunk(
            candidate_tokens, table_kind, begin + offset, copy_count);
        WaitMte2ToScalar();
        for (int32_t idx = 0; idx < copy_count; ++idx) {
          const int32_t token = candidate_tokens.GetValue(idx);
          if (token != first_token + offset + idx) {
            if (attribution_mode) {
              attribution.candidate_token_read_us +=
                  RecConstrainedTopKTimestampUs() - stage_start_us;
            }
            return false;
          }
        }
      }
    }
    if (attribution_mode) {
      attribution.candidate_token_read_us +=
          RecConstrainedTopKTimestampUs() - stage_start_us;
    }
    return true;
  }

  __aicore__ inline bool TryLoadContiguousCandidateScores(
      int32_t row,
      int32_t begin,
      int32_t degree,
      int32_t table_kind,
      float temperature,
      AscendC::LocalTensor<float>& candidate_logits,
      int32_t& cached_block_offset,
      AscendC::LocalTensor<bfloat16_t>& cached_logit_bf16,
      AscendC::LocalTensor<float>& cached_logit_float,
      float& max_value,
      bool attribution_mode,
      StageAttribution& attribution) {
    if (tiling_data_.enable_contiguous_scan == 0) {
      return false;
    }
    if (!AscendC::IsSameType<LogitType, bfloat16_t>::value &&
        !AscendC::IsSameType<LogitType, float>::value) {
      return false;
    }

    int32_t first_token = 0;
    if (!TryResolveContiguousCandidateRange(begin,
                                            degree,
                                            table_kind,
                                            first_token,
                                            attribution_mode,
                                            attribution)) {
      return false;
    }

    const int64_t row_base =
        static_cast<int64_t>(row) * tiling_data_.vocab_size + first_token;
    const float inv_temperature = 1.0f / temperature;
    AscendC::LocalTensor<bfloat16_t> contiguous_logit_bf16 =
        contiguous_logit_bf16_buf_.Get<bfloat16_t>(
            kRecConstrainedTopKContiguousLoadElements);
    for (int32_t offset = 0; offset < degree;
         offset += kRecConstrainedTopKContiguousLoadElements) {
      const int32_t remaining = degree - offset;
      const int32_t copy_len =
          remaining < kRecConstrainedTopKContiguousLoadElements
              ? remaining
              : kRecConstrainedTopKContiguousLoadElements;
      const int64_t load_start_us =
          attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
      AscendC::DataCopyExtParams copy_params;
      copy_params.blockCount = 1;
      copy_params.srcStride = 0;
      copy_params.dstStride = 0;
      copy_params.rsv = 0;
      if constexpr (AscendC::IsSameType<LogitType, bfloat16_t>::value) {
        (void)cached_logit_bf16;
        copy_params.blockLen = static_cast<uint32_t>(
            copy_len * static_cast<int32_t>(sizeof(bfloat16_t)));
        AscendC::DataCopyPadExtParams<bfloat16_t> pad_params{false, 0, 0, 0};
        AscendC::DataCopyPad(contiguous_logit_bf16,
                             logits_gm_[row_base + offset],
                             copy_params,
                             pad_params);
        WaitMte2ToVector();
        AscendC::Cast(candidate_logits[offset],
                      contiguous_logit_bf16,
                      AscendC::RoundMode::CAST_NONE,
                      copy_len);
        AscendC::PipeBarrier<PIPE_V>();
        WaitVectorToMte2();
      } else if constexpr (AscendC::IsSameType<LogitType, float>::value) {
        copy_params.blockLen = static_cast<uint32_t>(
            copy_len * static_cast<int32_t>(sizeof(float)));
        AscendC::DataCopyPadExtParams<float> pad_params{false, 0, 0, 0};
        AscendC::DataCopyPad(candidate_logits[offset],
                             logits_gm_[row_base + offset],
                             copy_params,
                             pad_params);
        WaitMte2ToVector();
      }
      if (attribution_mode) {
        attribution.logit_read_us +=
            RecConstrainedTopKTimestampUs() - load_start_us;
      }
      const int64_t score_start_us =
          attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
      if (temperature != 1.0f) {
        AscendC::Muls(candidate_logits[offset],
                      candidate_logits[offset],
                      inv_temperature,
                      copy_len);
        AscendC::PipeBarrier<PIPE_V>();
      }
      if (attribution_mode) {
        attribution.score_store_us +=
            RecConstrainedTopKTimestampUs() - score_start_us;
      }
    }

    const int64_t max_start_us =
        attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
    AscendC::LocalTensor<float> max_tensor = math_buf_.Get<float>();
    AscendC::LocalTensor<float> reduce_tmp = sort_tmp_buf_.Get<float>(
        RecConstrainedTopKReduceTmpElementsForCount(degree));
    AscendC::ReduceMax(max_tensor, candidate_logits, reduce_tmp, degree);
    AscendC::PipeBarrier<PIPE_V>();
    max_value = max_tensor.GetValue(0);
    if (attribution_mode) {
      attribution.score_store_us +=
          RecConstrainedTopKTimestampUs() - max_start_us;
    }
    return true;
  }

  __aicore__ inline int32_t DefaultToken(int32_t table_kind) const {
    if (table_kind == 0 && tiling_data_.first_token_count > 0) {
      return first_token_ids_gm_.GetValue(0);
    }
    if (table_kind == 1 && tiling_data_.prefix1_pair_count > 0) {
      return prefix1_values_gm_.GetValue(0);
    }
    if (table_kind == 2 && prefix2_value_offsets_gm_.GetValue(
                               tiling_data_.prefix1_pair_count) > 0) {
      return prefix2_values_gm_.GetValue(0);
    }
    return 0;
  }

  __aicore__ inline void WriteInvalidRow(int32_t row, int32_t table_kind) {
    const int32_t default_token = DefaultToken(table_kind);
    AscendC::LocalTensor<int32_t> out_tokens =
        output_token_buf_.Get<int32_t>(kRecConstrainedTopKMaxK);
    AscendC::LocalTensor<float> out_logprobs =
        output_logprob_buf_.Get<float>(kRecConstrainedTopKMaxK);
    for (int32_t idx = 0; idx < tiling_data_.top_k; ++idx) {
      out_tokens.SetValue(idx, default_token);
      out_logprobs.SetValue(idx, kRecConstrainedTopKInvalidLogProb);
    }
    FlushOutputRow(row, out_tokens, out_logprobs);
  }

  __aicore__ inline void WriteDebugRow(int32_t row,
                                       int32_t core_id,
                                       int32_t core_num) {
    AscendC::LocalTensor<int32_t> out_tokens =
        output_token_buf_.Get<int32_t>(kRecConstrainedTopKMaxK);
    AscendC::LocalTensor<float> out_logprobs =
        output_logprob_buf_.Get<float>(kRecConstrainedTopKMaxK);
    for (int32_t idx = 0; idx < tiling_data_.top_k; ++idx) {
      out_tokens.SetValue(idx, row + 100);
      out_logprobs.SetValue(idx, static_cast<float>(core_id * 1000 + core_num));
    }
    FlushOutputRow(row, out_tokens, out_logprobs);
  }

  __aicore__ inline void SetAttributionToken(
      AscendC::LocalTensor<int32_t>& out_tokens,
      int32_t index,
      int32_t value) const {
    if (index < AttributionSlotCount()) {
      out_tokens.SetValue(index, value);
    }
  }

  __aicore__ inline void SetAttributionLogprob(
      AscendC::LocalTensor<float>& out_logprobs,
      int32_t index,
      float value) const {
    if (index < AttributionSlotCount()) {
      out_logprobs.SetValue(index, value);
    }
  }

  __aicore__ inline int32_t AttributionSlotCount() const {
    return tiling_data_.top_k < kRecConstrainedTopKMaxK
               ? tiling_data_.top_k
               : kRecConstrainedTopKMaxK;
  }

  __aicore__ inline void WaitMte3ToScalar() {
    event_t event_mte3_s = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(event_mte3_s);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(event_mte3_s);
  }

  __aicore__ inline void WriteAttributionRow(
      int32_t row,
      int32_t core_id,
      int32_t core_num,
      int32_t table_kind,
      int32_t degree,
      bool use_block_sort_topk,
      int32_t filled_topk_count,
      bool invalid_row,
      int64_t row_start_us,
      StageAttribution attribution,
      bool used_contiguous_scan = false) {
    if (tiling_data_.top_k <= 0) {
      return;
    }

    AscendC::LocalTensor<int32_t> out_tokens =
        output_token_buf_.Get<int32_t>(kRecConstrainedTopKMaxK);
    AscendC::LocalTensor<float> out_logprobs =
        output_logprob_buf_.Get<float>(kRecConstrainedTopKMaxK);
    const int32_t slots = AttributionSlotCount();
    for (int32_t idx = 0; idx < slots; ++idx) {
      out_tokens.SetValue(idx, 0);
      out_logprobs.SetValue(idx, 0.0f);
    }

    SetAttributionToken(out_tokens, 0, kRecConstrainedTopKAttributionMagic);
    SetAttributionToken(out_tokens, 1, row);
    SetAttributionToken(out_tokens, 2, tiling_data_.current_step);
    SetAttributionToken(out_tokens, 3, table_kind);
    SetAttributionToken(out_tokens, 4, degree);
    SetAttributionToken(out_tokens, 5, tiling_data_.top_k);
    SetAttributionToken(out_tokens, 6, use_block_sort_topk ? 1 : 0);
    SetAttributionToken(out_tokens, 7, core_id);
    SetAttributionToken(out_tokens, 8, core_num);
    SetAttributionToken(out_tokens, 9, filled_topk_count);
    SetAttributionToken(out_tokens, 10, tiling_data_.temperature_count);
    SetAttributionToken(out_tokens, 11, tiling_data_.max_candidate_count);
    SetAttributionToken(out_tokens, 12, tiling_data_.enable_contiguous_scan);
    SetAttributionToken(out_tokens, 13, used_contiguous_scan ? 1 : 0);
    SetAttributionToken(out_tokens, 14, tiling_data_.enable_vector_final_sort);
    SetAttributionToken(out_tokens, 15, tiling_data_.p11_mode);
    SetAttributionToken(out_tokens, 16, tiling_data_.p19_mode);
    SetAttributionToken(out_tokens, 17, tiling_data_.streaming_mode);

    SetAttributionLogprob(
        out_logprobs, 1, static_cast<float>(attribution.resolve_us));
    SetAttributionLogprob(
        out_logprobs, 2, static_cast<float>(attribution.temperature_us));
    SetAttributionLogprob(
        out_logprobs, 3, static_cast<float>(attribution.candidate_scan_us));
    SetAttributionLogprob(
        out_logprobs, 4, static_cast<float>(attribution.logsumexp_us));
    SetAttributionLogprob(
        out_logprobs, 5, static_cast<float>(attribution.block_sort_us));
    SetAttributionLogprob(
        out_logprobs, 6, static_cast<float>(attribution.merge_us));
    SetAttributionLogprob(
        out_logprobs, 7, static_cast<float>(attribution.final_heap_ln_us));
    SetAttributionLogprob(
        out_logprobs, 8, static_cast<float>(attribution.output_fill_us));
    SetAttributionLogprob(out_logprobs, 10, invalid_row ? 1.0f : 0.0f);
    SetAttributionLogprob(
        out_logprobs,
        11,
        static_cast<float>(attribution.candidate_token_read_us));
    SetAttributionLogprob(
        out_logprobs, 12, static_cast<float>(attribution.logit_read_us));
    SetAttributionLogprob(
        out_logprobs, 13, static_cast<float>(attribution.score_store_us));
    SetAttributionLogprob(
        out_logprobs, 14, static_cast<float>(attribution.scan_topk_update_us));
    SetAttributionLogprob(
        out_logprobs, 15, static_cast<float>(attribution.final_order_us));
    SetAttributionLogprob(
        out_logprobs, 16, static_cast<float>(attribution.final_ln_us));
    SetAttributionLogprob(
        out_logprobs,
        17,
        static_cast<float>(attribution.candidate_token_copy_us));
    SetAttributionLogprob(
        out_logprobs,
        18,
        static_cast<float>(attribution.candidate_token_ub_read_us));
    SetAttributionLogprob(
        out_logprobs, 19, static_cast<float>(attribution.token_materialize_us));
    SetAttributionLogprob(
        out_logprobs, 20, static_cast<float>(attribution.compact_copy_us));
    SetAttributionLogprob(
        out_logprobs, 21, static_cast<float>(attribution.global_sort_us));
    SetAttributionLogprob(
        out_logprobs,
        22,
        static_cast<float>(attribution.pair_payload_prepare_us));
    SetAttributionLogprob(
        out_logprobs, 23, static_cast<float>(attribution.vector_pair_merge_us));
    SetAttributionLogprob(
        out_logprobs,
        24,
        static_cast<float>(attribution.merge_token_lookup_us));
    SetAttributionLogprob(
        out_logprobs, 25, static_cast<float>(attribution.stream_block_scan_us));
    SetAttributionLogprob(
        out_logprobs, 26, static_cast<float>(attribution.stream_block_lse_us));
    SetAttributionLogprob(
        out_logprobs, 27, static_cast<float>(attribution.stream_block_sort_us));
    SetAttributionLogprob(
        out_logprobs,
        28,
        static_cast<float>(attribution.stream_block_merge_us));
    SetAttributionLogprob(
        out_logprobs,
        29,
        static_cast<float>(attribution.stream_token_prefetch_issue_us));
    SetAttributionLogprob(
        out_logprobs,
        30,
        static_cast<float>(attribution.stream_token_prefetch_wait_us));
    SetAttributionLogprob(
        out_logprobs,
        31,
        static_cast<float>(attribution.final_pair_extract_us));

    const int64_t flush_start_us = RecConstrainedTopKTimestampUs();
    FlushOutputRow(row, out_tokens, out_logprobs);
    WaitMte3ToScalar();
    const int64_t flush_end_us = RecConstrainedTopKTimestampUs();
    attribution.flush_us = flush_end_us - flush_start_us;
    attribution.total_us = flush_end_us - row_start_us;

    const int32_t output_base = row * tiling_data_.top_k;
    out_logprobs_gm_.SetValue(output_base,
                              static_cast<float>(attribution.total_us));
    if (slots > 9) {
      out_logprobs_gm_.SetValue(output_base + 9,
                                static_cast<float>(attribution.flush_us));
    }
  }

  __aicore__ inline void FlushOutputRow(
      int32_t row,
      AscendC::LocalTensor<int32_t>& out_tokens,
      AscendC::LocalTensor<float>& out_logprobs) {
    event_t event_s_mte3 = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(event_s_mte3);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(event_s_mte3);

    const int32_t output_base = row * tiling_data_.top_k;
    AscendC::DataCopyExtParams copy_params;
    copy_params.blockCount = 1;
    copy_params.blockLen = static_cast<uint32_t>(
        tiling_data_.top_k * static_cast<int32_t>(sizeof(int32_t)));
    copy_params.srcStride = 0;
    copy_params.dstStride = 0;
    copy_params.rsv = 0;
    AscendC::DataCopyPad(out_tokens_gm_[output_base], out_tokens, copy_params);
    AscendC::DataCopyPad(
        out_logprobs_gm_[output_base], out_logprobs, copy_params);
  }

  __aicore__ inline void MinHeapSiftDown(float* top_values,
                                         int32_t* top_tokens,
                                         int32_t root,
                                         int32_t heap_size) const {
    while (true) {
      const int32_t left = root * 2 + 1;
      if (left >= heap_size) {
        return;
      }
      const int32_t right = left + 1;
      int32_t smallest = left;
      if (right < heap_size && top_values[right] < top_values[left]) {
        smallest = right;
      }
      if (top_values[root] <= top_values[smallest]) {
        return;
      }
      const float value = top_values[root];
      const int32_t token = top_tokens[root];
      top_values[root] = top_values[smallest];
      top_tokens[root] = top_tokens[smallest];
      top_values[smallest] = value;
      top_tokens[smallest] = token;
      root = smallest;
    }
  }

  __aicore__ inline void BuildMinHeap(float* top_values,
                                      int32_t* top_tokens,
                                      int32_t heap_size) const {
    for (int32_t idx = heap_size / 2 - 1; idx >= 0; --idx) {
      MinHeapSiftDown(top_values, top_tokens, idx, heap_size);
    }
  }

  __aicore__ inline void ReplaceMinHeapRoot(float value,
                                            int32_t token,
                                            float* top_values,
                                            int32_t* top_tokens,
                                            int32_t heap_size) const {
    if (value <= top_values[0]) {
      return;
    }
    top_values[0] = value;
    top_tokens[0] = token;
    MinHeapSiftDown(top_values, top_tokens, 0, heap_size);
  }

  __aicore__ inline void SortMinHeapDescending(float* top_values,
                                               int32_t* top_tokens,
                                               int32_t heap_size) const {
    int32_t remaining = heap_size;
    while (remaining > 1) {
      --remaining;
      const float value = top_values[0];
      const int32_t token = top_tokens[0];
      top_values[0] = top_values[remaining];
      top_tokens[0] = top_tokens[remaining];
      top_values[remaining] = value;
      top_tokens[remaining] = token;
      MinHeapSiftDown(top_values, top_tokens, 0, remaining);
    }
  }

  __aicore__ inline bool SortTopKWithVector(
      AscendC::LocalTensor<float>& scratch_values,
      float* top_values,
      int32_t* top_tokens,
      int32_t filled_topk_count) {
    if (tiling_data_.enable_vector_final_sort == 0 || filled_topk_count <= 1 ||
        filled_topk_count > kRecConstrainedTopKMaxK) {
      return false;
    }
    const int32_t aligned_count = RecConstrainedTopKAlign32(filled_topk_count);
    AscendC::LocalTensor<float> sort_pairs =
        sort_pair_buf_.Get<float>(kRecConstrainedTopKSortPairBufferElements);
    AscendC::LocalTensor<int32_t> sort_indices =
        sort_index_buf_.Get<int32_t>(kRecConstrainedTopKSortBlockElements);
    AscendC::LocalTensor<uint32_t> sort_indices_u32 =
        sort_indices.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<float> sort_tmp =
        sort_tmp_buf_.Get<float>(kRecConstrainedTopKSortBlockElements * 3);
    AscendC::LocalTensor<float> sorted_values =
        sort_tmp_buf_.Get<float>(kRecConstrainedTopKSortBlockElements);

    for (int32_t idx = 0; idx < filled_topk_count; ++idx) {
      scratch_values.SetValue(idx, top_values[idx]);
    }
    for (int32_t idx = filled_topk_count; idx < aligned_count; ++idx) {
      scratch_values.SetValue(idx, kRecConstrainedTopKInvalidLogProb);
    }

    AscendC::CreateVecIndex<int32_t>(sort_indices, 0, aligned_count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sort<float, true>(sort_pairs,
                               scratch_values,
                               sort_indices_u32,
                               sort_tmp,
                               aligned_count / 32);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Extract(
        sorted_values, sort_indices_u32, sort_pairs, aligned_count / 32);
    AscendC::PipeBarrier<PIPE_V>();

    int32_t sorted_tokens[kRecConstrainedTopKMaxK];
    for (int32_t idx = 0; idx < filled_topk_count; ++idx) {
      const int32_t original_index = sort_indices.GetValue(idx);
      top_values[idx] = sorted_values.GetValue(idx);
      sorted_tokens[idx] = top_tokens[original_index];
    }
    for (int32_t idx = 0; idx < filled_topk_count; ++idx) {
      top_tokens[idx] = sorted_tokens[idx];
    }
    return true;
  }

  __aicore__ inline void MergeSortedBlockTopK(
      AscendC::LocalTensor<float>& sorted_values,
      AscendC::LocalTensor<int32_t>& sorted_candidate_indices,
      int32_t sorted_count,
      int32_t begin,
      int32_t degree,
      int32_t table_kind,
      int32_t block_offset,
      int32_t block_count,
      bool use_prefetched_tokens,
      AscendC::LocalTensor<int32_t>& block_tokens,
      float* top_values,
      int32_t* top_tokens,
      int32_t& filled_topk_count,
      bool& top_values_are_heap,
      bool attribution_mode,
      StageAttribution& attribution) const {
    for (int32_t idx = 0; idx < sorted_count; ++idx) {
      const float value = sorted_values.GetValue(idx);
      if (value <= -1.0e19f) {
        continue;
      }
      if (top_values_are_heap && filled_topk_count == tiling_data_.top_k &&
          value <= top_values[0]) {
        return;
      }
      const int32_t candidate_index = sorted_candidate_indices.GetValue(idx);
      if (candidate_index < 0 || candidate_index >= degree) {
        continue;
      }
      const int32_t token =
          use_prefetched_tokens
              ? ReadPrefetchedMergeCandidateToken(candidate_index,
                                                  block_offset,
                                                  block_count,
                                                  block_tokens,
                                                  attribution_mode,
                                                  attribution)
              : ReadMergeCandidateToken(table_kind,
                                        begin,
                                        candidate_index,
                                        attribution_mode,
                                        attribution);
      if (token < 0) {
        continue;
      }
      if (filled_topk_count < tiling_data_.top_k) {
        top_values[filled_topk_count] = value;
        top_tokens[filled_topk_count] = token;
        ++filled_topk_count;
        if (filled_topk_count == tiling_data_.top_k) {
          BuildMinHeap(top_values, top_tokens, filled_topk_count);
          top_values_are_heap = true;
        }
      } else {
        ReplaceMinHeapRoot(
            value, token, top_values, top_tokens, filled_topk_count);
      }
    }
  }

  __aicore__ inline void MergeSortedBlockTopKOrdered(
      AscendC::LocalTensor<float>& sorted_values,
      AscendC::LocalTensor<int32_t>& sorted_candidate_indices,
      int32_t sorted_count,
      int32_t begin,
      int32_t degree,
      int32_t table_kind,
      int32_t block_offset,
      int32_t block_count,
      bool use_prefetched_tokens,
      AscendC::LocalTensor<int32_t>& block_tokens,
      float* top_values,
      int32_t* top_tokens,
      int32_t& filled_topk_count,
      bool& top_values_are_heap,
      bool attribution_mode,
      StageAttribution& attribution) const {
    float merged_values[kRecConstrainedTopKMaxK];
    int32_t merged_tokens[kRecConstrainedTopKMaxK];
    int32_t top_idx = 0;
    int32_t block_idx = 0;
    int32_t out_count = 0;

    while (out_count < tiling_data_.top_k) {
      bool block_available = false;
      float block_value = kRecConstrainedTopKInvalidLogProb;
      int32_t block_candidate_index = -1;
      while (block_idx < sorted_count) {
        block_value = sorted_values.GetValue(block_idx);
        if (block_value <= -1.0e19f) {
          block_idx = sorted_count;
          break;
        }
        block_candidate_index = sorted_candidate_indices.GetValue(block_idx);
        if (block_candidate_index >= 0 && block_candidate_index < degree) {
          block_available = true;
          break;
        }
        ++block_idx;
      }

      const bool top_available = top_idx < filled_topk_count;
      if (!top_available && !block_available) {
        break;
      }

      if (top_available &&
          (!block_available || top_values[top_idx] >= block_value)) {
        merged_values[out_count] = top_values[top_idx];
        merged_tokens[out_count] = top_tokens[top_idx];
        ++top_idx;
      } else {
        merged_values[out_count] = block_value;
        merged_tokens[out_count] =
            use_prefetched_tokens
                ? ReadPrefetchedMergeCandidateToken(block_candidate_index,
                                                    block_offset,
                                                    block_count,
                                                    block_tokens,
                                                    attribution_mode,
                                                    attribution)
                : ReadMergeCandidateToken(table_kind,
                                          begin,
                                          block_candidate_index,
                                          attribution_mode,
                                          attribution);
        if (merged_tokens[out_count] < 0) {
          ++block_idx;
          continue;
        }
        ++block_idx;
      }
      ++out_count;
    }

    for (int32_t idx = 0; idx < out_count; ++idx) {
      top_values[idx] = merged_values[idx];
      top_tokens[idx] = merged_tokens[idx];
    }
    filled_topk_count = out_count;
    top_values_are_heap = false;
  }

  __aicore__ inline void SortAndMergeBlockPairs(
      AscendC::LocalTensor<float>& candidate_logits,
      int32_t actual_count,
      AscendC::LocalTensor<int32_t>& block_tokens,
      AscendC::LocalTensor<float>& global_pairs,
      int32_t& global_pair_count,
      bool attribution_mode,
      StageAttribution& attribution) {
    AscendC::LocalTensor<float> block_pairs =
        sort_pair_buf_.Get<float>(kRecConstrainedTopKSortPairBufferElements);
    AscendC::LocalTensor<uint32_t> token_payload =
        block_tokens.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<float> sort_tmp =
        sort_tmp_buf_.Get<float>(kRecConstrainedTopKSortBlockElements * 3);
    const int32_t aligned_count = RecConstrainedTopKAlign32(actual_count);
    for (int32_t pad = actual_count; pad < aligned_count; ++pad) {
      candidate_logits.SetValue(pad, kRecConstrainedTopKInvalidLogProb);
      block_tokens.SetValue(pad, 0);
    }

    const int64_t sort_start_us =
        attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sort<float, true>(block_pairs,
                               candidate_logits,
                               token_payload,
                               sort_tmp,
                               aligned_count / 32);
    AscendC::PipeBarrier<PIPE_V>();
    if (attribution_mode) {
      const int64_t elapsed_us =
          RecConstrainedTopKTimestampUs() - sort_start_us;
      attribution.block_sort_us += elapsed_us;
      attribution.stream_block_sort_us += elapsed_us;
    }

    const int32_t block_pair_count =
        actual_count < tiling_data_.top_k ? actual_count : tiling_data_.top_k;
    if (global_pair_count == 0) {
      const int64_t prepare_start_us =
          attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
      AscendC::DataCopy(global_pairs,
                        block_pairs,
                        AscendC::GetSortLen<float>(block_pair_count));
      AscendC::PipeBarrier<PIPE_V>();
      global_pair_count = block_pair_count;
      if (attribution_mode) {
        attribution.pair_payload_prepare_us +=
            RecConstrainedTopKTimestampUs() - prepare_start_us;
      }
      return;
    }

    AscendC::LocalTensor<float> merged_pairs =
        sort_tmp_buf_.Get<float>(kRecConstrainedTopKSortBlockElements * 3);
    AscendC::MrgSortSrcList<float> sort_list(
        global_pairs, block_pairs, global_pairs, global_pairs);
    const uint16_t element_counts[4] = {
        static_cast<uint16_t>(global_pair_count),
        static_cast<uint16_t>(block_pair_count),
        0,
        0};
    uint32_t sorted_counts[4] = {0, 0, 0, 0};
    const int64_t merge_start_us =
        attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
    AscendC::MrgSort<float, false>(
        merged_pairs, sort_list, element_counts, sorted_counts, 0b11, 1);
    AscendC::PipeBarrier<PIPE_V>();
    global_pair_count = global_pair_count + block_pair_count;
    if (global_pair_count > tiling_data_.top_k) {
      global_pair_count = tiling_data_.top_k;
    }
    AscendC::DataCopy(global_pairs,
                      merged_pairs,
                      AscendC::GetSortLen<float>(global_pair_count));
    AscendC::PipeBarrier<PIPE_V>();
    if (attribution_mode) {
      const int64_t elapsed_us =
          RecConstrainedTopKTimestampUs() - merge_start_us;
      attribution.vector_pair_merge_us += elapsed_us;
      attribution.merge_us += elapsed_us;
      attribution.stream_block_merge_us += elapsed_us;
    }
  }

  __aicore__ inline void ExtractGlobalPairs(
      AscendC::LocalTensor<float>& global_pairs,
      int32_t global_pair_count,
      float* top_values,
      int32_t* top_tokens,
      int32_t& filled_topk_count,
      bool& top_values_are_heap,
      bool attribution_mode,
      StageAttribution& attribution) {
    const int64_t extract_start_us =
        attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
    AscendC::LocalTensor<float> final_values =
        sort_tmp_buf_.Get<float>(kRecConstrainedTopKSortBlockElements);
    AscendC::LocalTensor<int32_t> final_tokens =
        sort_index_buf_.Get<int32_t>(kRecConstrainedTopKSortBlockElements);
    AscendC::LocalTensor<uint32_t> final_tokens_u32 =
        final_tokens.ReinterpretCast<uint32_t>();
    AscendC::Extract(final_values,
                     final_tokens_u32,
                     global_pairs,
                     RecConstrainedTopKAlign32(global_pair_count) / 32);
    AscendC::PipeBarrier<PIPE_V>();
    for (int32_t idx = 0; idx < global_pair_count; ++idx) {
      top_values[idx] = final_values.GetValue(idx);
      top_tokens[idx] = final_tokens.GetValue(idx);
    }
    filled_topk_count = global_pair_count;
    top_values_are_heap = false;
    if (attribution_mode) {
      attribution.final_pair_extract_us +=
          RecConstrainedTopKTimestampUs() - extract_start_us;
    }
  }

  __aicore__ inline void SortAndMergeBlock(
      AscendC::LocalTensor<float>& candidate_logits,
      int32_t score_offset,
      int32_t candidate_index_offset,
      int32_t actual_count,
      int32_t begin,
      int32_t degree,
      int32_t table_kind,
      bool use_prefetched_tokens,
      AscendC::LocalTensor<int32_t>& block_tokens,
      float* top_values,
      int32_t* top_tokens,
      int32_t& filled_topk_count,
      bool& top_values_are_heap,
      bool attribution_mode,
      bool streaming_mode,
      StageAttribution& attribution) {
    AscendC::LocalTensor<float> sort_pairs =
        sort_pair_buf_.Get<float>(kRecConstrainedTopKSortPairBufferElements);
    AscendC::LocalTensor<int32_t> sort_indices =
        sort_index_buf_.Get<int32_t>(kRecConstrainedTopKSortBlockElements);
    AscendC::LocalTensor<uint32_t> sort_indices_u32 =
        sort_indices.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<float> sort_tmp =
        sort_tmp_buf_.Get<float>(kRecConstrainedTopKSortBlockElements * 3);
    AscendC::LocalTensor<float> sorted_values =
        sort_tmp_buf_.Get<float>(kRecConstrainedTopKSortBlockElements);

    const int32_t aligned_count = RecConstrainedTopKAlign32(actual_count);
    const int64_t block_sort_start_us =
        attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
    for (int32_t pad = actual_count; pad < aligned_count; ++pad) {
      candidate_logits.SetValue(score_offset + pad,
                                kRecConstrainedTopKInvalidLogProb);
    }

    AscendC::CreateVecIndex<int32_t>(
        sort_indices, candidate_index_offset, aligned_count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sort<float, true>(sort_pairs,
                               candidate_logits[score_offset],
                               sort_indices_u32,
                               sort_tmp,
                               aligned_count / 32);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Extract(
        sorted_values, sort_indices_u32, sort_pairs, aligned_count / 32);
    AscendC::PipeBarrier<PIPE_V>();
    if (attribution_mode) {
      const int64_t elapsed_us =
          RecConstrainedTopKTimestampUs() - block_sort_start_us;
      attribution.block_sort_us += elapsed_us;
      if (streaming_mode) {
        attribution.stream_block_sort_us += elapsed_us;
      }
    }

    const int32_t sorted_count =
        actual_count < tiling_data_.top_k ? actual_count : tiling_data_.top_k;
    const int64_t merge_start_us =
        attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
    if (tiling_data_.p11_mode == 3 || tiling_data_.p11_mode == 4) {
      MergeSortedBlockTopKOrdered(sorted_values,
                                  sort_indices,
                                  sorted_count,
                                  begin,
                                  degree,
                                  table_kind,
                                  candidate_index_offset,
                                  actual_count,
                                  use_prefetched_tokens,
                                  block_tokens,
                                  top_values,
                                  top_tokens,
                                  filled_topk_count,
                                  top_values_are_heap,
                                  attribution_mode,
                                  attribution);
    } else {
      MergeSortedBlockTopK(sorted_values,
                           sort_indices,
                           sorted_count,
                           begin,
                           degree,
                           table_kind,
                           candidate_index_offset,
                           actual_count,
                           use_prefetched_tokens,
                           block_tokens,
                           top_values,
                           top_tokens,
                           filled_topk_count,
                           top_values_are_heap,
                           attribution_mode,
                           attribution);
    }
    if (attribution_mode) {
      const int64_t elapsed_us =
          RecConstrainedTopKTimestampUs() - merge_start_us;
      attribution.merge_us += elapsed_us;
      if (streaming_mode) {
        attribution.stream_block_merge_us += elapsed_us;
      }
    }
  }

  __aicore__ inline void SelectTopKWithBlockSort(
      AscendC::LocalTensor<float>& candidate_logits,
      int32_t begin,
      int32_t degree,
      int32_t table_kind,
      float* top_values,
      int32_t* top_tokens,
      int32_t& filled_topk_count,
      bool& top_values_are_heap,
      bool attribution_mode,
      StageAttribution& attribution) {
    AscendC::LocalTensor<int32_t> unused_block_tokens =
        candidate_token_chunk_buf_.Get<int32_t>(
            kRecConstrainedTopKCandidateTokenChunkElements);
    for (int32_t offset = 0; offset < degree;
         offset += kRecConstrainedTopKSortBlockElements) {
      const int32_t remaining = degree - offset;
      const int32_t actual_count =
          remaining < kRecConstrainedTopKSortBlockElements
              ? remaining
              : kRecConstrainedTopKSortBlockElements;
      SortAndMergeBlock(candidate_logits,
                        offset,
                        offset,
                        actual_count,
                        begin,
                        degree,
                        table_kind,
                        false,
                        unused_block_tokens,
                        top_values,
                        top_tokens,
                        filled_topk_count,
                        top_values_are_heap,
                        attribution_mode,
                        false,
                        attribution);
    }
  }

  __aicore__ inline void ProcessStreamingBlocks(
      int32_t row,
      int32_t begin,
      int32_t degree,
      int32_t table_kind,
      float temperature,
      int32_t& cached_block_offset,
      AscendC::LocalTensor<bfloat16_t>& cached_logit_bf16,
      AscendC::LocalTensor<float>& cached_logit_float,
      AscendC::LocalTensor<float>& block_scores,
      float* top_values,
      int32_t* top_tokens,
      int32_t& filled_topk_count,
      bool& top_values_are_heap,
      float& global_max,
      float& global_sum_exp,
      bool attribution_mode,
      StageAttribution& attribution) {
    const bool use_p11_token_staging =
        (tiling_data_.p11_mode == 2 || tiling_data_.p11_mode == 4) &&
        degree >= kRecConstrainedTopKTokenStagingThreshold;
    AscendC::LocalTensor<int32_t> staged_candidate_tokens =
        candidate_token_chunk_buf_.Get<int32_t>(
            kRecConstrainedTopKCandidateTokenChunkElements);
    const int32_t staged_scan_step =
        use_p11_token_staging ? kRecConstrainedTopKCandidateTokenChunkElements
                              : 1;
    const bool use_block_token_pipeline =
        tiling_data_.streaming_mode == 2 &&
        degree > kRecConstrainedTopKSortBlockElements &&
        (tiling_data_.p11_mode == 3 || tiling_data_.p11_mode == 4);
    const bool use_pairwise_vector_merge =
        tiling_data_.p19_mode == 1 && use_block_token_pipeline;
    AscendC::LocalTensor<int32_t> block_token_buffers;
    AscendC::LocalTensor<float> global_pairs;
    int32_t global_pair_count = 0;
    if (use_pairwise_vector_merge) {
      global_pairs = global_pair_buf_.Get<float>(
          kRecConstrainedTopKGlobalPairBufferElements);
    }
    if (use_block_token_pipeline) {
      block_token_buffers = block_token_buf_.Get<int32_t>(
          kRecConstrainedTopKBlockTokenBufferCount *
          kRecConstrainedTopKSortBlockElements);
      AscendC::LocalTensor<int32_t> first_block_tokens = block_token_buffers;
      const int32_t first_count = degree < kRecConstrainedTopKSortBlockElements
                                      ? degree
                                      : kRecConstrainedTopKSortBlockElements;
      PrefetchCandidateTokenBlock(first_block_tokens,
                                  table_kind,
                                  begin,
                                  first_count,
                                  block_token_events_[0],
                                  attribution_mode,
                                  attribution);
    }

    int32_t block_index = 0;
    for (int32_t block_offset = 0; block_offset < degree;
         block_offset += kRecConstrainedTopKSortBlockElements, ++block_index) {
      const int32_t remaining = degree - block_offset;
      const int32_t actual_count =
          remaining < kRecConstrainedTopKSortBlockElements
              ? remaining
              : kRecConstrainedTopKSortBlockElements;
      float block_max = -3.4028234663852886e38f;
      const int64_t block_scan_start_us =
          attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
      const int32_t buffer_index =
          block_index % kRecConstrainedTopKBlockTokenBufferCount;
      AscendC::LocalTensor<int32_t> current_block_tokens =
          use_block_token_pipeline
              ? block_token_buffers[buffer_index *
                                    kRecConstrainedTopKSortBlockElements]
              : staged_candidate_tokens;
      if (use_block_token_pipeline) {
        WaitCandidateTokenBlock(
            block_token_events_[buffer_index], attribution_mode, attribution);
      }
      const int32_t scan_step =
          use_block_token_pipeline ? actual_count : staged_scan_step;

      for (int32_t chunk_begin = 0; chunk_begin < actual_count;
           chunk_begin += scan_step) {
        const int32_t block_remaining = actual_count - chunk_begin;
        const int32_t copy_count =
            block_remaining < scan_step ? block_remaining : scan_step;
        if (use_p11_token_staging && !use_block_token_pipeline) {
          const int64_t token_copy_start_us =
              attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
          CopyCandidateTokenChunk(staged_candidate_tokens,
                                  table_kind,
                                  begin + block_offset + chunk_begin,
                                  copy_count);
          WaitMte2ToScalar();
          if (attribution_mode) {
            attribution.candidate_token_copy_us +=
                RecConstrainedTopKTimestampUs() - token_copy_start_us;
          }
        }
        for (int32_t local_idx = 0; local_idx < copy_count; ++local_idx) {
          const int32_t score_index = chunk_begin + local_idx;
          int64_t sub_stage_start_us =
              attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
          const int32_t token =
              use_block_token_pipeline
                  ? current_block_tokens.GetValue(score_index)
              : use_p11_token_staging
                  ? staged_candidate_tokens.GetValue(local_idx)
                  : ReadCandidateToken(table_kind,
                                       begin + block_offset + score_index);
          if (attribution_mode) {
            const int64_t sub_stage_end_us = RecConstrainedTopKTimestampUs();
            if (use_p11_token_staging || use_block_token_pipeline) {
              attribution.candidate_token_ub_read_us +=
                  sub_stage_end_us - sub_stage_start_us;
            } else {
              attribution.candidate_token_read_us +=
                  sub_stage_end_us - sub_stage_start_us;
            }
            sub_stage_start_us = sub_stage_end_us;
          }
          if (token < 0 || token >= tiling_data_.vocab_size) {
            block_scores.SetValue(score_index,
                                  kRecConstrainedTopKInvalidLogProb);
            if (attribution_mode) {
              attribution.score_store_us +=
                  RecConstrainedTopKTimestampUs() - sub_stage_start_us;
            }
            continue;
          }
          const float logit = ReadLogit(row,
                                        token,
                                        cached_block_offset,
                                        cached_logit_bf16,
                                        cached_logit_float);
          if (attribution_mode) {
            const int64_t sub_stage_end_us = RecConstrainedTopKTimestampUs();
            attribution.logit_read_us += sub_stage_end_us - sub_stage_start_us;
            sub_stage_start_us = sub_stage_end_us;
          }
          const float value = logit / temperature;
          block_scores.SetValue(score_index, value);
          if (value > block_max) {
            block_max = value;
          }
          if (attribution_mode) {
            attribution.score_store_us +=
                RecConstrainedTopKTimestampUs() - sub_stage_start_us;
          }
        }
      }

      if (attribution_mode) {
        const int64_t elapsed_us =
            RecConstrainedTopKTimestampUs() - block_scan_start_us;
        attribution.candidate_scan_us += elapsed_us;
        attribution.stream_block_scan_us += elapsed_us;
      }

      const int32_t next_block_offset =
          block_offset + kRecConstrainedTopKSortBlockElements;
      if (use_block_token_pipeline && next_block_offset < degree) {
        const int32_t next_remaining = degree - next_block_offset;
        const int32_t next_count =
            next_remaining < kRecConstrainedTopKSortBlockElements
                ? next_remaining
                : kRecConstrainedTopKSortBlockElements;
        const int32_t next_buffer_index =
            (buffer_index + 1) % kRecConstrainedTopKBlockTokenBufferCount;
        AscendC::LocalTensor<int32_t> next_block_tokens =
            block_token_buffers[next_buffer_index *
                                kRecConstrainedTopKSortBlockElements];
        PrefetchCandidateTokenBlock(next_block_tokens,
                                    table_kind,
                                    begin + next_block_offset,
                                    next_count,
                                    block_token_events_[next_buffer_index],
                                    attribution_mode,
                                    attribution);
      }

      const int64_t block_lse_start_us =
          attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
      if (block_max > -3.0e38f) {
        const float block_sum_exp =
            SumExpFromCandidateLogits(block_scores, actual_count, block_max);
        if (global_max < -3.0e38f) {
          global_max = block_max;
          global_sum_exp = block_sum_exp;
        } else {
          const float new_max = global_max > block_max ? global_max : block_max;
          global_sum_exp = global_sum_exp * ExpScalar(global_max - new_max) +
                           block_sum_exp * ExpScalar(block_max - new_max);
          global_max = new_max;
        }
      }
      if (attribution_mode) {
        const int64_t elapsed_us =
            RecConstrainedTopKTimestampUs() - block_lse_start_us;
        attribution.logsumexp_us += elapsed_us;
        attribution.stream_block_lse_us += elapsed_us;
      }

      if (use_pairwise_vector_merge) {
        SortAndMergeBlockPairs(block_scores,
                               actual_count,
                               current_block_tokens,
                               global_pairs,
                               global_pair_count,
                               attribution_mode,
                               attribution);
      } else {
        SortAndMergeBlock(block_scores,
                          0,
                          block_offset,
                          actual_count,
                          begin,
                          degree,
                          table_kind,
                          use_block_token_pipeline,
                          current_block_tokens,
                          top_values,
                          top_tokens,
                          filled_topk_count,
                          top_values_are_heap,
                          attribution_mode,
                          true,
                          attribution);
      }
    }
    if (use_pairwise_vector_merge) {
      ExtractGlobalPairs(global_pairs,
                         global_pair_count,
                         top_values,
                         top_tokens,
                         filled_topk_count,
                         top_values_are_heap,
                         attribution_mode,
                         attribution);
    }
  }

  __aicore__ inline void ProcessRow(int32_t row,
                                    int32_t core_id,
                                    int32_t core_num) {
    const bool attribution_mode = tiling_data_.debug_mode == 2;
    StageAttribution attribution;
    const int64_t row_start_us =
        attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
    int32_t begin = 0;
    int32_t degree = 0;
    int32_t table_kind = 0;
    int64_t stage_start_us = row_start_us;
    ResolveRange(row, begin, degree, table_kind);
    if (attribution_mode) {
      const int64_t stage_end_us = RecConstrainedTopKTimestampUs();
      attribution.resolve_us = stage_end_us - stage_start_us;
      stage_start_us = stage_end_us;
    }
    if (degree <= 0 || tiling_data_.top_k <= 0 ||
        tiling_data_.top_k > kRecConstrainedTopKMaxK) {
      if (attribution_mode) {
        WriteAttributionRow(row,
                            core_id,
                            core_num,
                            table_kind,
                            degree,
                            false,
                            0,
                            true,
                            row_start_us,
                            attribution);
        return;
      }
      WriteInvalidRow(row, table_kind);
      return;
    }

    float temperature = ReadTemperature(row);
    if (temperature == 0.0f) {
      temperature = 1.0f;
    }
    if (attribution_mode) {
      const int64_t stage_end_us = RecConstrainedTopKTimestampUs();
      attribution.temperature_us = stage_end_us - stage_start_us;
      stage_start_us = stage_end_us;
    }

    int32_t cached_block_offset = -1;
    const int32_t logit_cache_elements = LogitCacheBufferElements();
    AscendC::LocalTensor<bfloat16_t> cached_logit_bf16 =
        logit_bf16_buf_.Get<bfloat16_t>(logit_cache_elements);
    AscendC::LocalTensor<float> cached_logit_float =
        logit_float_buf_.Get<float>(logit_cache_elements);
    AscendC::LocalTensor<float> candidate_logits =
        candidate_logit_buf_.Get<float>(CandidateLogitBufferElements());

    float top_values[kRecConstrainedTopKMaxK];
    int32_t top_tokens[kRecConstrainedTopKMaxK];
    const int32_t default_token = DefaultToken(table_kind);
    for (int32_t idx = 0; idx < tiling_data_.top_k; ++idx) {
      top_values[idx] = -3.4028234663852886e38f;
      top_tokens[idx] = default_token;
    }
    int32_t filled_topk_count = 0;
    bool top_values_are_heap = false;
    const bool no_candidate_prune_needed = degree <= tiling_data_.top_k;
    const bool use_block_sort_topk = degree > tiling_data_.top_k &&
                                     degree >= kRecConstrainedTopKSortThreshold;
    const bool use_streaming_mode =
        tiling_data_.streaming_mode != 0 && use_block_sort_topk;

    float max_value = -3.4028234663852886e38f;
    float sum_exp = 0.0f;
    bool used_contiguous_scan = false;
    stage_start_us = attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
    if (use_streaming_mode) {
      ProcessStreamingBlocks(row,
                             begin,
                             degree,
                             table_kind,
                             temperature,
                             cached_block_offset,
                             cached_logit_bf16,
                             cached_logit_float,
                             candidate_logits,
                             top_values,
                             top_tokens,
                             filled_topk_count,
                             top_values_are_heap,
                             max_value,
                             sum_exp,
                             attribution_mode,
                             attribution);
    } else {
      used_contiguous_scan =
          use_block_sort_topk &&
          TryLoadContiguousCandidateScores(row,
                                           begin,
                                           degree,
                                           table_kind,
                                           temperature,
                                           candidate_logits,
                                           cached_block_offset,
                                           cached_logit_bf16,
                                           cached_logit_float,
                                           max_value,
                                           attribution_mode,
                                           attribution);
      if (!used_contiguous_scan) {
        const bool use_p11_token_staging =
            (tiling_data_.p11_mode == 2 || tiling_data_.p11_mode == 4) &&
            degree >= kRecConstrainedTopKTokenStagingThreshold;
        AscendC::LocalTensor<int32_t> staged_candidate_tokens =
            candidate_token_chunk_buf_.Get<int32_t>(
                kRecConstrainedTopKCandidateTokenChunkElements);
        const int32_t scan_step =
            use_p11_token_staging
                ? kRecConstrainedTopKCandidateTokenChunkElements
                : 1;
        for (int32_t chunk_begin = 0; chunk_begin < degree;
             chunk_begin += scan_step) {
          const int32_t remaining = degree - chunk_begin;
          const int32_t copy_count =
              remaining < scan_step ? remaining : scan_step;
          if (use_p11_token_staging) {
            const int64_t token_copy_start_us =
                attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
            CopyCandidateTokenChunk(staged_candidate_tokens,
                                    table_kind,
                                    begin + chunk_begin,
                                    copy_count);
            WaitMte2ToScalar();
            if (attribution_mode) {
              attribution.candidate_token_copy_us +=
                  RecConstrainedTopKTimestampUs() - token_copy_start_us;
            }
          }
          for (int32_t local_idx = 0; local_idx < copy_count; ++local_idx) {
            const int32_t idx = chunk_begin + local_idx;
            int64_t sub_stage_start_us =
                attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
            const int32_t token =
                use_p11_token_staging
                    ? staged_candidate_tokens.GetValue(local_idx)
                    : ReadCandidateToken(table_kind, begin + idx);
            if (attribution_mode) {
              const int64_t sub_stage_end_us = RecConstrainedTopKTimestampUs();
              if (use_p11_token_staging) {
                attribution.candidate_token_ub_read_us +=
                    sub_stage_end_us - sub_stage_start_us;
              } else {
                attribution.candidate_token_read_us +=
                    sub_stage_end_us - sub_stage_start_us;
              }
              sub_stage_start_us = sub_stage_end_us;
            }
            if (token < 0 || token >= tiling_data_.vocab_size) {
              candidate_logits.SetValue(idx, kRecConstrainedTopKInvalidLogProb);
              if (attribution_mode) {
                attribution.score_store_us +=
                    RecConstrainedTopKTimestampUs() - sub_stage_start_us;
              }
              continue;
            }
            const float logit = ReadLogit(row,
                                          token,
                                          cached_block_offset,
                                          cached_logit_bf16,
                                          cached_logit_float);
            if (attribution_mode) {
              const int64_t sub_stage_end_us = RecConstrainedTopKTimestampUs();
              attribution.logit_read_us +=
                  sub_stage_end_us - sub_stage_start_us;
              sub_stage_start_us = sub_stage_end_us;
            }
            const float value = logit / temperature;
            candidate_logits.SetValue(idx, value);
            if (value > max_value) {
              max_value = value;
            }
            if (attribution_mode) {
              const int64_t sub_stage_end_us = RecConstrainedTopKTimestampUs();
              attribution.score_store_us +=
                  sub_stage_end_us - sub_stage_start_us;
              sub_stage_start_us = sub_stage_end_us;
            }
            if (use_block_sort_topk) {
              continue;
            }
            if (no_candidate_prune_needed) {
              top_values[filled_topk_count] = value;
              top_tokens[filled_topk_count] = token;
              ++filled_topk_count;
            } else if (filled_topk_count < tiling_data_.top_k) {
              top_values[filled_topk_count] = value;
              top_tokens[filled_topk_count] = token;
              ++filled_topk_count;
              if (filled_topk_count == tiling_data_.top_k) {
                BuildMinHeap(top_values, top_tokens, filled_topk_count);
                top_values_are_heap = true;
              }
            } else {
              ReplaceMinHeapRoot(
                  value, token, top_values, top_tokens, filled_topk_count);
            }
            if (attribution_mode) {
              attribution.scan_topk_update_us +=
                  RecConstrainedTopKTimestampUs() - sub_stage_start_us;
            }
          }
        }
      }
      if (attribution_mode) {
        attribution.candidate_scan_us =
            RecConstrainedTopKTimestampUs() - stage_start_us;
      }
    }

    if (max_value < -3.0e38f) {
      if (attribution_mode) {
        WriteAttributionRow(row,
                            core_id,
                            core_num,
                            table_kind,
                            degree,
                            use_block_sort_topk,
                            filled_topk_count,
                            true,
                            row_start_us,
                            attribution);
        return;
      }
      WriteInvalidRow(row, table_kind);
      return;
    }

    if (!use_streaming_mode) {
      stage_start_us = attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
      sum_exp = SumExpFromCandidateLogits(candidate_logits, degree, max_value);
      if (attribution_mode) {
        attribution.logsumexp_us =
            RecConstrainedTopKTimestampUs() - stage_start_us;
      }
    }

    if (sum_exp <= 0.0f) {
      if (attribution_mode) {
        WriteAttributionRow(row,
                            core_id,
                            core_num,
                            table_kind,
                            degree,
                            use_block_sort_topk,
                            filled_topk_count,
                            true,
                            row_start_us,
                            attribution);
        return;
      }
      WriteInvalidRow(row, table_kind);
      return;
    }

    if (use_block_sort_topk && !use_streaming_mode) {
      SelectTopKWithBlockSort(candidate_logits,
                              begin,
                              degree,
                              table_kind,
                              top_values,
                              top_tokens,
                              filled_topk_count,
                              top_values_are_heap,
                              attribution_mode,
                              attribution);
    }
    stage_start_us = attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
    bool final_sorted_by_vector = false;
    const bool final_already_ordered =
        use_block_sort_topk &&
        (tiling_data_.p11_mode == 3 || tiling_data_.p11_mode == 4);
    if (filled_topk_count > 1 && !final_already_ordered) {
      final_sorted_by_vector = SortTopKWithVector(
          candidate_logits, top_values, top_tokens, filled_topk_count);
    }
    if (!final_sorted_by_vector && !final_already_ordered) {
      if (filled_topk_count > 1 && !top_values_are_heap) {
        BuildMinHeap(top_values, top_tokens, filled_topk_count);
        top_values_are_heap = true;
      }
      if (top_values_are_heap) {
        SortMinHeapDescending(top_values, top_tokens, filled_topk_count);
      }
    }
    if (attribution_mode) {
      const int64_t stage_end_us = RecConstrainedTopKTimestampUs();
      attribution.final_order_us = stage_end_us - stage_start_us;
      stage_start_us = stage_end_us;
    }
    const float log_denom = LnScalar(sum_exp) + max_value;
    if (attribution_mode) {
      attribution.final_ln_us =
          RecConstrainedTopKTimestampUs() - stage_start_us;
      attribution.final_heap_ln_us =
          attribution.final_order_us + attribution.final_ln_us;
    }
    AscendC::LocalTensor<int32_t> out_tokens =
        output_token_buf_.Get<int32_t>(kRecConstrainedTopKMaxK);
    AscendC::LocalTensor<float> out_logprobs =
        output_logprob_buf_.Get<float>(kRecConstrainedTopKMaxK);
    stage_start_us = attribution_mode ? RecConstrainedTopKTimestampUs() : 0;
    for (int32_t idx = 0; idx < tiling_data_.top_k; ++idx) {
      if (top_values[idx] < -3.0e38f) {
        out_tokens.SetValue(idx, default_token);
        out_logprobs.SetValue(idx, kRecConstrainedTopKInvalidLogProb);
      } else {
        out_tokens.SetValue(idx, top_tokens[idx]);
        out_logprobs.SetValue(idx, top_values[idx] - log_denom);
      }
    }
    if (attribution_mode) {
      attribution.output_fill_us =
          RecConstrainedTopKTimestampUs() - stage_start_us;
      WriteAttributionRow(row,
                          core_id,
                          core_num,
                          table_kind,
                          degree,
                          use_block_sort_topk,
                          filled_topk_count,
                          false,
                          row_start_us,
                          attribution,
                          used_contiguous_scan);
      return;
    }
    FlushOutputRow(row, out_tokens, out_logprobs);
  }

  RecConstrainedTopKTilingData tiling_data_;
  AscendC::TPipe* pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> math_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> exp_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> reduce_tmp_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> logit_bf16_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> logit_float_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> contiguous_logit_bf16_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> candidate_token_chunk_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> block_token_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> candidate_logit_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> sort_pair_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> global_pair_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> sort_index_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> sort_tmp_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> output_token_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> output_logprob_buf_;
  event_t block_token_events_[kRecConstrainedTopKBlockTokenBufferCount];
  AscendC::GlobalTensor<LogitType> logits_gm_;
  AscendC::GlobalTensor<int32_t> sequence_group_gm_;
  AscendC::GlobalTensor<int32_t> first_token_ids_gm_;
  AscendC::GlobalTensor<int32_t> prefix1_offsets_gm_;
  AscendC::GlobalTensor<int32_t> prefix1_values_gm_;
  AscendC::GlobalTensor<int64_t> prefix1_pair_keys_gm_;
  AscendC::GlobalTensor<int32_t> prefix2_value_offsets_gm_;
  AscendC::GlobalTensor<int32_t> prefix2_values_gm_;
  AscendC::GlobalTensor<float> temperatures_gm_;
  AscendC::GlobalTensor<int32_t> out_tokens_gm_;
  AscendC::GlobalTensor<float> out_logprobs_gm_;
};

}  // namespace kernels
