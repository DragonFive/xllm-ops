# RecConstrainedTopK

This directory contains the opt-in fused OneRec constrained top-k custom op.

The xLLM runtime wrapper is gated by
`XLLM_ENABLE_ONEREC_FUSED_CONSTRAINED_TOPK=1` and falls back to the composite
selector when the custom op symbols are unavailable or execution fails.

The first implementation prioritizes operator-oracle equivalence and runtime
fallback safety over making the fused selector the default path. Do not make it
default until the operator oracle, debug fused-vs-composite comparison, HTTP
benchmark, and msprof gates pass.

Supported contract:

- `current_step` in `{0, 1, 2}`;
- `top_k <= 512`;
- `logits` dtype `float16`, `bfloat16`, or `float32`;
- constraint tables use `int32`, with `prefix1_pair_keys` as `int64`;
- output tokens are `int32`, output logprobs are `float32`;
- invalid or empty prefixes emit token `0` and logprob `-1e20`.
