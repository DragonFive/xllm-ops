# RecConstrainedTopK

This directory contains the fused OneRec constrained top-k custom op.

The xLLM runtime wrapper attempts the fused path by default and falls back to
the composite selector when the custom op symbols are unavailable, the shape
is unsupported, or execution fails. Set
`XLLM_ENABLE_ONEREC_FUSED_CONSTRAINED_TOPK=0` for an emergency composite-path
rollback.

The retained production defaults are load mode4, streaming mode2, P11 mode4,
and P19 mode1. Their existing environment variables remain numeric overrides;
setting them to `0` selects the corresponding baseline behavior. Shape guards
still decide whether each optimized sub-path applies to a particular row.

Supported contract:

- `current_step` in `{0, 1, 2}`;
- `top_k <= 512`;
- `logits` dtype `float16`, `bfloat16`, or `float32`;
- constraint tables use `int32`, with `prefix1_pair_keys` as `int64`;
- output tokens are `int32`, output logprobs are `float32`;
- invalid or empty prefixes emit token `0` and logprob `-1e20`.
