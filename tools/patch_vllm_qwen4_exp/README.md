# tools/patch_vllm_qwen4_exp

Three small, idempotent patches to vLLM's own `Qwen4ExpForConditionalGeneration`
model code (`vllm/models/qwen4_exp/nvidia/`), needed so the
plugin's `get_quant_method` is actually consulted for the layers a native EXL3
pack quantizes outside the routed experts. Each script takes the path to a
vLLM `site-packages/vllm` directory, checks whether it has already been
applied, compile-checks the patched source before writing it, and keeps a
backup of the original file (`.orig` / `.orig2`) so the patch can be reverted
by hand.

Status against current vLLM (v0.30.0) is tracked in
[`docs/VLLM_COMPATIBILITY.md`](../../docs/VLLM_COMPATIBILITY.md);
`tools/check_vllm_compat.py <tree>` re-checks all three anchors against a vLLM
source tree without touching it.

- `patch_vllm_qwen4_ple.py` adds `quant_config=` to the `ParallelLMHead`
  constructor in `model.py`, and — on vLLM builds that still construct the
  n-gram table without a quant config — to `PLEVocabParallelEmbedding` in
  `ple_layer.py`. Without the first, vLLM builds the head unquantized
  regardless of `--quantization`, so a pack whose `lm_head` is an EXL3 trellis
  tensor cannot load: the quant config never gets asked for a method, and the
  layer expects a plain `lm_head.weight` that the checkpoint does not have.

  Current vLLM constructs the n-gram table with a quant config itself, so the
  PLE half now reports itself obsolete instead of failing its anchor check. It
  does not follow that EXL3 n-gram tables work there: the layer selects its
  storage format through `Qwen4ExpPLEEmbeddingMethod.from_quant_config`, which
  rejects anything that is not ModelOpt or `Fp8Config`. That needs the adapter
  described in the compatibility page.

- `patch_vllm_vision_split.py` drops the split vision `q_proj` / `k_proj` /
  `v_proj` name substrings from `Qwen3_VisionTransformer`'s checkpoint
  name-mapping. turboderp's Qwen3.8-Flash-Next pack ships both the split
  trellis tensors and the original fused bf16 `attn.qkv.weight/bias`, but
  vLLM's vision transformer only holds the fused tensor, so
  `AutoWeightsLoader` raised `no module or parameter named
  'blocks.0.attn.k_proj'` once the language model had already finished
  loading. Dropping the split names is correct either way: the fused bf16
  copy is the one vLLM's module actually uses.

- `patch_vllm_mtp_lmhead.py` is the same `quant_config=` addition as the PLE
  patch, applied to the MTP draft model's `ParallelLMHead` in `mtp.py`. The
  draft shares the main checkpoint's `lm_head` weights (the checkpoint name
  mapper routes `lm_head.` to both), so it needs the same fix or the draft's
  head fails to load with the same missing-`lm_head.weight` error.

Usage: `python3 patch_vllm_qwen4_ple.py <site-packages/vllm>`, then the other
two the same way. Safe to run more than once.
