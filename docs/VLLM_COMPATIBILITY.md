# vLLM compatibility

What this plugin requires from a vLLM runtime, and the state of that surface against the
current release. Re-run `tools/check_vllm_compat.py` against any tree to reproduce the
mechanical part of this page.

## Reference

| Field | Value |
|---|---|
| vLLM release checked | **v0.30.0**, published 2026-09-22 |
| How | release source tree, read directly: module paths, symbol definitions, class abstract surface, patch-tool targets, doc paths. `tools/check_vllm_compat.py <tree>` |
| Last GPU-qualified runtime | `0.28.1rc1.dev324`, the runtime behind the 0.4.2 validation table and the Qwen3.8 measurement in the README |
| What "checked" is not | not a boot qualification. Registration inside a live engine, `FULL_DECODE_ONLY` capture with the EXL3 ops, and the EP loader seams still need a GB10 run |

A vLLM release is not the same artifact as the fork runtime a recipe ships. The plugin is
built against a runtime, not against a version string; this page tracks the upstream release
because that is where the interfaces move and where a future recipe runtime will come from.

## CI does not cover this surface

CI installs `pytest`, `safetensors`, `numpy` and CPU `torch` — no vLLM. With vLLM absent,
`_VLLM_AVAILABLE` is false, so the vLLM-facing half of the plugin (quant-config
registration, `RoutedExperts`, the fused-MoE method base, custom-op registration) never
loads and is never imported by a test. The CPU suite is a check on the plugin's own logic.
Everything below was established by reading the v0.30.0 source tree.

## Verified present in v0.30.0

| Integration point | State |
|---|---|
| `vllm.general_plugins` entry-point group | present (with `endpoint`, `io_processor`, `platform`, `stat_logger` groups) |
| Every `from vllm... import ...` in `src/` and `tools/` | resolves |
| `Exl3MoEMethod` vs `FusedMoEMethodBase` abstract surface | vLLM requires `create_weights`, `get_fused_moe_quant_config`; both implemented |
| `register_quantization_config`, `QuantizationConfig`, `QuantizeMethodBase` | present |
| `FusedMoEQuantConfig`, `RoutedExperts`, `LinearBase`/`LinearMethodBase`/`UnquantizedLinearMethod` | present |
| `VocabParallelEmbedding`, `ParallelLMHead` | present at `model_executor/layers/vocab_parallel_embedding.py` |
| `set_weight_attrs`, `get_accelerator_view_from_cpu_tensor` | present |
| `get_tp_group`, `get_ep_group`, tensor-parallel rank/world-size helpers | re-exported from `parallel_state` via `from ... import *` |
| `model_executor/model_loader/ep_weight_filter.py`, `model_executor/offloader/uva.py` | unchanged paths |
| `vllm.utils.torch_utils.direct_register_custom_op` | present |

## Changed in v0.30.0

### 1. The Qwen4Exp model tree moved

`vllm/model_executor/models/qwen4_exp/**` is now `vllm/models/qwen4_exp/**` (the same move
applies to the other new-style model subtrees; `model_executor/models/` still holds the rest).

The three scripts in `tools/patch_vllm_qwen4_exp/` already resolve
`<site-packages/vllm>/models/qwen4_exp/nvidia/...`, so they were already targeting the current
layout — only their README described the old path. That is corrected.

### 2. PLE n-gram tables: the old patch is obsolete, and the EXL3 path is closed

Two separate facts, and the second one matters more than the first.

**The patch has nothing left to do.** `nvidia/ple_layer.py` now constructs the n-gram table
with `quant_config=quant_config` itself, which is exactly what `patch_vllm_qwen4_ple.py`'s PLE
half used to add. The tool now detects that shape and reports it as obsolete instead of failing
its anchor check.

**But the layer no longer consults `quant_config.get_quant_method(...)`.** It selects its
storage format through `Qwen4ExpPLEEmbeddingMethod.from_quant_config(...)`, which raises for
anything that is not ModelOpt mixed precision, a ModelOpt-excluded layer, or `Fp8Config`:

```python
if not isinstance(quant_config, Fp8Config):
    raise NotImplementedError(
        "Qwen4Exp PLE embedding does not support quantization config "
        f"{type(quant_config).__name__}"
    )
```

So an EXL3-quantized n-gram table (`Exl3EmbeddingMethod`, the `ngram_embedding` spec) is not
reachable on v0.30.0 as it stands. Consequences by model line:

- **DeepSeek-V4.1-Flash is unaffected.** The V4.1 path has no Qwen4Exp PLE table and this
  plugin registers no embedding method for it. TP4/EP4 and TP2 serving do not touch this code.
- **Qwen3.8-Flash-Next packs with a quantized n-gram table are affected** on v0.30.0. The
  table's own `dequantize()` is called on the fused PLE path, so the layer's interface is
  required, not optional.

Two ways forward, both needing a GB10 run to qualify:

1. **Adapter.** Implement upstream's `Qwen4ExpPLEEmbeddingMethod` interface (`embedding`,
   `dequantize`, `apply`) over the plugin's trellis decode, and teach `from_quant_config` to
   return it for an `Exl3Config`. Note `requires_device_loading: bool = False` and the Fp8
   variant's expectations before copying the interface wholesale.
2. **Keep the n-gram table out of EXL3** for v0.30.0 runs and let the layer take its FP8 or
   unquantized path.

### 3. Doc paths corrected

`vllm/models/deepseek_v4_1/common/engram.py`, cited in `docs/CPU_OFFLOAD.md`, is now
`vllm/models/deepseek_v41/common/engram.py` (note the `v41`). The UVA offloader path
`vllm/model_executor/offloader/uva.py` is unchanged.

### 4. Removed APIs this plugin does not use

v0.30.0 removed the `seq_lens_cpu` / `num_computed_tokens_cpu` attention-metadata properties,
the `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` and `VLLM_MM_HASHER_ALGORITHM` environment
variables, the `use_fp4_indexer_cache` alias, GPTQ `g_idx` activation ordering, and the
`CUDA_VISIBLE_DEVICES` fallback on ROCm. Nothing in `src/`, `tools/` or `docs/` references any
of them.

### 5. One dead fallback (harmless)

`exl3.py` registers its n-gram custom op through
`vllm.utils.torch_utils.direct_register_custom_op`, falling back to
`vllm.utils.direct_register_custom_op` under `except ImportError`. The first path exists in
v0.30.0; the second no longer does. The fallback is never reached on a current runtime and is
left in place for older ones — the checker reports it as a NOTICE, not a failure.

## Upstream work in v0.30.0 that overlaps the recipe repos

Worth reading before attributing anything to the plugin's own history:
DeepSeek-V4.1-Flash model support is now upstream (#56214, #56228, #56208), with the MXFP8 KV
record on SM100 (#56893), async Engram prefetch with Engram DP sharding (#56512), collapsed
DSpark draft states before the SP all-gather (#56903), and the DSpark drafter no longer
inheriting uninitialized EPLB state (#56387). Persistent top-k also now falls back on
low-shared-memory GPUs (#54110), which is the condition the 1M-context Spark runs hit. The
repo overlays in the DeepSeek recipe were written against a runtime that owned less of this
graph than v0.30.0 does.

## Re-check procedure

```bash
# against a downloaded release
python tools/check_vllm_compat.py --tag v0.30.0

# against an installed package or an unpacked tree
python tools/check_vllm_compat.py /usr/local/lib/python3.12/dist-packages/vllm
python tools/check_vllm_compat.py ~/vllm-0.30.0/vllm
```

Exit status is 0 when nothing hard-broken was found. `NOTICE` lines are changes that need a
decision; `BAD` lines are unresolved imports, a missing entry-point group, an unsatisfied
abstract surface, or a patch anchor that no longer exists and has no known replacement shape.
