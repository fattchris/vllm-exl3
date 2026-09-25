# vllm-exl3 0.5.0

Serving EXL3 trellis-quantized checkpoints under vLLM. This release adds the routed-expert MoE
kernels that make tensor-parallel MoE practical across four Sparks, fixes three defects that changed
served output or silently dropped weights, and records what the current upstream vLLM release does to
this plugin's integration surface.

Nothing here changes a default serving path. The new kernels are exported, not selected: enabling
them is a recipe decision, and each is described with the boundary that applies to it.

## The MoE kernels

**Multi-K fused MoE** (`p2b_fused_moe_mk`) keeps per-expert K tables on the device, so a layer with
heterogeneous per-expert bit widths takes one cooperative launch instead of one launch per expert.

**Fixed-shape padded MoE** (`p2b_fused_moe_padded`) has fixed shapes and no host synchronization, so
it can be captured into a CUDA graph. The single cooperative launch it replaces cannot be captured at
all — `cudaLaunchCooperativeKernel` is illegal under default capture mode, and relaxed mode records
nothing — so it is split into nine ordinary same-stream stage kernels. Splitting it surfaced a silent
bug that had been introduced during the split: the prologue's `accum` zeroing had replicated into all
nine stages, so stage 8 zeroed `accum` and wrote the zeros back, making the output independent of the
inputs. Fixed in the same change.

**Grouped padded MoE** groups the live slots by expert, up to sixteen per group, so each trellis tile
is decoded once per expert rather than once per slot. `P2B_GROUPED=0` reverts to the ungrouped path.

## MoE-TP4 without re-quantizing the pack

`2304 / 4 = 576` is not 128-aligned, and EXL3's Hadamard block requires alignment, so an even
four-way split of the routed experts cuts a block and decodes every shard against the wrong
transform. That is why expert parallelism was the safe configuration and MoE-TP4 required a
re-quantized 640-wide pack.

`VLLM_EXL3_MOE_TP_ALIGN=128` removes the requirement. The router's columns are cut in whole
128-blocks — 640 / 640 / 512 / 512 — from the pack that already exists, with the trellis tile
indices landing exactly on block boundaries. The contributor measures +16% single-stream decode on
four Sparks against the expert-parallel configuration. Unset behaviour is unchanged, and a geometry
that cannot be block-aligned now raises at load rather than falling back to the even split, which is
the silent wrong-transform case the aligned path exists to remove.

## Correctness fixes

**The native-MoE dispatch gate rejected an unset codebook flag.** `main` failed 103 tests — 100 in
`test_routing_parity.py`, 3 in `test_activation_parity.py` — because an absent
`_exl3_codebook_flags` was compared against the MCG tuple and treated as a mismatch, returning
before the code the tests exercise. The default now resolves to the MCG tuple wherever it is read,
and a layer carrying a genuinely non-MCG codebook still fails closed.

**The expert codebook is now a build-time parameter.** The packs use the **mul1** codebook
(`0x83DCD12D`), not MCG (`0xCBAC1FED`), and a cb=1 kernel called against a mul1 pack does not crash —
it decodes every expert to a plausible-looking but wrong vector. Build with `-DP2B_CB=2` for mul1
packs. All three codebook checks now fail closed.

**The loader pre-filtered speculative-draft expert weights with the main stack's window.** A V4.1
pack carries two expert counts in one checkpoint: the main stack's 384 and the DSpark draft's 128.
The pre-read EP weight filter was sized once from the main stack and applied to both, so EP ranks 1
to 3 silently loaded none of the draft experts they owned, with no error. The contributor measures
+9% to +20% decode at acceptance about 1.7 to 2.0.

## Load path and host memory

The GB10 load path now uses pinned bounce buffers, `POPULATE_READ`, batched markers and a prescan
cache: the four-rank load drops from about 8.7 to about 5 minutes and fill from 7.5 to 1.6 minutes,
with decode unchanged. Routed-expert trellis arenas can be placed in pinned host memory for UVA runs,
trellises are copied directly into their final arena slots, and the load path is UMA-safe (direct-fill
plus MADV after the H2D). The bounded SAGE NVMe and Engram cache components are shared rather than
duplicated, and a gapped-safetensors repair rebuilds a compressed shard without changing tensor bytes.

## vLLM 0.30.0

The upstream release this integration surface is now checked against is **vLLM v0.30.0**, released
2026-09-22. [`docs/VLLM_COMPATIBILITY.md`](docs/VLLM_COMPATIBILITY.md) records the per-point state,
and `tools/check_vllm_compat.py <vllm-tree>` re-runs the check against any tree or tag:

- the `vllm.general_plugins` entry-point group is still present;
- every `from vllm... import ...` in `src/` and `tools/` resolves;
- `FusedMoEMethodBase` requires exactly `create_weights` and `get_fused_moe_quant_config`, both of
  which `Exl3MoEMethod` implements;
- the Qwen4Exp model tree moved from `vllm/model_executor/models/` to `vllm/models/`. The patch tools
  already resolved the new layout; only their README described the old one.

Two things need a decision rather than a patch. The PLE n-gram table: current vLLM builds it with a
quant config itself, so the PLE half of `patch_vllm_qwen4_ple.py` is obsolete and now reports itself
as such instead of failing its anchor check — but the EXL3 path itself does not carry over, because
the layer selects its storage format through `Qwen4ExpPLEEmbeddingMethod.from_quant_config`, which
raises `NotImplementedError` for any config that is not ModelOpt or `Fp8Config`. An adapter is
required, and the compatibility page sets out the two options. DeepSeek-V4.1 serving does not touch
this code. Separately, `exl3.py`'s `except ImportError` fallback to `vllm.utils.direct_register_custom_op`
now points at a path that no longer exists; the primary path is current and the fallback is only
reached on older runtimes.

## Validation

| check | result |
|---|---|
| CI on the tagged commit | 6/6 jobs green: lint and bytecode, unit and attribution suite, documentation links, packaging build, aarch64 patch against exllamav3 v1.5.0 and against the Spark pin |
| Unit suite | 481 passed, 31 skipped |
| vLLM 0.30.0 integration surface | verified by reading the release tree with `tools/check_vllm_compat.py`; one NOTICE (the guarded fallback above) |

What that does not cover: CI installs no vLLM and has no GPU, so the plugin's vLLM-facing half never
loads there and no kernel runs. The performance figures in this document are the ones their authors
measured on their own GB10 configurations; they were not re-measured for this tag. The MoE kernel
batch needs a GB10 qualification run — parity against the Python `LinearEXL3` reference, then the
standard gate — before a recipe adopts it.

## Upgrading

Rebuild the extension. Packs carrying the **mul1** codebook need `-DP2B_CB=2` through
`NVCC_APPEND_FLAGS`; MCG packs build as before. To try the padded or grouped MoE paths, size
`VLLM_EXL3_PADDED_MAX_T` for the batches you actually serve and set `VLLM_EXL3_MOE_TP_ALIGN=128` only
on a runtime where the routed-expert geometry is block-aligned. Qwen3.8-Flash-Next packs with an
EXL3-quantized n-gram table should stay on their pinned runtime until the PLE adapter lands.
