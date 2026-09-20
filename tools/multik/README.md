# Multi-K fused MoE kernel (`p2b_fused_moe_mk`)

Adds a fused MoE entry point that reads **per-expert K values from the device**
instead of sizing its work from host-side counts, so a layer needs one launch
rather than one per expert. It is the kernel behind the native decode path.

- **Source:** `csrc/p2b_moe.cu` (+ `.cuh`, `csrc/bindings.cpp`)
- **Build:** `bash tools/multik/build.sh`
- **Smoke check:** `python3 tools/multik/parity_sanity.py --so build/vllm_exl3_c*.so`

## What it adds

```
p2b_fused_moe_mk(
    x, out,
    gate_trellis_ptrs, gate_suh_ptrs, gate_svh_ptrs,
    up_trellis_ptrs,   up_suh_ptrs,   up_svh_ptrs,
    down_trellis_ptrs, down_suh_ptrs, down_svh_ptrs,
    expert_indices, routing_weights,
    k_gate_tbl, k_up_tbl, k_down_tbl,
    n_local, mcg,
    intermediate_size=2048, swiglu_limit=0.0,
)
```

Nine `int64` pointer tables (three tensors x three projections) plus three `int8`
per-expert K tables. `out` is written **in place** — the out-param contract
matters for CUDA-graph capture, where a fresh allocation per call would change
address on every replay.

## Building

```bash
# EXL3_EXT_INCLUDE: the exllamav3 extension source tree (it ships the trellis
# decode headers the kernel includes). Point it at your exllamav3 checkout;
# build.sh also accepts EXL3_EXT_INCLUDE from the environment.
export EXL3_EXT_INCLUDE=/path/to/exllamav3
export CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include:/usr/local/cuda/include
export NVCC_APPEND_FLAGS="-Xptxas -v -DP2B_CB=2"
bash tools/multik/build.sh --src-tree /path/to/vllm_exl3
```

The kernel sources come from this repo's own `csrc/`; `build.sh` overlays them
onto a *copy* of `--src-tree`. An optional `tools/multik/patched/` directory
overrides the source location for local experiments and is not required.

Three flags matter and all three have bitten us:

| flag | why |
|---|---|
| `TORCH_CUDA_ARCH_LIST=12.1a` | GB10 is SM121; the `a` suffix is required for the arch-specific features |
| `NVCC_APPEND_FLAGS` | the only route that reaches `nvcc` in this recipe — **`NVCC_PREPEND_FLAGS` silently does not propagate** |
| `-DP2B_CB=<1\|2>` | expert codebook: `1` = MCG, `2` = mul1. A mismatch **fails closed** at the `TORCH_CHECK` rather than decoding garbage |

`--src-tree` defaults to `$VLLM_EXL3_SRC`, then `/opt/vllm-exl3`. The script
overlays the patched sources onto a *copy* of that tree; the installed plugin is
never mutated. The artifact and its `ptxas -v` log land in `build/`.

## K range

The kernel instantiates **K2..K6** (`cases 2,3,4,5,6` in the tile dispatch).
There is no K7/K8 instantiation; an out-of-range table value lands in the
`default:` arm, which zeroes the tile deterministically (and traps if built with
`-DP2B_STRICT_K`). The Python caller keeps any K7/K8 experts on the `LinearEXL3`
loop, which is where the pack's heterogeneous geometry is qualified.

## Verifying

```bash
python3 tools/multik/parity_sanity.py            # imports the .so, checks ABI + determinism
python -m pytest tests/test_exl3_moe_arity.py tests/test_routing_parity.py   # numeric contracts
```

`parity_sanity.py` is a smoke check: it proves the extension loads, exposes the
expected entry points, and that repeated launches are deterministic (the
property graph replay depends on). The **numeric oracle** is the plugin's own
test suite, which compares against the Python `LinearEXL3` reference.

## Measured

On 4x DGX Spark (GB10, TP4/EP4), bs1, DeepSeek-V4.1-Flash EXL3:

| path | decode tok/s |
|---|---|
| Python `LinearEXL3` loop | 16.1 |
| this kernel (native, eager) | 21.2 |
| + `FULL_DECODE_ONLY` CUDA graphs, k=2 spec decode | 26.3 |

Correct output at every stage (`The capital of France is` -> `Paris`, temp 0).
Measured with `llm-inference-bench` sustained 30 s cells at c=1 — see
`tools/padded/README.md` for the graph-side kernel and the measurement caveats.
