# Padded fixed-shape MoE (`p2b_fused_moe_padded`)

A second entry point for the DSV4.1-Flash EXL3 MoE that takes **fixed-shape** inputs and performs
**zero host synchronization**, so it can be captured into a CUDA graph.

## Why

The multi-K kernel (`p2b_fused_moe_mk`) is correct but reads per-expert K counts to the host to size
its work — a host sync per layer per step. That is legal eagerly and illegal under `torch.cuda.graph`
capture. This kernel takes the opposite trade: pad the batch to a compile-time maximum, pass sentinel
expert ids for the padding rows, and let the kernel skip them.

## Interface

```
p2b_fused_moe_padded(
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

- `x`: `[max_t, hidden]` fp16, padded. Rows past the real batch carry sentinel ids.
- `out`: `[max_t, hidden]` fp16, written in place (out-param contract: safe for capture, since a fresh
  allocation per call would change address on every replay).
- `k_*_tbl`: `[n_local]` int8 per-expert K values, read **on device** (this is what removes the sync).
- `mcg`: must match the `P2B_CB` the kernel was built with; see the codebook PR.

## Structure: the 9-stage split

`p2b_fused_moe_padded_kernel` was originally a single cooperative launch with 8 `grid.sync()` barriers.
A cooperative launch (`cudaLaunchCooperativeKernel`) **cannot be captured** — under the default capture
mode it is an illegal call, and under `relaxed` mode the graph records nothing.

It is therefore split into **9 ordinary same-stream kernels** (`p2b_moe_padded_stage0..8`), one per
phase of the original pipeline, replacing each `grid.sync()` with a kernel boundary. The launch count
goes from 1 to 9 per layer, but every launch is an ordinary capture-legal launch, and the whole sequence
captures and replays correctly.

One subtlety worth recording, because it silently produced wrong output: the original kernel zeroed its
`accum` buffer in its prologue. When the prologue is replicated into all 9 stage kernels, **stage 8
zeroes `accum` and then writes back the zeros** — the output becomes independent of the inputs. The fix
is `_ACCUM_ONCE`: only stage 0 runs the zeroing loop; stages 1-8 use a prologue without it. Verified
`stage0 zero=1 writeback=0`, `stage8 zero=0 writeback=1`.

## Verification

- Deterministic: repeated runs give bit-identical output (`0.0000000` delta).
- Parity vs the multi-K kernel: max abs delta `4.88e-4`, absmax `36.031`.
- Capture: the 9-stage sequence records and replays exactly (bucket-exact canary:
  `replay == eager, maxdiff 0.00000`).
- End-to-end: this is the kernel behind the 26.3 tok/s `FULL_DECODE_ONLY` result.

## Build

```bash
export CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include:/usr/local/cuda/include
export NVCC_APPEND_FLAGS="-Xptxas -v -DP2B_CB=2"   # cb=2 for the mul1 packs
bash tools/multik/build.sh --allow-drift --skip-smoke
```

Note: `NVCC_PREPEND_FLAGS` does not propagate in the multi-mode build script; `NVCC_APPEND_FLAGS` does.
