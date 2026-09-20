# Reproducing the measured results

End-to-end recipe for a third party. Everything here is on the public repos; no
cluster-specific paths or credentials are required.

Target: **4x DGX Spark (GB10, SM121), TP4/EP4, bs1, k=2 spec decode** —
**26-27 tok/s repeat**, correct output at every state.

## 0. What you need

| | |
|---|---|
| Hardware | 4x NVIDIA DGX Spark (GB10). Fewer nodes will run but the measured numbers are for 4. |
| Model pack | a DeepSeek-V4.1-Flash **EXL3** pack (this is what `-DP2B_CB=2` targets) |
| Base repo | `vcruz305/vllm-exl3` at `main` |
| Kernel PRs | **#31** (multi-K), **#32** (codebook), **#33** (padded) — stack in that order |
| Toolchain | CUDA 13.0, `nvcc` for `sm_121`, Python 3.12, PyTorch with CUDA |

Verify the pack's codebook before building — this is the single most common
mistake. The pack's own flags are the truth source:

```python
# after the layer's finalize() has run
layer._exl3_codebook_flags        # (mcg, mul1) x (gate, up, down)
# (False, True, False, True, False, True)  -> mul1 -> build with -DP2B_CB=2
# (True, False, True, False, True, False)  -> MCG  -> build with -DP2B_CB=1
```

A mismatch does **not** crash: it decodes every expert to a plausible-looking but
wrong vector. That is why the kernel's `TORCH_CHECK` is written to fail closed.

## 1. Build the kernel

```bash
git clone https://github.com/vcruz305/vllm-exl3 && cd vllm-exl3
git fetch origin pull/31/head:pr31 && git fetch origin pull/32/head:pr32 && git fetch origin pull/33/head:pr33
git checkout pr33            # pr33 contains pr31 and pr32

export CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include:/usr/local/cuda/include
export NVCC_APPEND_FLAGS="-Xptxas -v -DP2B_CB=2"     # 2 for mul1 packs; see step 0
bash tools/multik/build.sh --src-tree "$PWD"
```

Three flags matter, and all three have bitten us:

| flag | why |
|---|---|
| `TORCH_CUDA_ARCH_LIST=12.1a` | GB10 is SM121; the `a` suffix is required |
| `NVCC_APPEND_FLAGS` | the only route that reaches `nvcc` in this recipe — **`NVCC_PREPEND_FLAGS` silently does not propagate** |
| `-DP2B_CB=<1\|2>` | expert codebook; mismatch fails closed rather than decoding garbage |

## 2. Smoke-check the artifact

```bash
python3 tools/multik/parity_sanity.py --so build/vllm_exl3_c*.so
python -m pytest tests/ -q        # the numeric contracts; should be green
```

`parity_sanity.py` proves the extension loads, exposes the entry points, and that
repeated launches are deterministic (the property CUDA-graph replay depends on).
Numeric parity against the `LinearEXL3` reference is the test suite's job.

## 3. Serve

Deploy the built `.so` into your serving image, then serve with the graph config
that was measured:

```yaml
max-model-len: 4096
enforce-eager: false
enable-prefix-caching: true
speculative-config:
  method: dspark
  num_speculative_tokens: 2
  draft_sample_method: probabilistic
  rejection_sample_method: block
  quantization: mxfp4
compilation-config:
  cudagraph_mode: FULL_DECODE_ONLY
  cudagraph_capture_sizes: [3, 6, 9, 12]
```

`num_speculative_tokens: 2` is the measured optimum on this hardware — k=5 gave
22.4 tok/s, k=2 gives 26.0. Larger k costs more per step than the extra accepted
tokens return.

## 4. Measure it correctly

**Use `llm-inference-bench`** (https://github.com/local-inference-lab/llm-inference-bench).
Ad-hoc timing scripts over-report by 3-5x because they measure short bursts
rather than sustained serving throughput.

```bash
python3 llm_decode_bench.py \
  --host <IP> --port <PORT> --model <MODEL> \
  --concurrency 1 --contexts 0,2048,3072 \
  --max-tokens 512 --duration 30 \
  --display-mode plain --no-hw-monitor --output results.json
```

Keep `ctx <= 3072` when serving at `max-model-len: 4096`, or the cell 400s.

### Two measurement traps

1. **The model emits ~12 reasoning tokens before any content.** Gates using
   `max_tokens: 8-48` see empty `content` with `finish_reason: length` and score
   it as a wrong answer. Use `max_tokens >= 160` and require
   `finish_reason == "stop"`. This produced a multi-day false alarm here.
2. **Compare `steps/s`, not tok/s, across different acceptance.** tok/s scales
   with acceptance at fixed step time; the tool reports both.

## Expected results

| config | ctx0 | repeat | ctx2048 |
|---|---|---|---|
| Python `LinearEXL3` loop (baseline) | — | 16.1 | — |
| native kernel, eager | 14.7 | 21.2 | 15.6 |
| + `FULL_DECODE_ONLY` graphs | 18.6 | 26.3 | 17.8 |

Quality gate at every state, temp 0: `The capital of France is` -> `Paris`.

Per-domain, after the draft-load fix (see the recipe repo):

| domain | ctx0 tok/s | ctx2048 tok/s | acceptance |
|---|---|---|---|
| code | 27.02 | 27.01 | 1.89 / 1.92 |
| prose | 27.44 | 30.21 | 1.97 / 2.21 |
| structured | 27.73 | 29.70 | 1.99 / 2.14 |

## If your numbers are lower

Check these in order — each has cost us a day:

1. **Codebook mismatch** (`-DP2B_CB`). Symptom: acceptance collapses to exactly
   1.00, short completions come back empty. The kernel now refuses rather than
   decoding wrong, so a startup `TORCH_CHECK` failure means exactly this.
2. **Draft experts not loading.** On EP, the loader's pre-read filter is sized
   from `n_routed_experts` (384) but the DSpark draft has 128
   (`dspark_n_routed_experts`), so ranks 1-3 silently load **zero** draft expert
   weights. Symptom: acceptance flat and low across every domain, and **code no
   faster than prose** — the opposite of the expected code premium. Fix and
   probe: `tools/engram/draft_ep_filter_fix.py` in the recipe repo.
3. **Graph mode actually engaged.** `Capturing CUDA graphs (FULL...)` in the log
   proves capture, not dispatch. Confirm you are not silently falling back to
   eager.
4. **Benchmark method.** See the two traps above.
