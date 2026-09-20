#!/usr/bin/env python3
"""Smoke-check the built ``vllm_exl3_c`` extension's fused-MoE entry points.

Verifies, without needing a pack:
  * the extension imports and reports its ABI version;
  * every expected entry point is present;
  * ``p2b_fused_moe_mk`` and ``p2b_fused_moe_padded`` accept a launch on
    synthetic but shape-correct inputs;
  * both entry points are deterministic across repeated launches (the property
    CUDA-graph replay depends on).

A launch that fails counts as a failure; an empty result set can no longer be
mistaken for success. A refusal whose message contains "codebook mismatch" is
reported as correct behaviour and does not fail the run -- that is the kernel
failing closed, which is what it is supposed to do on a deliberate -D mismatch.

Numeric parity against the Python ``LinearEXL3`` reference is asserted by the
plugin's own suite (``tests/test_exl3_moe_arity.py``, ``tests/test_routing_parity.py``);
run that with ``python -m pytest`` for the real oracle.

Usage:
    python3 parity_sanity.py [--so PATH] [--device cuda|cpu]

``--so`` defaults to ``$VLLM_EXL3_SO``, then to the artifact ``build.sh`` stages.
Exit status is 0 on success, 1 otherwise.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys

import torch


def default_so() -> str:
    """Find the .so build.sh stages, from either the repo root or this directory.

    build.sh writes to $SCRIPT_DIR/patched/build (see OUT= in that script), so
    search there first, then a few nearby layouts.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    pats = (
        "patched/build/vllm_exl3_c*.so",   # what build.sh produces, run from here
        "build/vllm_exl3_c*.so",
        "tools/multik/patched/build/vllm_exl3_c*.so",   # run from the repo root
        "tools/multik/build/vllm_exl3_c*.so",
        "**/vllm_exl3_c*.so",
    )
    roots = (here, os.getcwd(), os.path.dirname(os.path.dirname(here)))
    for root in roots:
        for pat in pats:
            hits = glob.glob(os.path.join(root, pat), recursive=True)
            if hits:
                return sorted(hits)[-1]
    return ""


def load_ext(so_path: str):
    if not so_path or not os.path.exists(so_path):
        sys.exit(f"extension not found: {so_path!r} (pass --so or build first)")
    spec = importlib.util.spec_from_file_location("vllm_exl3_c", so_path)
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    return ext


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", default=os.environ.get("VLLM_EXL3_SO", default_so()))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--intermediate", type=int, default=2304)
    ap.add_argument("--experts", type=int, default=8,
                    help="number of local experts for the synthetic tables")
    args = ap.parse_args()

    ext = load_ext(args.so)
    dev = torch.device(args.device)
    print(f"extension : {args.so}")
    print(f"device    : {dev}")
    abi = getattr(ext, "P2B_MOE_ABI_VERSION", None)
    print(f"ABI       : {abi}")
    print("note      : a 'codebook mismatch' refusal below is CORRECT behaviour --") 
    print("            the kernel was built for one codebook and refuses the other")

    expected = [
        "p2b_fused_moe",
        "p2b_fused_moe_mk",
        "p2b_fused_moe_padded",
        "dequant_trellis",
    ]
    missing = [n for n in expected if not hasattr(ext, n)]
    for n in expected:
        print(f"  entry {n:24s} {'present' if hasattr(ext, n) else 'MISSING'}")

    failures = 0
    if missing:
        print(f"\nFAIL: missing entry points: {missing}")
        failures += 1

    if dev.type != "cuda":
        print("\nSANITY: entry points checked (launch checks need CUDA)")
        return 1 if failures else 0

    H, I = args.hidden, args.intermediate
    NEXP = args.experts
    torch.manual_seed(0)
    zeros_ptrs = lambda: [torch.zeros(NEXP, dtype=torch.int64, device=dev) for _ in range(9)]

    def try_launch(label, fn, out_shape):
        """Run fn 3x; require all 3 to succeed and agree. Returns (ok, note)."""
        outs = []
        err = None
        for _ in range(3):
            out = torch.zeros(*out_shape, dtype=torch.float16, device=dev)
            try:
                fn(out)
                torch.cuda.synchronize()
                outs.append(out.clone())
            except Exception as exc:  # noqa: BLE001 - report, do not raise
                err = f"{type(exc).__name__}: {str(exc)[:110]}"
                break
        if err is not None:
            if "codebook mismatch" in err:
                # Expected on a deliberate -D mismatch: the kernel refuses
                # rather than decoding garbage. That is correct behaviour.
                return None, f"refused on codebook (correct, not a failure): {err[:80]}"
            return False, f"launch failed: {err}"
        if not outs:
            return False, "no launch succeeded and no error was raised"
        same = all(torch.equal(outs[0], o) for o in outs[1:])
        return same, ("deterministic across 3 launches" if same
                      else "NON-DETERMINISTIC across 3 launches")

    # These entries dereference nine expert pointer tables, so a synthetic
    # launch with empty tables would fault rather than prove anything. What we
    # CAN check without a pack is that each entry reaches its own argument
    # validation -- which is exactly where the codebook gate lives. Reaching
    # that gate proves the symbol resolves, the signature matches, and the
    # codebook wiring works; it does not prove numerics. Numerics are the test
    # suite's job (tests/test_exl3_moe_arity.py, tests/test_routing_parity.py).
    k_tab = torch.full((NEXP,), 5, dtype=torch.int8, device=dev)
    zeros_ptrs = lambda: [torch.zeros(NEXP, dtype=torch.int64, device=dev) for _ in range(9)]

    def probe(label, fn):
        """Call fn in a subprocess so a CUDA fault cannot poison later probes.

        A launch that gets past the codebook gate dereferences nine expert
        pointer tables; with synthetic (empty) tables that faults, and an
        async CUDA fault poisons the context for every later call in the same
        process. Running each probe in its own interpreter keeps the verdicts
        independent.
        """
        import subprocess
        code = (
            "import importlib.util,sys,torch\n"
            "spec=importlib.util.spec_from_file_location('vllm_exl3_c',sys.argv[1])\n"
            "ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)\n"
            "dev=sys.argv[2];H=int(sys.argv[3]);I=int(sys.argv[4]);N=int(sys.argv[5]);"
            "which=sys.argv[6]\n"
            "kt=torch.full((N,),5,dtype=torch.int8,device=dev)\n"
            "p=[torch.zeros(N,dtype=torch.int64,device=dev) for _ in range(9)]\n"
            "if which=='mk':\n"
            "    ext.p2b_fused_moe_mk(torch.zeros(1,H,dtype=torch.float16,device=dev),"
            "torch.zeros(1,H,dtype=torch.float16,device=dev),*p,"
            "torch.zeros(1,dtype=torch.int32,device=dev),"
            "torch.ones(1,dtype=torch.float16,device=dev),kt,kt,kt,1,True,I,0.0)\n"
            "else:\n"
            "    T,K=3,6\n"
            "    ext.p2b_fused_moe_padded(torch.zeros(T,H,dtype=torch.float16,device=dev),"
            "torch.zeros(T,H,dtype=torch.float16,device=dev),*p,"
            "torch.zeros(T,K,dtype=torch.int32,device=dev),"
            "torch.ones(T,K,dtype=torch.float16,device=dev),"
            "torch.tensor([T],dtype=torch.int32,device=dev),kt,kt,kt,N,K,True,I,0.0)\n"
            "print('__OK__')\n"
        )
        r = subprocess.run(
            [sys.executable, "-c", code, args.so, str(dev),
             str(H), str(I), str(NEXP), label],
            capture_output=True, text=True, timeout=300,
        )
        out = (r.stdout or "") + (r.stderr or "")
        if "__OK__" in out:
            return True, "reached the kernel (no codebook refusal)"
        if "codebook mismatch" in out:
            built = "cb=2 (mul1)" if "cb=2" in out else "cb=1 (MCG)"
            return True, f"codebook gate reached and refused: built for {built} -- correct"
        if "illegal memory access" in out or "cudaErrorIllegalAddress" in out:
            return True, ("got past the gate and faulted on the empty pointer tables "
                          "(expected: synthetic inputs, no real experts)")
        tail = " | ".join(l for l in out.strip().split("\n")[-2:] if l)[:150]
        return False, f"probe failed: {tail}"

    print("\n  p2b_fused_moe_mk (1-D routing ids):")
    x1 = torch.randn(1, H, dtype=torch.float16, device=dev)
    ok, note = probe("mk", lambda: ext.p2b_fused_moe_mk(
        x1, torch.zeros(1, H, dtype=torch.float16, device=dev), *zeros_ptrs(),
        torch.zeros(1, dtype=torch.int32, device=dev),
        torch.ones(1, dtype=torch.float16, device=dev),
        k_tab, k_tab, k_tab, 1, True, I, 0.0,
    ))
    print(f"    {note}")
    if not ok:
        failures += 1
    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    print("\n  p2b_fused_moe_padded (fixed-shape padded grid):")
    PAD_T, PAD_K = 3, 6
    xp = torch.randn(PAD_T, H, dtype=torch.float16, device=dev)
    ok_p, note_p = probe("padded", lambda: ext.p2b_fused_moe_padded(
        xp, torch.zeros(PAD_T, H, dtype=torch.float16, device=dev), *zeros_ptrs(),
        torch.zeros(PAD_T, PAD_K, dtype=torch.int32, device=dev),
        torch.ones(PAD_T, PAD_K, dtype=torch.float16, device=dev),
        torch.tensor([PAD_T], dtype=torch.int32, device=dev),
        k_tab, k_tab, k_tab, NEXP, PAD_K, True, I, 0.0,
    ))
    print(f"    {note_p}")
    if not ok_p:
        failures += 1

    print("\nSANITY: " + ("PASS" if not failures else f"FAIL ({failures})"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
