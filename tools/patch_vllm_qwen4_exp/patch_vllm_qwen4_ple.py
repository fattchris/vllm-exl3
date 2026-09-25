"""Two plumbing lines in vLLM's qwen4_exp so the quant config is consulted for
lm_head and for the PLE n-gram table.

  nvidia/model.py   ParallelLMHead(...)              gains quant_config=self.quant_config
  nvidia/ple_layer.py PLEVocabParallelEmbedding(...) gains quant_config=quant_config

Without these vLLM builds both layers unquantized regardless of --quantization,
so an EXL3 pack whose lm_head / n-gram rows are trellis tensors cannot load.
Backs up each file to <file>.orig (kept if present), idempotent, compile-checked.

The PLE half only applies to vLLM builds that still construct the n-gram table
without a quant config. Current vLLM does that itself and selects the table's
storage format through Qwen4ExpPLEEmbeddingMethod.from_quant_config, which rejects
an EXL3 config outright; this tool reports that as obsolete rather than failing.
See docs/VLLM_COMPATIBILITY.md.

usage: python3 patch_vllm_qwen4_ple.py <site-packages/vllm>
"""
import os
import shutil
import sys

MODEL_OLD = '''        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
'''
MODEL_NEW = '''        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
'''
PLE_OLD = '''        self.ngram_embedding = PLEVocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
'''
PLE_NEW = '''        self.ngram_embedding = PLEVocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            quant_config=quant_config,
            prefix=f"{prefix}.ngram_embedding",
'''

# Markers of the current upstream shape, where the n-gram table is built with a quant
# config already and the layer picks its own storage format.
UPSTREAM_PLE_MARKERS = (
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpPLEEmbeddingMethod",
    "from_quant_config",
)


def patch(path: str, old: str, new: str) -> bool:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return False
    src = open(path, encoding="utf-8").read()
    if new in src:
        print(f"already patched: {path}")
        return True
    n = src.count(old)
    if n != 1:
        print(f"ERROR: anchor found {n} times in {path} (need 1)")
        return False
    out = src.replace(old, new)
    try:
        compile(out, path, "exec")
    except SyntaxError as e:
        print(f"ERROR: patched source does not compile: {e}")
        return False
    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
    open(path, "w", encoding="utf-8").write(out)
    print(f"patched {path} (backup {backup})")
    return True


def patch_ple(path: str) -> bool:
    """The PLE n-gram table half of this tool.

    Two upstream shapes exist:

    * older vLLM built the n-gram table without a quant config, so the layer had to
      be handed one -- that is what ``PLE_OLD``/``PLE_NEW`` do;
    * current vLLM builds it with ``quant_config=quant_config`` itself and picks the
      storage format through ``Qwen4ExpPLEEmbeddingMethod.from_quant_config``, which
      raises ``NotImplementedError`` for any config that is not ModelOpt or
      ``Fp8Config``.

    On the current shape there is nothing to add here, and an EXL3-quantized n-gram
    table is not reachable through that selector at all. That needs an adapter
    implementing the upstream embedding-method interface; see
    ``docs/VLLM_COMPATIBILITY.md``.
    """
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return False
    src = open(path, encoding="utf-8").read()
    if PLE_NEW in src:
        print(f"already patched: {path}")
        return True
    if PLE_OLD in src:
        return patch(path, PLE_OLD, PLE_NEW)
    if any(marker in src for marker in UPSTREAM_PLE_MARKERS):
        print(
            f"obsolete on this vLLM: {os.path.basename(path)} already builds the n-gram table "
            "with a quant config and selects its storage format through "
            "Qwen4ExpPLEEmbeddingMethod.from_quant_config. Nothing to patch. An EXL3-quantized "
            "n-gram table needs the adapter described in docs/VLLM_COMPATIBILITY.md."
        )
        return True
    print(f"ERROR: neither the patch anchor nor a known upstream shape found in {path}")
    return False


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    root = os.path.join(sys.argv[1], "models", "qwen4_exp", "nvidia")
    ok = patch(os.path.join(root, "model.py"), MODEL_OLD, MODEL_NEW)
    ok = patch_ple(os.path.join(root, "ple_layer.py")) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
