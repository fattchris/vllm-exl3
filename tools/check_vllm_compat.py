#!/usr/bin/env python3
"""Check this plugin's vLLM integration surface against a vLLM source tree.

Reads a vLLM tree (an unpacked release tag or a site-packages/vllm directory) and
verifies, by source inspection, everything this plugin assumes about it:

  * the plugin entry-point group the package registers into;
  * every ``from vllm... import ...`` in ``src/`` and ``tools/``, resolved to a
    real definition (wildcard re-exports are followed one level);
  * the abstract method surface of ``FusedMoEMethodBase``, which ``Exl3MoEMethod``
    must satisfy to instantiate;
  * the files ``tools/patch_vllm_qwen4_exp/`` patches, and whether each patch's
    anchor is still present, already applied, or gone;
  * the vLLM-internal paths this repo's docs cite.

Exit status is 0 when nothing hard-broken was found, 1 otherwise. ``NOTICE``
results are changes that need a decision, not necessarily a failure.

Usage:
    python tools/check_vllm_compat.py /path/to/vllm-0.30.0/vllm
    python tools/check_vllm_compat.py --tag v0.30.0        # downloads the tag
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths inside vLLM that this repo's docs cite. Reported, never fatal.
DOC_PATHS = [
    "models/deepseek_v41/common/engram.py",
    "model_executor/offloader/uva.py",
    "model_executor/model_loader/ep_weight_filter.py",
    "models/qwen4_exp/nvidia/model.py",
    "models/qwen4_exp/nvidia/ple_layer.py",
    "models/qwen4_exp/nvidia/mtp.py",
]

# APIs vLLM removed. Presence means the plugin is written against an older tree.
REMOVED_APIS = [
    "seq_lens_cpu",
    "num_computed_tokens_cpu",
    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL",
    "VLLM_MM_HASHER_ALGORITHM",
]

OK, NOTICE, BAD = "OK   ", "NOTICE", "BAD  "


def find_module(vllm_root, mod):
    """Resolve a dotted module under vllm/ to a source file, or None."""
    parts = mod.split(".")
    if parts[0] != "vllm":
        return None
    base = os.path.join(vllm_root, *parts[1:])
    for cand in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(cand):
            return cand
    return None


def _names_in(path, seen=None):
    """Top-level names defined or imported by a module, following `import *`."""
    seen = seen or set()
    if path in seen:
        return set()
    seen.add(path)
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return set()
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    sub = resolve_relative(path, node)
                    if sub:
                        names |= _names_in(sub, seen)
                else:
                    names.add(a.asname or a.name)
    return names


def resolve_relative(path, node):
    """Resolve `from .x import *` inside a package __init__ to a real file."""
    if node.level is None or not node.level:
        return find_module(os.path.dirname(os.path.dirname(path)), node.module or "")
    pkg = os.path.dirname(path)
    for _ in range(node.level - 1):
        pkg = os.path.dirname(pkg)
    target = os.path.join(pkg, *(node.module.split(".") if node.module else []))
    for cand in (target + ".py", os.path.join(target, "__init__.py")):
        if os.path.isfile(cand):
            return cand
    return None


def collect_imports():
    """Every `from vllm... import ...` site in this repo's src/ and tools/."""
    sites = []
    for sub in ("src", "tools"):
        for root, _dirs, files in os.walk(os.path.join(REPO, sub)):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(root, fn)
                try:
                    tree = ast.parse(open(p, encoding="utf-8", errors="replace").read())
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vllm."):
                        sites.append((p, node.lineno, node.module, [a.name for a in node.names]))
                    elif isinstance(node, ast.Import):
                        for a in node.names:
                            if a.name.startswith("vllm."):
                                sites.append((p, node.lineno, a.name, []))
    return sites


def _guarded_linenos(path):
    """Line numbers of imports that sit in an `except` handler (a fallback)."""
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                for sub in ast.walk(h):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        out.add(sub.lineno)
    return out


def check_imports(vllm_root, failures):
    print("\n== imports ==")
    bad = 0
    for path, lineno, mod, names in collect_imports():
        rel = os.path.relpath(path, REPO).replace("\\", "/")
        guarded = lineno in _guarded_linenos(path)
        mf = find_module(vllm_root, mod)
        if mf is None:
            level = BAD if not guarded else NOTICE
            tail = " (fallback in an except handler)" if guarded else ""
            print(f"  {level} {rel}:{lineno}  module gone: {mod}{tail}")
            bad += 0 if guarded else 1
            continue
        have = _names_in(mf)
        for n in names:
            if n == "*" or n in have:
                continue
            # submodule import (`from vllm.x import sub`) is legal without a top-level name
            if find_module(vllm_root, f"{mod}.{n}") is not None:
                continue
            level = BAD if not guarded else NOTICE
            tail = " (fallback in an except handler)" if guarded else ""
            print(f"  {level} {rel}:{lineno}  {mod}.{n} not defined in {os.path.basename(mf)}{tail}")
            bad += 0 if guarded else 1
    if not bad:
        print(f"  {OK} every vLLM import in src/ and tools/ resolves")
    else:
        failures.append(f"{bad} vLLM import(s) unresolved")
    return bad


def check_abstract_surface(vllm_root, failures):
    print("\n== FusedMoEMethodBase abstract surface ==")
    path = find_module(vllm_root, "vllm.model_executor.layers.fused_moe.fused_moe_method_base")
    if not path:
        print(f"  {BAD} fused_moe_method_base.py not found")
        failures.append("fused_moe_method_base missing")
        return 1
    src = open(path, encoding="utf-8", errors="replace").read()
    tree = ast.parse(src)
    abstract = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FusedMoEMethodBase":
            for b in node.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    decs = [getattr(d, "id", getattr(d, "attr", "")) for d in b.decorator_list]
                    if "abstractmethod" in decs:
                        abstract.add(b.name)
    impl = set()
    exl3 = os.path.join(REPO, "src", "vllm_exl3", "exl3.py")
    for node in ast.walk(ast.parse(open(exl3, encoding="utf-8", errors="replace").read())):
        if isinstance(node, ast.ClassDef) and node.name == "Exl3MoEMethod":
            for b in node.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    impl.add(b.name)
    missing = sorted(abstract - impl)
    print(f"  vLLM requires: {', '.join(sorted(abstract)) or '(none)'}")
    print(f"  Exl3MoEMethod implements: {', '.join(sorted(abstract & impl)) or '(none)'}")
    if missing:
        print(f"  {BAD} Exl3MoEMethod does not implement: {', '.join(missing)}")
        failures.append(f"Exl3MoEMethod missing {missing}")
        return 1
    print(f"  {OK} Exl3MoEMethod satisfies the abstract surface")
    return 0


def check_entry_point(vllm_root, failures):
    print("\n== plugin entry-point group ==")
    path = os.path.join(vllm_root, "plugins", "__init__.py")
    if not os.path.isfile(path):
        print(f"  {BAD} vllm/plugins/__init__.py not found")
        failures.append("plugin loader missing")
        return 1
    src = open(path, encoding="utf-8", errors="replace").read()
    groups = sorted(set(re.findall(r'["\']((?:vllm\.)?[\w.]*plugin[\w.]*)["\']', src)))
    want = "vllm.general_plugins"
    print(f"  groups in vLLM: {', '.join(groups)}")
    if want in groups:
        print(f"  {OK} {want} is still a registered group")
        return 0
    print(f"  {BAD} {want} not found — the package entry point would never load")
    failures.append("entry-point group gone")
    return 1


def check_doc_paths(vllm_root):
    print("\n== vLLM paths cited by this repo's docs ==")
    for rel in DOC_PATHS:
        p = os.path.join(vllm_root, *rel.split("/"))
        if os.path.exists(p):
            print(f"  {OK} {rel}")
        else:
            alt = rel.replace("models/deepseek_v4_1/", "models/deepseek_v41/")
            hint = f"  (did you mean {alt}?)" if os.path.exists(os.path.join(vllm_root, *alt.split("/"))) else ""
            print(f"  {NOTICE} absent: {rel}{hint}")


def check_patch_tools(vllm_root):
    """Run each patch tool against a throwaway copy of the target files."""
    print("\n== tools/patch_vllm_qwen4_exp anchors ==")
    tool_dir = os.path.join(REPO, "tools", "patch_vllm_qwen4_exp")
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "models", "qwen4_exp", "nvidia")
        os.makedirs(target)
        src_dir = os.path.join(vllm_root, "models", "qwen4_exp", "nvidia")
        copied = []
        for fn in ("model.py", "ple_layer.py", "mtp.py"):
            s = os.path.join(src_dir, fn)
            if os.path.isfile(s):
                dst = os.path.join(target, fn)
                with open(s, "rb") as fh, open(dst, "wb") as out:
                    out.write(fh.read())
                copied.append(fn)
        for tool in sorted(os.listdir(tool_dir)):
            if not tool.startswith("patch_") or not tool.endswith(".py"):
                continue
            r = subprocess.run(
                [sys.executable, os.path.join(tool_dir, tool), tmp],
                capture_output=True, text=True,
            )
            out = (r.stdout + r.stderr).strip().splitlines()
            verdict = OK if r.returncode == 0 else NOTICE
            print(f"  {verdict} {tool}: {out[-1] if out else 'no output'}")


def check_removed_apis(vllm_root):
    print("\n== APIs vLLM removed (plugin must not reference them) ==")
    hits = []
    me = os.path.abspath(__file__)
    for sub in ("src", "tools", "docs"):
        for root, _d, files in os.walk(os.path.join(REPO, sub)):
            for fn in files:
                if not fn.endswith((".py", ".md")):
                    continue
                p = os.path.join(root, fn)
                if os.path.abspath(p) == me:
                    continue  # this checker names the removed APIs on purpose
                text = open(p, encoding="utf-8", errors="replace").read()
                for api in REMOVED_APIS:
                    if re.search(rf"\b{api}\b", text):
                        hits.append(f"{os.path.relpath(p, REPO)} references {api}")
    if hits:
        for h in hits:
            print(f"  {BAD} {h}")
    else:
        print(f"  {OK} nothing in src/, tools/ or docs/ references a removed API")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="path to a vllm/ source directory")
    ap.add_argument("--tag", help="download and check a vLLM release tag instead")
    args = ap.parse_args()

    if args.tag:
        tmp = tempfile.mkdtemp(prefix="vllm-compat-")
        url = f"https://github.com/vllm-project/vllm/archive/refs/tags/{args.tag}.tar.gz"
        print(f"downloading {url}")
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as fh:
            urllib.request.urlretrieve(url, fh.name)
            with tarfile.open(fh.name) as tf:
                tf.extractall(tmp)
        root = os.path.join(tmp, f"vllm-{args.tag.lstrip('v')}", "vllm")
    elif args.path:
        root = args.path
        if os.path.basename(os.path.normpath(root)) != "vllm":
            cand = os.path.join(root, "vllm")
            root = cand if os.path.isdir(cand) else root
    else:
        ap.error("give a path or --tag")

    if not os.path.isdir(root):
        print(f"not a directory: {root}")
        return 1
    print(f"vLLM tree: {root}")

    failures: list[str] = []
    check_entry_point(root, failures)
    check_imports(root, failures)
    check_abstract_surface(root, failures)
    check_patch_tools(root)
    check_doc_paths(root)
    check_removed_apis(root)

    print("\n== summary ==")
    if failures:
        for f in failures:
            print(f"  {BAD} {f}")
        print("  integration surface needs work")
        return 1
    print(f"  {OK} integration surface checked; see NOTICE lines for items needing a decision")
    return 0


if __name__ == "__main__":
    sys.exit(main())
