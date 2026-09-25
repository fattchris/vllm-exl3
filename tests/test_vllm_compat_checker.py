"""Tests for tools/check_vllm_compat.py.

CI has no vLLM installed, so the checker is exercised against synthetic vLLM trees
built in tmp_path. These cover the parts with logic in them: module resolution,
wildcard re-export following, fallback-import classification, the entry-point group,
and the abstract-surface comparison.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECKER = REPO / "tools" / "check_vllm_compat.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_vllm_compat_under_test", CHECKER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_tree(tmp_path: Path, *, entry_point: bool = True, extra_abstract: str | None = None) -> Path:
    """A minimal vllm/ tree with the pieces the checker looks at."""
    root = tmp_path / "vllm"
    (root / "plugins").mkdir(parents=True)
    (root / "model_executor" / "layers" / "fused_moe").mkdir(parents=True)
    if entry_point:
        (root / "plugins" / "__init__.py").write_text(
            'GROUPS = ["vllm.general_plugins", "vllm.platform_plugins"]\n', encoding="utf-8"
        )
    else:
        (root / "plugins" / "__init__.py").write_text('GROUPS = ["vllm.platform_plugins"]\n', encoding="utf-8")
    abstract = ["create_weights", "get_fused_moe_quant_config"]
    if extra_abstract:
        abstract.append(extra_abstract)
    body = "\n".join(
        f"    @abstractmethod\n    def {name}(self, *a, **k):\n        raise NotImplementedError\n"
        for name in abstract
    )
    (root / "model_executor" / "layers" / "fused_moe" / "fused_moe_method_base.py").write_text(
        "from abc import abstractmethod\n\n\nclass FusedMoEMethodBase:\n" + body, encoding="utf-8"
    )
    return root


def test_entry_point_group_found(checker, tmp_path):
    root = make_tree(tmp_path)
    failures: list[str] = []
    assert checker.check_entry_point(root, failures) == 0
    assert failures == []


def test_entry_point_group_missing_is_a_failure(checker, tmp_path):
    root = make_tree(tmp_path, entry_point=False)
    failures: list[str] = []
    assert checker.check_entry_point(root, failures) == 1
    assert failures


def test_abstract_surface_satisfied_by_exl3_moe_method(checker, tmp_path):
    root = make_tree(tmp_path)
    failures: list[str] = []
    assert checker.check_abstract_surface(root, failures) == 0
    assert failures == []


def test_abstract_surface_reports_a_new_requirement(checker, tmp_path):
    # A method vLLM adds that Exl3MoEMethod does not implement must be reported.
    root = make_tree(tmp_path, extra_abstract="some_new_required_hook")
    failures: list[str] = []
    assert checker.check_abstract_surface(root, failures) == 1
    assert any("some_new_required_hook" in f for f in failures)


def test_find_module_resolves_package_and_module(checker, tmp_path):
    root = make_tree(tmp_path)
    assert checker.find_module(str(root), "vllm.plugins") is not None
    assert checker.find_module(str(root), "vllm.model_executor.layers.fused_moe.fused_moe_method_base") is not None
    assert checker.find_module(str(root), "vllm.not.a.real.module") is None


def test_names_follow_wildcard_reexports(checker, tmp_path):
    """`from .x import *` must be followed, or real vLLM re-exports read as missing."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "inner.py").write_text("def reexported_helper():\n    return 1\n", encoding="utf-8")
    (pkg / "__init__.py").write_text("from .inner import *\n", encoding="utf-8")
    names = checker._names_in(str(pkg / "__init__.py"))
    assert "reexported_helper" in names


def test_guarded_imports_are_classified_as_fallbacks(checker, tmp_path):
    """A fallback import in an `except ImportError` block is a NOTICE, not a failure."""
    p = tmp_path / "mod.py"
    p.write_text(
        "try:\n    from vllm.utils.torch_utils import direct_register_custom_op\nexcept ImportError:\n"
        "    from vllm.utils import direct_register_custom_op\n",
        encoding="utf-8",
    )
    guarded = checker._guarded_linenos(str(p))
    assert 4 in guarded, "the except-handler import should be marked guarded"
    assert 2 not in guarded, "the primary import should not be"
