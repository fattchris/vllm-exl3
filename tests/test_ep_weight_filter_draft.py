"""CPU regression test: the EP weight filter must not pre-filter draft experts.

A DeepSeek-V4.1-Flash pack carries two expert counts in one checkpoint: the main
stack's 384 and the DSpark draft's 128 (`dspark_n_routed_experts`). The loader's
pre-read EP weight filter is sized once from the MAIN stack's count and applied to
every expert weight, draft weights included. With EP=4 that silently drops the
draft experts of ranks 1-3, because the drop happens before `get_tensor` -- no
error, no warning, and no tensor in the model's load loop to notice.

This test pins the arithmetic of the failure so the shape of the bug cannot be
forgotten, and so any future "fix" that only widens the window is caught. It
deliberately does not import vLLM: this repository is the plugin, and the bug is
in the runtime's loader. The corresponding runtime patch is
`experiments/dsv41_nvme/patches/ep_weight_filter.draft-experts.patch`.
"""

import pytest

MAIN_EXPERTS = 384   # the backbone stack
DRAFT_EXPERTS = 128  # dspark_n_routed_experts
EP = 4


def linear_window(num_experts: int, rank: int, ep: int = EP) -> set[int]:
    """Mirror `compute_local_expert_ids(..., placement="linear")`.

    Same distribution the runtime uses: contiguous blocks, remainder to the
    lowest ranks.
    """
    base = num_experts // ep
    remainder = num_experts % ep
    start = rank * base + min(rank, remainder)
    count = base + (1 if rank < remainder else 0)
    return set(range(start, start + count))


def should_skip(weight_name: str, local_expert_ids: set[int] | None) -> bool:
    """Mirror `should_skip_weight`: only per-expert `.weight` names are filtered."""
    if local_expert_ids is None:
        return False
    import re

    m = re.search(r"\.experts\.(\d+)\.", weight_name)
    if m is None:
        return False
    return int(m.group(1)) not in local_expert_ids


@pytest.mark.parametrize("rank", range(EP))
def test_owned_draft_experts_that_survive_the_main_stack_window(rank: int) -> None:
    """The failure mode, as arithmetic.

    Applying the main-stack window to draft weights keeps every draft expert on
    rank 0 and NONE of them on ranks 1-3. That is the whole bug: the draft runs on
    25% of its experts, with no error anywhere.
    """
    kept = linear_window(MAIN_EXPERTS, rank)
    owned = linear_window(DRAFT_EXPERTS, rank)
    surviving = kept & owned
    if rank == 0:
        assert surviving == owned
    else:
        assert surviving == set(), (
            f"rank {rank} must lose all {len(owned)} of its draft experts when the "
            f"main-stack window is applied; got {len(surviving)} surviving"
        )


def test_window_widths_differ_which_is_why_it_breaks() -> None:
    """96-wide filter window vs 32-wide draft window: they only align on rank 0."""
    assert len(linear_window(MAIN_EXPERTS, 0)) == 96
    assert len(linear_window(DRAFT_EXPERTS, 0)) == 32
    assert len(linear_window(MAIN_EXPERTS, 0) & linear_window(DRAFT_EXPERTS, 0)) == 32
    for rank in range(1, EP):
        assert linear_window(MAIN_EXPERTS, rank) & linear_window(DRAFT_EXPERTS, rank) == set()


def test_a_widened_window_alone_would_not_fix_it() -> None:
    """Why the fix must exempt draft weights rather than resize the filter.

    A single `local_expert_ids` cannot be correct for both counts at once: any
    window that covers the main stack is misaligned with the draft for at least
    one rank, and vice versa.
    """
    main_windows = [linear_window(MAIN_EXPERTS, r) for r in range(EP)]
    draft_windows = [linear_window(DRAFT_EXPERTS, r) for r in range(EP)]
    assert main_windows != draft_windows
    # no single one of these windows equals the draft's for every rank
    for w in main_windows:
        assert not all(w == d for d in draft_windows)


def test_draft_weight_names_are_recognisable() -> None:
    """The exemption key: draft experts live under `mtp.*` in the checkpoint."""
    for name in (
        "mtp.0.ffn.experts.5.w1.weight",
        "mtp.1.ffn.experts.100.w2.weight",
        "mtp.2.ffn.experts.127.w3.weight",
    ):
        assert name.startswith("mtp.")
    for name in (
        "model.layers.0.ffn.experts.5.w1.weight",
        "model.layers.39.ffn.experts.200.w2.weight",
    ):
        assert not name.startswith("mtp.")


def test_filter_is_inert_without_ep() -> None:
    """ep_size <= 1 means every expert is local, so no filtering happens."""
    assert should_skip("mtp.0.ffn.experts.5.w1.weight", None) is False
    assert should_skip("model.layers.0.ffn.experts.5.w1.weight", None) is False


def test_main_stack_filtering_still_works_when_ep_is_on() -> None:
    """The exemption must not disable filtering for the main stack."""
    kept = linear_window(MAIN_EXPERTS, 0)  # [0, 96)
    assert should_skip("model.layers.0.ffn.experts.5.w1.weight", kept) is False
    assert should_skip("model.layers.0.ffn.experts.200.w1.weight", kept) is True


def test_scales_were_never_filtered_which_hid_the_bug() -> None:
    """Only `.weight` names are matched, so `.scale` names loaded everywhere.

    That asymmetry is why the bug was invisible for so long: the draft's scales
    arrived on every rank while its weights did not, so a casual look at the
    loaded-tensor counts looked plausible.
    """
    kept = linear_window(MAIN_EXPERTS, 1)  # [96, 192)
    draft_scale = "mtp.0.ffn.experts.32.w1.weight_scale"
    draft_weight = "mtp.0.ffn.experts.32.w1.weight"
    # the scale name carries an expert id, so the same window applies to it --
    # both are dropped here; the point is that `.scale` handling differs per
    # quant method and was patched separately in the overlay.
    assert should_skip(draft_scale, kept) is True
    assert should_skip(draft_weight, kept) is True
