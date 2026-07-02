import math

import numpy as np

from VICReg_review import oom_proxy


def test_standard_transient_matches_full_chunk_attention_score():
    transient = oom_proxy.estimate_standard_transient_bytes(
        worst_game_sentences=534_641,
        view=0.8,
        num_latents=1024,
    )

    assert math.isclose(transient / oom_proxy.GIB, 13.05, rel_tol=0.01)


def test_view80_1024_latents_routes_to_split_on_48gb_card():
    gib = oom_proxy.GIB
    calib = {
        "_meta": {"input_dim": 1024},
        "1024|standard": {"C": 8.52 * 1024, "R": 0.31 * gib},
        "1024|split_recompute": {"C": 1.0, "R": 0.18 * gib},
    }

    plan = oom_proxy.plan_combo_chunked(
        calib,
        worst_game_sentences=534_641,
        free_vram_bytes=44.31 * gib,
        num_latents=1024,
        view=0.8,
        batch_size=128,
        safety=0.85,
        try_paired=False,
        total_sentences=2_422_551,
        cache_bytes=0,
        ram_budget=0,
    )

    assert plan["backward_mode"] == "split_recompute"
    assert plan["standard_peak_gib"] == 31.8
    assert plan["standard_transient_gib"] == 13.05
    assert plan["standard_required_gib"] > plan["budget_gib"]
    assert plan["stem_chunk_size"] >= int(534_641 * 0.8)
    assert plan["stem_chunk_size"] < 1_000_000


def test_same_combo_can_use_standard_when_required_memory_fits():
    gib = oom_proxy.GIB
    calib = {
        "_meta": {"input_dim": 1024},
        "1024|standard": {"C": 8.52 * 1024, "R": 0.31 * gib},
        "1024|split_recompute": {"C": 1.0, "R": 0.18 * gib},
    }

    plan = oom_proxy.plan_combo_chunked(
        calib,
        worst_game_sentences=534_641,
        free_vram_bytes=80.0 * gib,
        num_latents=1024,
        view=0.8,
        batch_size=128,
        safety=0.85,
        try_paired=False,
        total_sentences=2_422_551,
        cache_bytes=0,
        ram_budget=0,
    )

    assert plan["backward_mode"] == "standard"
    assert plan["standard_required_gib"] < plan["budget_gib"]


def test_subset_batch_sentences_is_expected_plus_worst():
    stats = oom_proxy.GameStats(
        sentence_counts=np.array([100, 200, 300, 400, 500], dtype=np.int64),
        appids=["a", "b", "c", "d", "e"],
        input_dim=8,
    )
    # whole pool total=1500, worst=500. batch_size=2 -> expected 1500*2/5=600, +500 headroom.
    assert stats.subset_batch_worst_sentences(0, seed=1, anchor_appids=[], batch_size=2) == 1100
    # batch_size >= num_games -> the whole subset sum (one batch holds every game).
    assert stats.subset_batch_worst_sentences(0, seed=1, anchor_appids=[], batch_size=10) == 1500
    assert stats.subset_batch_worst_sentences(0, seed=1, anchor_appids=[], batch_size=10) \
        == stats.subset_total_sentences(0, seed=1, anchor_appids=[])


def test_per_batch_sentences_reclaim_standard_for_large_n():
    """Log scenario: n=500, batch=128, view=0.2 on a 44GiB card. Budgeting standard
    against the whole 500-game subset forces split_recompute; budgeting it against
    one 128-game batch fits standard -- same calib, same card."""
    gib = oom_proxy.GIB
    calib = {
        "_meta": {"input_dim": 1024},
        "512|standard": {"C": 8.59 * 1024, "R": 0.5 * gib},
        "512|split_recompute": {"C": 1.0, "R": 0.18 * gib},
    }
    common = dict(
        worst_game_sentences=850_071, free_vram_bytes=44.31 * gib, num_latents=512,
        view=0.2, batch_size=128, safety=0.85, try_paired=False,
        total_sentences=19_859_200, cache_bytes=0, ram_budget=0,
    )

    over = oom_proxy.plan_combo_chunked(calib, **common)                       # old behaviour
    fixed = oom_proxy.plan_combo_chunked(calib, standard_batch_sentences=5_000_000, **common)

    assert over["backward_mode"] == "split_recompute"
    assert over["standard_required_gib"] > over["budget_gib"]
    assert fixed["backward_mode"] == "standard"
    assert fixed["standard_required_gib"] < fixed["budget_gib"]
