"""Lane-selection policy: every GPU stays active; full-vs-queue is a per-combo
decision (_ram_budget), never a permanent GPU disable (see the L40x2/A100x5 bug).
Plus the claim-order policy: full-train-count combos drain before any smaller n."""

from types import SimpleNamespace

from VICReg_review import oom_proxy
from VICReg_review.sweep.supervisor import Supervisor


def _fake_supervisor(pool_gib, cache_gib_by_combo):
    """A Supervisor with only the two methods _select_active_gpus calls, stubbed."""
    sup = Supervisor.__new__(Supervisor)                      # bypass __init__ (no GPU/config)
    sup._ram_pool_budget = lambda: pool_gib * oom_proxy.GIB
    sup._cache_bytes_for_combo = lambda c: cache_gib_by_combo[c] * oom_proxy.GIB
    return sup


def test_keeps_all_gpus_when_heaviest_combo_exceeds_pool():
    # L40x2 profile: pool ~197GiB, heaviest combo (n2000 x view0.8) ~221GiB > pool.
    # Old code -> full_slots=1 -> one GPU parked for the whole run. Now: both stay.
    caches = {"light": 20.0, "medium": 55.0, "heavy": 221.0}
    sup = _fake_supervisor(197.0, caches)
    combos = ["light", "light", "medium", "heavy"]
    assert sup._select_active_gpus([0, 1], combos) == [0, 1]
    assert sup._select_active_gpus([0, 1, 2, 3], combos) == [0, 1, 2, 3]   # L40x4: all four
    assert sup._select_active_gpus([0, 1, 2, 3, 4], combos) == [0, 1, 2, 3, 4]  # A100x5: all five


def test_single_gpu_and_empty_combos_are_passthrough():
    sup = _fake_supervisor(197.0, {"light": 20.0})
    assert sup._select_active_gpus([0], ["light"]) == [0]
    assert sup._select_active_gpus([0, 1], []) == [0, 1]


class _FakeStats:
    input_dim = 1024

    def subset_total_sentences(self, n, seed, anchors):
        return int(n) * 1000

    def subset_worst_sentences(self, n, seed, anchors):
        return 50_000

    def subset_batch_worst_sentences(self, n, seed, anchors, batch_size):
        return 200_000


def _order_supervisor():
    sup = Supervisor.__new__(Supervisor)                      # bypass __init__ (no GPU/config)
    sup.config = SimpleNamespace(
        data_seed=SimpleNamespace(train_game_seed=1, anchors=[]),
        train=SimpleNamespace(batch_size=128),
        memory=SimpleNamespace(vram_safety=0.85),
    )
    sup.stats = _FakeStats()
    sup.calib = {}                                            # no calib -> conservative fallback plan
    sup._resident_enabled = True
    sup._free_vram = lambda: 80.0 * oom_proxy.GIB
    sup._total_vram = lambda: 80.0 * oom_proxy.GIB
    sup._ram_budget = lambda cache_bytes=0.0: 100.0 * oom_proxy.GIB
    return sup


def _combo(cid, n, view, latents=256):
    return SimpleNamespace(combo_id=cid, train_games=n, view=view, num_latents=latents)


def test_full_train_count_combos_are_ordered_first():
    # Interleave n values; every n=2000 combo must come before ANY smaller n,
    # regardless of view/latents difficulty -- next_claim walks this order, so
    # smaller n only starts once no full-n combo is left claimable.
    combos = [
        _combo("a_n200_v80", 200, 0.8, 1024),
        _combo("b_n2000_v20", 2000, 0.2),
        _combo("c_n1000_v60", 1000, 0.6),
        _combo("d_n2000_v80", 2000, 0.8, 1024),
        _combo("e_n500_v40", 500, 0.4),
        _combo("f_n2000_v40", 2000, 0.4),
    ]
    ordered = _order_supervisor()._order_combos(combos)
    ns = [c.train_games for c in ordered]
    assert ns[:3] == [2000, 2000, 2000], ns
    assert sorted(ns[3:]) == [200, 500, 1000], ns


def test_train_games_all_beats_explicit_counts():
    # 'all' (0) is the full pool -> it outranks every explicit count.
    combos = [_combo("x_n2000", 2000, 0.4), _combo("y_all", 0, 0.4), _combo("z_n500", 500, 0.4)]
    ordered = _order_supervisor()._order_combos(combos)
    assert [c.train_games for c in ordered][0] == 0
