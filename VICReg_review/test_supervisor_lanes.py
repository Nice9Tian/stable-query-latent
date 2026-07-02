"""Lane-selection policy: every GPU stays active; full-vs-queue is a per-combo
decision (_ram_budget), never a permanent GPU disable (see the L40x2/A100x5 bug)."""

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
