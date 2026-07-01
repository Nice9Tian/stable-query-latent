"""Torch-free host-RAM accounting shared by the planner and the notebook monitor.

Kept dependency-light (stdlib + optional psutil) ON PURPOSE: ``oom_proxy`` imports
torch and the whole trainer, so the monitor cannot cheaply reuse its helpers.
This module is the single source of truth for "how much RAM can we actually use",
so what the monitor displays == what the sweep planner budgets against.

cgroup note: ``memory.current`` counts page cache (reclaimable file pages) as
"used". Reading the 164GB embedding H5 fills that cache and would otherwise look
like ~0 free. The real pressure is ``anon + shmem + unevictable`` (non-reclaimable);
page cache is dropped by the kernel on demand and must NOT be treated as used.
"""

from __future__ import annotations

from pathlib import Path

_LIMIT_PATHS = ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
_CURRENT_PATHS = ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")
_STAT_PATHS = ("/sys/fs/cgroup/memory.stat", "/sys/fs/cgroup/memory/memory.stat")


def _read_int_file(path: str):
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _first_int(paths) -> int | None:
    for p in paths:
        v = _read_int_file(p)
        if v is not None:
            return v
    return None


def _read_cgroup_stat() -> dict:
    for path in _STAT_PATHS:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        stat = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    stat[parts[0]] = int(parts[1])
                except ValueError:
                    pass
        if stat:
            return stat
    return {}


def _cgroup_limit() -> int | None:
    limit = _first_int(_LIMIT_PATHS)
    return limit if (limit and 0 < limit < (1 << 60)) else None


def _non_reclaimable(stat: dict) -> int | None:
    """anon (+ shmem + unevictable) = memory that cannot be freed on demand.
    v2 exposes 'anon'; v1 exposes 'rss'. Returns None if neither is present."""
    anon = stat.get("anon", stat.get("rss"))
    if anon is None:
        return None
    return anon + stat.get("shmem", 0) + stat.get("unevictable", 0)


def available_ram_bytes() -> float:
    """Usable host RAM = limit - NON-reclaimable usage (anon + shmem + unevictable).

    Page cache (cgroup 'file') is reclaimable, so it is NOT subtracted. 0 means
    "unknown" -> callers should not force a RAM downgrade."""
    limit = _cgroup_limit()
    if limit is None:
        try:
            import psutil
            return float(psutil.virtual_memory().available)
        except Exception:
            return 0.0
    non_reclaimable = _non_reclaimable(_read_cgroup_stat())
    if non_reclaimable is not None:
        return float(max(0, limit - non_reclaimable))
    # last resort: limit - current (counts page cache; conservative)
    current = _first_int(_CURRENT_PATHS)
    return float(max(0, limit - (current or 0)))


def memory_breakdown() -> dict:
    """A displayable snapshot using the SAME accounting as available_ram_bytes().

    Returns bytes: {source, limit, real, cache, usable}. ``real`` is the
    non-reclaimable footprint (what can actually starve the next process),
    ``cache`` is reclaimable page cache, ``usable`` == available_ram_bytes()."""
    limit = _cgroup_limit()
    if limit is not None:
        stat = _read_cgroup_stat()
        real = _non_reclaimable(stat)
        return {
            "source": "cgroup",
            "limit": float(limit),
            "real": float(real if real is not None else 0),
            "cache": float(stat.get("file", stat.get("cache", 0))),
            "usable": available_ram_bytes(),
        }
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"source": "host", "limit": float(vm.total),
                "real": float(vm.total - vm.available),
                "cache": float(getattr(vm, "cached", 0)),
                "usable": float(vm.available)}
    except Exception:
        return {"source": "n/a", "limit": 0.0, "real": 0.0, "cache": 0.0, "usable": 0.0}
