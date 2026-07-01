"""File-based job/result protocol between supervisor and worker(s).

Mirrors the probe_queue idiom (on-disk, restart-safe). All functions take a
``qdir`` -- the queue directory a single worker polls. For a single-GPU sweep
that is ``<out_dir>/sweep_jobs``; for multi-GPU each lane gets its own subdir
(``<out_dir>/sweep_jobs/gpu<N>``) so lane workers never steal each other's jobs.

Per queue dir: the supervisor writes one ``<combo_id>.job.json`` and waits for
``<combo_id>.result.json``; the worker polls for pending jobs, trains, writes the
result. ``worker.heartbeat`` carries the worker PID + current combo; ``worker.ready``
signals startup (calibration) is done; ``STOP`` tells the worker to exit.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

_JOB_SUFFIX = ".job.json"


def default_qdir(out_dir) -> Path:
    """The single-GPU queue dir. Multi-GPU lanes use default_qdir(out_dir)/gpuN."""
    return Path(out_dir) / "sweep_jobs"


def job_path(qdir, combo_id: str) -> Path:
    return Path(qdir) / f"{combo_id}{_JOB_SUFFIX}"


def result_path(qdir, combo_id: str) -> Path:
    return Path(qdir) / f"{combo_id}.result.json"


def heartbeat_path(qdir) -> Path:
    return Path(qdir) / "worker.heartbeat"


def ready_path(qdir) -> Path:
    return Path(qdir) / "worker.ready"


def stop_path(qdir) -> Path:
    return Path(qdir) / "STOP"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def atomic_write(path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_job(qdir, job: dict) -> None:
    atomic_write(job_path(qdir, job["combo_id"]), job)


def write_result(qdir, result: dict) -> None:
    atomic_write(result_path(qdir, result["combo_id"]), result)


def read_result(qdir, combo_id: str) -> dict | None:
    return read_json(result_path(qdir, combo_id))


def clear_combo(qdir, combo_id: str) -> None:
    Path(job_path(qdir, combo_id)).unlink(missing_ok=True)
    Path(result_path(qdir, combo_id)).unlink(missing_ok=True)
    (Path(qdir) / f"{combo_id}{_JOB_SUFFIX}.done").unlink(missing_ok=True)


def pending_jobs(qdir) -> list[Path]:
    qdir = Path(qdir)
    if not qdir.exists():
        return []
    out = []
    for p in sorted(qdir.glob(f"*{_JOB_SUFFIX}")):
        combo_id = p.name[: -len(_JOB_SUFFIX)]
        if not result_path(qdir, combo_id).exists():
            out.append(p)
    return out


def mark_job_consumed(job_file) -> None:
    # Tolerant: the supervisor may have already cleared the job (e.g. it reacted
    # to the result and moved on) -- renaming a gone file must not crash the worker.
    try:
        Path(job_file).rename(str(job_file) + ".done")
    except OSError:
        pass


def write_heartbeat(qdir, pid: int, combo_id: str | None = None) -> None:
    atomic_write(heartbeat_path(qdir), {"pid": int(pid), "ts": _now(), "combo_id": combo_id})


def write_ready(qdir, pid: int) -> None:
    atomic_write(ready_path(qdir), {"pid": int(pid), "ts": _now()})


def write_stop(qdir) -> None:
    atomic_write(stop_path(qdir), {"ts": _now()})


def clear_signals(qdir) -> None:
    for p in (ready_path(qdir), stop_path(qdir)):
        Path(p).unlink(missing_ok=True)
