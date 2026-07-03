"""Standalone convergence-probe worker, decoupled from the training loop.

For each probe epoch the trainers (`train_vicreg_review_h5.py` and
`train_vicreg_review_h5_paired.py`) running with ``--probe-queue-dir`` emit:

  * a slim encoder checkpoint  ``<combo>/probe_ckpts/epNNN.pt``
  * a queue marker             ``<queue>/<combo>__epNNN.json``

This worker polls ``<queue>``, rebuilds the frozen encoder from each slim
checkpoint, runs the same sentiment / recommendation / tag-generalization /
text-variant probe battery the inline path used, and appends a row to the
combo's ``dual_probe_history.tsv`` (+ ``.jsonl``). Because it lives in its own
process, the probe's GPU forward / CPU cross-validation never blocks training.

Markers stay as ``*.json`` until a worker finishes them. Workers claim a marker
by creating an exclusive ``*.json.claim`` sidecar, refresh ``*.heartbeat`` while
working, and rename the marker to ``*.json.done`` or ``*.json.failed`` at the
end. The claim sidecar makes a single queue safe for many GPUs and many VMs.
The worker drains any backlog and then exits once ``--stop-file`` exists and
the queue is empty (the collect step touches that file after training finishes).

Run alongside the sweep, e.g.:

    python -u VICReg_review/probe_worker.py \
        --queue-dir VICReg_review/heads/cloud_full_sweep_a100/probe_queue \
        --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace

import h5py
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.logging_tee import run_with_optional_tee  # noqa: E402
from VICReg_review.train_tag_probe import load_frozen_encoder  # noqa: E402
from VICReg_review import train_vicreg_review_h5 as trainer_mod  # noqa: E402
from VICReg_review.train_vicreg_review_h5 import (  # noqa: E402
    append_probe_history,
    probe_report_and_row,
    read_history,
)


class LostClaim(RuntimeError):
    """Raised when another worker reclaimed this marker before commit."""


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_create_json(path: Path, payload: dict) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except (FileExistsError, OSError):
        return False
    try:
        os.write(fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    finally:
        os.close(fd)
    return True


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{int(time.time() * 1e6)}"
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))


def claim_path(marker: Path) -> Path:
    return marker.parent / f"{marker.name}.claim"


def heartbeat_path(marker: Path) -> Path:
    return marker.parent / f"{marker.name}.claim.heartbeat"


def _path_age_seconds(path: Path) -> float | None:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _claim_expired(marker: Path, timeout: float) -> bool:
    if timeout <= 0:
        return False
    cp = claim_path(marker)
    hp = heartbeat_path(marker)
    age = _path_age_seconds(hp if hp.exists() else cp)
    return age is not None and age > timeout


def _reclaim_expired_claim(marker: Path, timeout: float, worker_id: str) -> bool:
    cp = claim_path(marker)
    if not cp.exists() or not _claim_expired(marker, timeout):
        return False
    aside = cp.parent / f".{cp.name}.stale.{_safe_name(worker_id)}.{int(time.time() * 1000)}"
    try:
        os.rename(str(cp), str(aside))
    except OSError:
        return False
    for path in (aside, heartbeat_path(marker)):
        try:
            Path(path).unlink()
        except OSError:
            pass
    return True


class MarkerClaim:
    def __init__(self, marker: Path, payload: dict, refresh: float):
        self.marker = marker
        self.claim = claim_path(marker)
        self.heartbeat = heartbeat_path(marker)
        self.payload = payload
        self.token = str(payload["token"])
        self.refresh = max(1.0, float(refresh))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def owns(self) -> bool:
        rec = _read_json(self.claim)
        return bool(rec) and rec.get("token") == self.token

    def start(self) -> None:
        self._write_heartbeat()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def release(self, *, remove: bool) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if remove and self.owns():
            for path in (self.claim, self.heartbeat):
                try:
                    path.unlink()
                except OSError:
                    pass

    def _write_heartbeat(self) -> None:
        if not self.owns():
            return
        payload = {**self.payload, "heartbeat_ts": time.time()}
        _atomic_write_json(self.heartbeat, payload)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.refresh):
            if not self.owns():
                return
            try:
                self._write_heartbeat()
            except OSError:
                return


def acquire_marker_claim(marker: Path, args, device: torch.device) -> MarkerClaim | None:
    if not marker.exists():
        return None
    worker_id = str(args.worker_id)
    payload = {
        "worker_id": worker_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "device": str(device),
        "token": uuid.uuid4().hex,
        "started_ts": time.time(),
        "claim_timeout": float(args.claim_timeout),
    }
    if not _atomic_create_json(claim_path(marker), payload):
        if _reclaim_expired_claim(marker, float(args.claim_timeout), worker_id):
            if not _atomic_create_json(claim_path(marker), payload):
                return None
        else:
            return None
    claim = MarkerClaim(marker, payload, float(args.claim_refresh))
    claim.start()
    return claim


@contextlib.contextmanager
def probe_history_lock(tsv_path: Path, args):
    lock = Path(f"{tsv_path}.lock")
    token = uuid.uuid4().hex
    payload = {
        "worker_id": str(args.worker_id),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "token": token,
        "ts": time.time(),
    }
    deadline = time.time() + max(1.0, float(args.history_lock_timeout))
    stale_after = float(args.history_lock_stale_seconds)
    while True:
        if _atomic_create_json(lock, payload):
            break
        age = _path_age_seconds(lock)
        if stale_after > 0 and age is not None and age > stale_after:
            aside = lock.parent / f".{lock.name}.stale.{_safe_name(args.worker_id)}.{int(time.time() * 1000)}"
            try:
                os.rename(str(lock), str(aside))
                aside.unlink(missing_ok=True)
                continue
            except OSError:
                pass
        if time.time() >= deadline:
            raise TimeoutError(f"timed out waiting for probe history lock: {lock}")
        time.sleep(0.25)
    try:
        yield
    finally:
        rec = _read_json(lock)
        if rec and rec.get("token") == token:
            try:
                lock.unlink()
            except OSError:
                pass


def _same_probe_row(row: dict, epoch: int, global_step: int) -> bool:
    try:
        return int(row.get("epoch")) == int(epoch) and int(row.get("global_step")) == int(global_step)
    except (TypeError, ValueError):
        return False


def marker_already_recorded(marker: Path) -> bool:
    """True when an older/non-claim worker already appended this marker's row.

    This makes migration from the old single-worker drain safe: if that worker
    wrote the history row but died before renaming the marker to ``*.done``, the
    new distributed worker will claim the marker, notice the existing row, and
    only publish the missing done marker.
    """
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        epoch = int(data["epoch"])
        global_step = int(data["global_step"])
        tsv_path = Path(data["probe_history_tsv"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return any(_same_probe_row(row, epoch, global_step) for row in read_history(tsv_path))


def append_probe_history_once(tsv_path: Path, row: dict, report: dict) -> bool:
    """Append unless this epoch/global_step is already present.

    The caller holds ``probe_history_lock``. Returns True when a new row was
    written, False when this marker had already been materialized by a previous
    worker.
    """
    try:
        epoch = int(row["epoch"])
        global_step = int(row["global_step"])
    except (KeyError, TypeError, ValueError):
        append_probe_history(tsv_path, row, report)
        return True
    if any(_same_probe_row(existing, epoch, global_step) for existing in read_history(tsv_path)):
        return False
    append_probe_history(tsv_path, row, report)
    return True


def _h5_has_vector_data(h5_path: str) -> bool:
    """False for a resident-mode meta-H5, whose 'vectors' dataset is shape-only
    (chunked, never written -- reads would SILENTLY return zeros)."""
    with h5py.File(h5_path, "r") as h5:
        ds = h5["vectors"]
        try:
            return ds.id.get_num_chunks() > 0
        except Exception:
            return True   # not chunked / old h5py: assume a real source H5


def resolve_vector_source(saved: dict, input_h5: str, fallback_h5: str | None,
                          local_data_dir: str | None = None):
    """Where do this probe's sentence vectors actually come from?

    Ladder (mirrors the trainer's resident design):
      1. the machine-local resident .dat named in the training args, when it
         exists here -> gather-by-pointer, offsets/labels from the meta-H5;
      2. a .dat staged locally for ``--fallback-h5`` under ``--local-data-dir``
         (by eval.ipynb's staging cell, or a training run on this pod) ->
         NVMe gather on ANY pod; the probe re-reads vectors for EVERY marker,
         so this is what keeps a big drain off the network FS;
      3. ``input_h5`` itself, when its 'vectors' dataset holds real data
         (streaming-mode combos point at the source H5 already);
      4. ``--fallback-h5`` (the shared source H5) -> slow but correct anywhere;
      5. otherwise fail LOUDLY -- never probe the meta-H5's shape-only zeros.
    Returns (vectors_dat_descriptor_or_None, h5_path_for_probe).
    """
    desc = trainer_mod.resident_descriptor(saved.get("vectors_dat"), input_h5)
    if desc is not None:
        return desc, input_h5
    if fallback_h5 and local_data_dir:
        dat_path = trainer_mod.default_vectors_dat_path(fallback_h5, local_data_dir)
        meta_h5 = trainer_mod.default_offsets_h5_path(fallback_h5, local_data_dir)
        desc = trainer_mod.resident_descriptor(dat_path, fallback_h5)
        if desc is not None:
            # Offsets/labels from the local companion meta-H5 when staged,
            # else from the shared source H5 (small index reads only).
            if Path(meta_h5).exists():
                return desc, str(meta_h5)
            if Path(fallback_h5).exists():
                return desc, str(fallback_h5)
    if Path(input_h5).exists() and _h5_has_vector_data(input_h5):
        return None, input_h5
    if fallback_h5 and Path(fallback_h5).exists() and _h5_has_vector_data(str(fallback_h5)):
        return None, str(fallback_h5)
    raise RuntimeError(
        f"no vector source on this machine: input_h5={input_h5} is missing or "
        f"shape-only (resident meta-H5), the training .dat "
        f"({saved.get('vectors_dat')}) is not here, and no usable --fallback-h5 "
        f"was given. Run the drain on the pod that staged the .dat, or pass "
        f"--fallback-h5 <shared embedding_h5.h5>."
    )


def compute_marker(marker: Path, device: torch.device, fallback_h5: str | None = None,
                   local_data_dir: str | None = None) -> tuple[Path, dict, dict]:
    data = json.loads(marker.read_text(encoding="utf-8"))
    ckpt_path = str(data["checkpoint"])
    tsv_path = Path(data["probe_history_tsv"])
    input_h5 = str(data["input_h5"])
    epoch = int(data["epoch"])
    global_step = int(data["global_step"])

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = dict(checkpoint.get("args", {}))
    vectors_desc, probe_h5 = resolve_vector_source(saved, input_h5, fallback_h5, local_data_dir)

    with h5py.File(probe_h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])

    model, _cfg, _ep, _gs = load_frozen_encoder(ckpt_path, input_dim, device)
    # The marker is authoritative for where output goes and which device we run on.
    saved["probe_history_tsv"] = str(tsv_path)
    saved["device"] = str(device)
    saved["input_h5"] = str(probe_h5)
    saved_args = SimpleNamespace(**saved)

    # load_game_views consults the trainer module's _VECTORS_DAT global (that is
    # how training itself gathers in resident mode); point it at the local .dat
    # for this marker and always reset afterwards.
    trainer_mod._VECTORS_DAT = vectors_desc
    try:
        result = probe_report_and_row(model, saved_args, device, epoch, global_step)
    finally:
        trainer_mod._VECTORS_DAT = None
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if result is None:
        raise RuntimeError("probe_report_and_row returned None (probe failed)")
    report, row = result
    return tsv_path, row, report


def process_marker(marker: Path, device: torch.device, fallback_h5: str | None = None,
                   local_data_dir: str | None = None) -> dict:
    tsv_path, row, report = compute_marker(
        marker, device, fallback_h5=fallback_h5, local_data_dir=local_data_dir
    )
    append_probe_history(tsv_path, row, report)
    return row


def pending_markers(queue_dir: Path) -> list[Path]:
    # "*.json" excludes already-processed *.json.done/.failed and in-flight *.tmp.
    return sorted(queue_dir.glob("*.json"))


def main_loop(args) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not args.worker_id:
        args.worker_id = f"{socket.gethostname()}:{os.getpid()}:{device}"
    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    stop_file = Path(args.stop_file) if args.stop_file else (queue_dir / "STOP")
    poll = max(0.5, float(args.poll_interval))
    processed = 0
    failed = 0
    skipped = 0
    print(
        f"probe_worker: watching {queue_dir} device={device} "
        f"worker={args.worker_id} stop_file={stop_file} poll={poll}s "
        f"run_once={args.run_once}",
        flush=True,
    )
    while True:
        markers = pending_markers(queue_dir)
        if not markers:
            if stop_file.exists():
                print(
                    f"probe_worker: stop file present and queue drained; "
                    f"processed={processed} failed={failed} skipped={skipped}",
                    flush=True,
                )
                break
            if args.run_once:
                break
            time.sleep(poll)
            continue
        claimed_any = False
        for marker in markers:
            claim = acquire_marker_claim(marker, args, device)
            if claim is None:
                skipped += 1
                continue
            claimed_any = True
            try:
                if marker_already_recorded(marker):
                    if not claim.owns():
                        raise LostClaim(f"lost claim before done marker: {marker.name}")
                    marker.rename(marker.parent / (marker.name + ".done"))
                    processed += 1
                    print(
                        f"probe_worker: materialized old completed marker {marker.name} "
                        f"(processed={processed} failed={failed} skipped={skipped})",
                        flush=True,
                    )
                    continue
                tsv_path, row, report = compute_marker(
                    marker, device, fallback_h5=args.fallback_h5,
                    local_data_dir=args.local_data_dir,
                )
                if not claim.owns():
                    raise LostClaim(f"lost claim before commit: {marker.name}")
                with probe_history_lock(tsv_path, args):
                    if not claim.owns():
                        raise LostClaim(f"lost claim before history append: {marker.name}")
                    appended = append_probe_history_once(tsv_path, row, report)
                if not claim.owns():
                    raise LostClaim(f"lost claim before done marker: {marker.name}")
                marker.rename(marker.parent / (marker.name + ".done"))
                processed += 1
                print(
                    f"probe_worker: {'done' if appended else 'materialized'} {marker.name} "
                    f"(processed={processed} failed={failed} skipped={skipped})",
                    flush=True,
                )
            except LostClaim as exc:
                skipped += 1
                print(f"probe_worker: skipped {marker.name}: {exc}", flush=True)
            except BaseException as exc:  # noqa: BLE001 - one bad job must not stop the worker
                failed += 1
                tb = traceback.format_exc()
                (marker.parent / (marker.name + ".error.txt")).write_text(
                    f"{type(exc).__name__}: {exc}\n\n{tb}", encoding="utf-8"
                )
                if claim.owns() and marker.exists():
                    try:
                        marker.rename(marker.parent / (marker.name + ".failed"))
                    except OSError:
                        pass
                print(
                    f"probe_worker: FAILED {marker.name}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            finally:
                claim.release(remove=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if args.run_once:
            break
        if not claimed_any:
            time.sleep(poll)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", required=True, type=Path, help="Probe job queue directory (shared with the trainers' --probe-queue-dir).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker-id", default=None,
                        help="Stable worker identity for distributed claims. Defaults to host:pid:device.")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between polls when the queue is empty.")
    parser.add_argument("--stop-file", default=None, help="Worker exits once this file exists and the queue is empty. Defaults to <queue-dir>/STOP.")
    parser.add_argument("--run-once", action="store_true", help="Process the current backlog once, then exit (for testing).")
    parser.add_argument("--claim-timeout", type=float, default=21600.0,
                        help="Seconds before a marker claim is considered stale. 0 disables stale-claim reclaim.")
    parser.add_argument("--claim-refresh", type=float, default=60.0,
                        help="Seconds between heartbeat writes while processing a claimed marker.")
    parser.add_argument("--history-lock-timeout", type=float, default=3600.0,
                        help="Seconds to wait for the per-combo probe-history append lock.")
    parser.add_argument("--history-lock-stale-seconds", type=float, default=21600.0,
                        help="Seconds before a probe-history append lock is considered stale. 0 disables lock reclaim.")
    parser.add_argument("--fallback-h5", default=None,
                        help="Shared source embedding H5 with REAL vectors. Used when a marker's "
                             "input_h5 is a resident-mode meta-H5 (shape-only vectors) and the "
                             "machine-local .dat from training is not on this pod -- makes the "
                             "drain runnable on any machine.")
    parser.add_argument("--local-data-dir", default="/root/data",
                        help="Machine-local dir checked for a resident .dat staged for "
                             "--fallback-h5 (plus its companion meta-H5). When present, every "
                             "probe gathers vectors from local NVMe instead of re-reading the "
                             "shared H5 per marker. Stage it with eval.ipynb's staging cell or "
                             "train_vicreg_review_h5.stage_resident_vectors.")
    parser.add_argument("--logout-address", default=None, help="Append stdout/stderr to this log file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_with_optional_tee(args.logout_address, main_loop, args)


if __name__ == "__main__":
    main()
