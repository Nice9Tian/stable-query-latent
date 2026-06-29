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

Markers are renamed ``*.json.done`` on success and ``*.json.failed`` (with a
sibling ``*.error.txt``) on failure, so the queue is idempotent and restartable.
The worker drains any backlog and then exits once ``--stop-file`` exists and the
queue is empty (the collect step touches that file after training finishes).

Run alongside the sweep, e.g.:

    python -u VICReg_review/probe_worker.py \
        --queue-dir VICReg_review/heads/cloud_full_sweep_a100/probe_queue \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
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
from VICReg_review.train_vicreg_review_h5 import (  # noqa: E402
    append_probe_history,
    probe_report_and_row,
)


def process_marker(marker: Path, device: torch.device) -> dict:
    data = json.loads(marker.read_text(encoding="utf-8"))
    ckpt_path = str(data["checkpoint"])
    tsv_path = Path(data["probe_history_tsv"])
    input_h5 = str(data["input_h5"])
    epoch = int(data["epoch"])
    global_step = int(data["global_step"])

    with h5py.File(input_h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])

    model, _cfg, _ep, _gs = load_frozen_encoder(ckpt_path, input_dim, device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = dict(checkpoint.get("args", {}))
    # The marker is authoritative for where output goes and which device we run on.
    saved["probe_history_tsv"] = str(tsv_path)
    saved["device"] = str(device)
    saved_args = SimpleNamespace(**saved)

    result = probe_report_and_row(model, saved_args, device, epoch, global_step)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if result is None:
        raise RuntimeError("probe_report_and_row returned None (probe failed)")
    report, row = result
    append_probe_history(tsv_path, row, report)
    return row


def pending_markers(queue_dir: Path) -> list[Path]:
    # "*.json" excludes already-processed *.json.done/.failed and in-flight *.tmp.
    return sorted(queue_dir.glob("*.json"))


def main_loop(args) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    stop_file = Path(args.stop_file) if args.stop_file else (queue_dir / "STOP")
    poll = max(0.5, float(args.poll_interval))
    processed = 0
    failed = 0
    print(
        f"probe_worker: watching {queue_dir} device={device} "
        f"stop_file={stop_file} poll={poll}s run_once={args.run_once}",
        flush=True,
    )
    while True:
        markers = pending_markers(queue_dir)
        if not markers:
            if stop_file.exists():
                print(
                    f"probe_worker: stop file present and queue drained; "
                    f"processed={processed} failed={failed}",
                    flush=True,
                )
                break
            if args.run_once:
                break
            time.sleep(poll)
            continue
        for marker in markers:
            try:
                process_marker(marker, device)
                marker.rename(marker.parent / (marker.name + ".done"))
                processed += 1
                print(
                    f"probe_worker: done {marker.name} (processed={processed} failed={failed})",
                    flush=True,
                )
            except BaseException as exc:  # noqa: BLE001 - one bad job must not stop the worker
                failed += 1
                tb = traceback.format_exc()
                (marker.parent / (marker.name + ".error.txt")).write_text(
                    f"{type(exc).__name__}: {exc}\n\n{tb}", encoding="utf-8"
                )
                try:
                    marker.rename(marker.parent / (marker.name + ".failed"))
                except OSError:
                    pass
                print(
                    f"probe_worker: FAILED {marker.name}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if args.run_once:
            break


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", required=True, type=Path, help="Probe job queue directory (shared with the trainers' --probe-queue-dir).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between polls when the queue is empty.")
    parser.add_argument("--stop-file", default=None, help="Worker exits once this file exists and the queue is empty. Defaults to <queue-dir>/STOP.")
    parser.add_argument("--run-once", action="store_true", help="Process the current backlog once, then exit (for testing).")
    parser.add_argument("--logout-address", default=None, help="Append stdout/stderr to this log file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_with_optional_tee(args.logout_address, main_loop, args)


if __name__ == "__main__":
    main()
