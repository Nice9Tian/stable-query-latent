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
from VICReg_review import train_vicreg_review_h5 as trainer_mod  # noqa: E402
from VICReg_review.train_vicreg_review_h5 import (  # noqa: E402
    append_probe_history,
    probe_report_and_row,
)


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


def process_marker(marker: Path, device: torch.device, fallback_h5: str | None = None,
                   local_data_dir: str | None = None) -> dict:
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
                process_marker(marker, device, fallback_h5=args.fallback_h5,
                               local_data_dir=args.local_data_dir)
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
