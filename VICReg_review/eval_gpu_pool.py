"""Launch one probe-worker process per GPU for a shared eval queue.

This is intentionally a local process pool, not a scheduler. Distributed eval
comes from all machines pointing their local pool at the same shared
``probe_queue``; ``probe_worker.py`` owns the atomic per-marker claim protocol.
"""

from __future__ import annotations

import argparse
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.logging_tee import run_with_optional_tee  # noqa: E402


def detect_gpus() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        ids = [line.strip() for line in out.splitlines() if line.strip()]
        if ids:
            return ids
    except Exception:
        pass
    return ["0"]


def parse_gpus(text: str) -> list[str]:
    text = str(text or "auto").strip()
    if not text or text.lower() == "auto":
        return detect_gpus()
    return [part.strip() for part in text.split(",") if part.strip()]


def device_for(gpu: str) -> str:
    gpu = str(gpu).strip()
    if gpu.startswith(("cuda", "cpu")):
        return gpu
    return f"cuda:{gpu}"


def queue_counts(queue_dir: Path) -> dict[str, int]:
    return {
        "pending": len(list(queue_dir.glob("*.json"))),
        "claimed": len(list(queue_dir.glob("*.json.claim"))),
        "done": len(list(queue_dir.glob("*.json.done"))),
        "failed": len(list(queue_dir.glob("*.json.failed"))),
    }


def build_worker_cmd(args, gpu: str, index: int) -> list[str]:
    device = device_for(gpu)
    worker_id = f"{args.vm_name}:gpu{gpu}:pool{os.getpid()}:w{index}"
    cmd = [
        str(args.python),
        "-u",
        str(SCRIPT_DIR / "probe_worker.py"),
        "--queue-dir",
        str(args.queue_dir),
        "--device",
        device,
        "--worker-id",
        worker_id,
        "--fallback-h5",
        str(args.fallback_h5),
        "--local-data-dir",
        str(args.local_data_dir),
        "--poll-interval",
        str(args.poll_interval),
        "--claim-timeout",
        str(args.claim_timeout),
        "--claim-refresh",
        str(args.claim_refresh),
        "--history-lock-timeout",
        str(args.history_lock_timeout),
        "--history-lock-stale-seconds",
        str(args.history_lock_stale_seconds),
    ]
    if args.run_once:
        cmd.append("--run-once")
    if args.stop_file:
        cmd.extend(["--stop-file", str(args.stop_file)])
    return cmd


def run_pool(args) -> int:
    args.queue_dir = Path(args.queue_dir)
    if not args.queue_dir.is_absolute():
        args.queue_dir = (ROOT / args.queue_dir).resolve()
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    args.fallback_h5 = Path(args.fallback_h5)
    if not args.fallback_h5.is_absolute():
        args.fallback_h5 = (ROOT / args.fallback_h5).resolve()
    args.local_data_dir = Path(args.local_data_dir)
    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise SystemExit("no GPUs selected")
    if not args.vm_name:
        args.vm_name = socket.gethostname()

    print(
        f"eval_gpu_pool: queue={args.queue_dir} gpus={','.join(gpus)} "
        f"vm={args.vm_name} run_once={args.run_once}",
        flush=True,
    )
    print(f"eval_gpu_pool: initial counts {queue_counts(args.queue_dir)}", flush=True)

    procs: list[tuple[str, subprocess.Popen]] = []
    for idx, gpu in enumerate(gpus):
        cmd = build_worker_cmd(args, gpu, idx)
        print("eval_gpu_pool: start", " ".join(shlex.quote(part) for part in cmd), flush=True)
        procs.append((gpu, subprocess.Popen(cmd, cwd=ROOT)))
        time.sleep(float(args.stagger_seconds))

    failures = 0
    try:
        for gpu, proc in procs:
            rc = proc.wait()
            print(f"eval_gpu_pool: worker gpu={gpu} exit={rc}", flush=True)
            if rc != 0:
                failures += 1
    except KeyboardInterrupt:
        print("eval_gpu_pool: KeyboardInterrupt; terminating workers", flush=True)
        for _gpu, proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for _gpu, proc in procs:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise

    print(f"eval_gpu_pool: final counts {queue_counts(args.queue_dir)}", flush=True)
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", required=True, type=Path)
    parser.add_argument("--gpus", default="auto", help="'auto' or comma-separated GPU ids/devices, e.g. 0,1,2 or cuda:0,cuda:1")
    parser.add_argument("--python", default=sys.executable, type=Path)
    parser.add_argument("--vm-name", default=None, help="Distributed worker identity prefix. Defaults to hostname.")
    parser.add_argument("--run-once", action="store_true", help="Drain the current backlog once, then exit.")
    parser.add_argument("--fallback-h5", default="game_review_data/embedding_h5.h5")
    parser.add_argument("--local-data-dir", default="/root/data")
    parser.add_argument("--stop-file", default=None)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--stagger-seconds", type=float, default=2.0)
    parser.add_argument("--claim-timeout", type=float, default=21600.0)
    parser.add_argument("--claim-refresh", type=float, default=60.0)
    parser.add_argument("--history-lock-timeout", type=float, default=3600.0)
    parser.add_argument("--history-lock-stale-seconds", type=float, default=21600.0)
    parser.add_argument("--logout-address", default=None, help="Append pool and child stdout/stderr to this log file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    code = run_with_optional_tee(args.logout_address, run_pool, args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
