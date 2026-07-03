"""Small file-based gates for coordinated eval notebooks.

Every generated ``Pod_N`` bundle can run ``eval.ipynb`` end-to-end. The probe
queue is drained by all VMs, while final eval/archive should run once. This
module gives the notebook a shared "final owner" claim plus a wait-for-drain
helper without depending on Jupyter state.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path


def atomic_create_json(path: Path, payload: dict) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except (FileExistsError, OSError):
        return False
    try:
        os.write(fd, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    finally:
        os.close(fd)
    return True


def read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def queue_counts(queue_dir: Path) -> dict[str, int]:
    return {
        "pending": len(list(queue_dir.glob("*.json"))),
        "claimed": len(list(queue_dir.glob("*.json.claim"))),
        "done": len(list(queue_dir.glob("*.json.done"))),
        "failed": len(list(queue_dir.glob("*.json.failed"))),
    }


def wait_queue_drained(queue_dir: Path, poll: float, timeout: float) -> dict[str, int]:
    deadline = None if timeout <= 0 else time.time() + timeout
    last_print = 0.0
    while True:
        counts = queue_counts(queue_dir)
        if counts["pending"] == 0 and counts["claimed"] == 0:
            return counts
        now = time.time()
        if now - last_print >= max(5.0, poll):
            print(f"eval_stage_gate: waiting for probe queue drain {counts}", flush=True)
            last_print = now
        if deadline is not None and now >= deadline:
            raise TimeoutError(f"probe queue did not drain before timeout: {counts}")
        time.sleep(max(1.0, poll))


def _owner_pid_alive_here(owner: dict) -> bool | None:
    """True/False when the claim was made on THIS host (pid check is meaningful);
    None when it belongs to another machine (unknowable from here)."""
    if str(owner.get("host", "")) != socket.gethostname():
        return None
    try:
        os.kill(int(owner.get("pid")), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def claim_final(out_dir: Path, vm_name: str, wait: bool, poll: float, timeout: float,
                stale_seconds: float = 0.0) -> bool:
    gate_dir = out_dir / "_eval_coord"
    claim = gate_dir / "final_owner.json"
    done = gate_dir / "final_done.json"
    payload = {
        "vm": vm_name,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "claimed_ts": time.time(),
    }
    while True:
        if atomic_create_json(claim, payload):
            print(f"eval_stage_gate: final owner = {vm_name}", flush=True)
            return True
        if done.exists():
            break                     # a previous run finished -> plain skip path
        owner = read_json(claim) or {}
        # DEAD-OWNER RECOVERY: without this, any eval interrupted after claiming
        # (kernel restart, crash) deadlocks every later run behind a claim whose
        # process no longer exists. Same-host dead pid -> reclaim immediately;
        # other hosts -> reclaim only past --stale-seconds (0 = never).
        alive_here = _owner_pid_alive_here(owner)
        age = time.time() - float(owner.get("claimed_ts", 0) or 0)
        if alive_here is False or (stale_seconds > 0 and age > stale_seconds):
            aside = claim.parent / f".final_owner.stale.{int(time.time() * 1000)}.json"
            try:
                os.rename(str(claim), str(aside))
            except OSError:
                break                 # someone else reclaimed first -> fall through
            print(f"eval_stage_gate: reclaimed stale final claim from "
                  f"{owner.get('vm', '?')} (pid_alive_here={alive_here}, age={age:.0f}s)",
                  flush=True)
            continue
        break
    owner = read_json(claim) or {}
    print(f"eval_stage_gate: final already owned by {owner.get('vm', '?')}", flush=True)
    if wait:
        deadline = None if timeout <= 0 else time.time() + timeout
        while not done.exists():
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(f"timed out waiting for final_done.json; owner={owner}")
            time.sleep(max(5.0, poll))
        print("eval_stage_gate: final_done.json present; skipping final/archive here", flush=True)
    return False


def mark_final_done(out_dir: Path, vm_name: str) -> None:
    done = out_dir / "_eval_coord" / "final_done.json"
    payload = {
        "vm": vm_name,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "done_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "done_ts": time.time(),
    }
    if atomic_create_json(done, payload):
        print("eval_stage_gate: final marked done", flush=True)
    else:
        print("eval_stage_gate: final_done.json already exists", flush=True)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    wait = sub.add_parser("wait-queue")
    wait.add_argument("--queue-dir", required=True, type=Path)
    wait.add_argument("--poll", type=float, default=30.0)
    wait.add_argument("--timeout", type=float, default=0.0, help="Seconds; 0 waits forever.")

    claim = sub.add_parser("claim-final")
    claim.add_argument("--out-dir", required=True, type=Path)
    claim.add_argument("--vm-name", required=True)
    claim.add_argument("--wait", action="store_true")
    claim.add_argument("--poll", type=float, default=30.0)
    claim.add_argument("--timeout", type=float, default=0.0, help="Seconds; 0 waits forever.")
    claim.add_argument("--stale-seconds", type=float, default=0.0,
                       help="Reclaim another HOST's final claim older than this with no "
                            "final_done.json (same-host dead pids are always reclaimed). "
                            "0 = never.")

    done = sub.add_parser("mark-final-done")
    done.add_argument("--out-dir", required=True, type=Path)
    done.add_argument("--vm-name", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cmd == "wait-queue":
        counts = wait_queue_drained(args.queue_dir, args.poll, args.timeout)
        print(f"eval_stage_gate: queue drained {counts}", flush=True)
        return
    if args.cmd == "claim-final":
        raise SystemExit(0 if claim_final(args.out_dir, args.vm_name, args.wait, args.poll,
                                          args.timeout, stale_seconds=args.stale_seconds) else 2)
    if args.cmd == "mark-final-done":
        mark_final_done(args.out_dir, args.vm_name)
        return
    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
