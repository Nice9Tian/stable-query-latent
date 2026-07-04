"""Battery worker: parallel per-combo downstream eval with streaming grid metrics.

One worker owns ONE GPU (the spawner pins CUDA_VISIBLE_DEVICES); several workers
-- across GPUs and across VMs -- share the sweep out_dir and coordinate through
per-combo claim files, so the full-n battery parallelises the same way training
does. After EVERY finished combo the worker republishes
``<out_dir>/grid_metrics.json`` (a complete rebuild from the per-combo
eval_report.json files, atomic unique-tmp publish), so the metrics file streams
forward while the battery runs and is race-safe under concurrent writers.

Scope mirrors eval_champions stage 1: DONE combos at the FULL train count of
the sweep.yaml FULL grid (grid.exclude cleared -> masked done cells included),
evaluated on their BEST checkpoint.

Run one warmup first (builds the shared raw/description/text caches once):

    python -u VICReg_review/eval_battery_worker.py --warmup-only

then one worker per GPU:

    CUDA_VISIBLE_DEVICES=0 python -u VICReg_review/eval_battery_worker.py &
    CUDA_VISIBLE_DEVICES=1 python -u VICReg_review/eval_battery_worker.py &
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.logging_tee import run_with_optional_tee  # noqa: E402

CLAIM_NAME = "eval_claim.json"
BEST_CKPT = "vicreg_review_h5_best.pt"
LATEST_CKPT = "vicreg_review_h5_latest.pt"
MANIFEST = "vicreg_review_h5_manifest.json"


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def atomic_create_json(path: Path, payload: dict) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except OSError:
        return False
    try:
        os.write(fd, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    finally:
        os.close(fd)
    return True


def publish_json(path: Path, payload: dict) -> None:
    """Atomic publish with a per-writer tmp (a shared fixed .tmp lets two
    concurrent writers interleave write/rename and publish a torn file)."""
    tmp = path.with_name(f"{path.name}.tmp.{socket.gethostname()}.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------ grid metrics
def _mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def extract_metrics_row(combo, rep: dict, masked: bool, report_path: Path) -> dict:
    tag = rep.get("tag_probe") or {}
    ident = rep.get("identity") or {}
    senti = rep.get("sentiment_probe") or {}
    tv = rep.get("text_variant_eval") or {}
    rt = tv.get("real_text_tag") or {}
    drops = [(rt.get(v) or {}).get("drop_micro_f1") for v in rt]
    tag_f1 = tag.get("micro_f1")
    r2 = senti.get("r2")
    sel = None
    if tag_f1 is not None and r2 is not None and not math.isnan(float(r2)):
        sel = float(tag_f1) * (1.0 - min(1.0, max(0.0, float(r2))))
    reco = rep.get("recommendation_probe")
    return {
        "combo_id": combo.combo_id,
        "arm": combo.arm, "output_dim": combo.output_dim, "num_latents": combo.num_latents,
        "view": combo.view, "train_games": combo.train_games,
        "masked": masked,
        "tag_f1": tag_f1,
        "tag_fold_f1_mean": tag.get("fold_micro_f1_mean"),
        "identity_hit_at_1": ident.get("hit_at_1"),
        "identity_mean_rank": ident.get("mean_rank"),
        "identity_pr": ident.get("participation_ratio"),
        "identity_mean_vicreg_cosine": ident.get("mean_vicreg_cosine"),
        "sentiment_r2": r2,
        "variant_drop_mean": _mean(drops),
        "variant_anchor_test_f1": ((tv.get("tag_generalization") or {}).get("anchor_test") or {}).get("micro_f1"),
        "recommendation_pearson": (reco.get("pearson") if isinstance(reco, dict) else None),
        "selectivity": sel,
        "report_path": str(report_path),
    }


def full_pool(sweep_yaml: Path):
    """(full-n combos of the FULL grid, live combo-id set, full_n)."""
    from VICReg_review.sweep.config import SweepConfig

    cfg = SweepConfig.load(str(sweep_yaml))
    full = copy.deepcopy(cfg)
    full.grid.exclude = []
    counts = [int(c.train_games) for c in full.iter_combos()]
    full_n = 0 if any(n <= 0 for n in counts) else (max(counts) if counts else 0)
    pool = [c for c in full.iter_combos() if int(c.train_games) == full_n]
    live_ids = {c.combo_id for c in cfg.iter_combos()}
    return pool, live_ids, full_n


def rebuild_grid_metrics(out_root: Path, sweep_yaml: Path, git_commit: str = "") -> dict:
    """Full rebuild from the per-combo reports -- a pure derived view, so the
    streaming update after each combo is just 'rebuild and publish'."""
    pool, live_ids, full_n = full_pool(sweep_yaml)
    rows, missing = [], []
    for c in pool:
        rp = out_root / c.combo_id / "eval_report.json"
        rep = read_json(rp)
        if not isinstance(rep, dict):
            missing.append(c.combo_id)
            continue
        rows.append(extract_metrics_row(c, rep, c.combo_id not in live_ids, rp))
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit,
        "full_n": full_n,
        "pool_size": len(rows),
        "missing_reports": missing,
        "rows": rows,
    }
    publish_json(out_root / "grid_metrics.json", payload)
    return payload


# ------------------------------------------------------------------ claims
def _pid_alive_here(pid) -> bool:
    """POSIX existence probe. PermissionError means the pid exists (not ours).
    Only meaningful on the host that wrote the claim."""
    if os.name != "posix":
        return True                 # never probe on Windows (os.kill(pid,0) kills there)
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except (OSError, TypeError, ValueError):
        return False


def try_claim(cdir: Path, ttl: float) -> bool:
    claim = cdir / CLAIM_NAME
    payload = {"host": socket.gethostname(), "pid": os.getpid(), "ts": time.time()}
    if atomic_create_json(claim, payload):
        return True
    owner = read_json(claim) or {}
    age = time.time() - float(owner.get("ts", 0) or 0)
    # Same-host dead owner -> reclaim IMMEDIATELY (a pod reboot / SIGKILL never
    # runs release_claim; without this the restarted machine would stare at its
    # own predecessor's claim for the full TTL). Other hosts' deaths can't be
    # verified from here, so they still wait out the TTL.
    same_host_dead = (str(owner.get("host", "")) == payload["host"]
                      and not _pid_alive_here(owner.get("pid")))
    if not same_host_dead and age <= ttl:
        return False
    aside = claim.parent / f".{CLAIM_NAME}.stale.{int(time.time() * 1000)}"
    try:
        os.rename(str(claim), str(aside))
    except OSError:
        return False               # someone else reclaimed first
    try:
        aside.unlink()
    except OSError:
        pass
    return atomic_create_json(claim, payload)


def release_claim(cdir: Path) -> None:
    try:
        (cdir / CLAIM_NAME).unlink()
    except OSError:
        pass


# ------------------------------------------------------------------ worker
def build_eval_args(args):
    """The exact args namespace sweep_cloud's eval path uses, so per-combo
    reports are byte-compatible with the single-process route."""
    from VICReg_review import sweep_cloud

    argv = [
        "--h5", str(args.h5),
        "--out-dir", str(args.out_dir),
        "--skip-train", "--eval-mode", "none", "--calib-mode", "off",
    ]
    eval_args = sweep_cloud.parse_args(argv)
    eval_args.out_dir = Path(eval_args.out_dir)
    eval_args.h5 = Path(eval_args.h5)
    if not getattr(eval_args, "device", None):
        eval_args.device = "cuda"
    if getattr(eval_args, "tag_text_split_json", None) is None:
        eval_args.tag_text_split_json = str(Path(eval_args.out_dir) / "tag_text_eval_split.json")
    return eval_args


def combo_ready(cdir: Path):
    """(done, checkpoint) -- done requires manifest status=done or done.json;
    prefer the best checkpoint like final_best/per_combo do."""
    manifest = read_json(cdir / MANIFEST)
    done = (isinstance(manifest, dict) and manifest.get("status") == "done") \
        or (cdir / "done.json").exists()
    if not done:
        return False, None
    ckpt = cdir / BEST_CKPT
    if not ckpt.exists():
        ckpt = cdir / LATEST_CKPT
    return ckpt.exists(), (ckpt if ckpt.exists() else None)


def enable_resident_vectors(h5_path, local_data_dir):
    """Point the trainer module's _VECTORS_DAT global at the machine-local
    resident .dat when it is staged here. extract_features -> load_game_views
    consults that global (exactly how training and the probe drain gather in
    resident mode); without it the battery reads EVERY game's vector block
    serially from the shared H5 over the network FS and the GPU sleeps.
    Returns the .dat path, or None when not staged on this machine."""
    from VICReg_review import train_vicreg_review_h5 as trainer_mod

    dat_path = trainer_mod.default_vectors_dat_path(h5_path, local_data_dir)
    desc = trainer_mod.resident_descriptor(dat_path, h5_path)
    if desc is None:
        return None
    trainer_mod._VECTORS_DAT = desc
    return str(desc[0])


def run_worker(args) -> None:
    from VICReg_review import run_data_view_sweep as sweep

    out_root = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    sweep_yaml = Path(args.sweep_yaml) if Path(args.sweep_yaml).is_absolute() else ROOT / args.sweep_yaml
    eval_args = build_eval_args(args)
    eval_args.out_dir = out_root
    dat = enable_resident_vectors(eval_args.h5, args.local_data_dir)
    if dat:
        print(f"battery: resident .dat ON {dat} (bulk vectors from local NVMe)", flush=True)
    else:
        print(f"battery: resident .dat NOT staged under {args.local_data_dir}; bulk "
              f"vectors stream from {eval_args.h5} over the shared FS -- SLOW. "
              f"Stage it (prepare_training / trainer startup does) and restart.", flush=True)
    git_commit = ""
    try:
        import subprocess
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip()
    except Exception:
        pass

    pool, _live, full_n = full_pool(sweep_yaml)
    tag = f"[{socket.gethostname()}:{os.getpid()} gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '?')}]"
    print(f"{tag} battery worker: {len(pool)} full-n (n={full_n}) combos in scope", flush=True)

    # Shared caches, loaded ONCE per worker (evaluate_combo reloads them per call).
    names, appids, titles = sweep.h5_game_metadata(eval_args.h5)
    raw_cache = sweep.load_or_build_raw_game_vectors(eval_args, appids, names, titles)
    description_cache = sweep.load_or_build_description_raw_cache(eval_args, appids, names, titles)
    text_cache = sweep.embed_test_cases(eval_args)
    if args.warmup_only:
        print(f"{tag} warmup done: shared caches built/loaded; exiting", flush=True)
        return

    done_n = skip_n = fail_n = 0
    for combo in pool:
        cdir = out_root / combo.combo_id
        ready, ckpt = combo_ready(cdir)
        if not ready:
            continue
        report_path = cdir / "eval_report.json"
        arm = sweep.combo_arm_from_dir(cdir)
        if report_path.exists() and sweep.report_is_current(report_path, ckpt, cdir / MANIFEST, arm):
            skip_n += 1
            continue
        if not try_claim(cdir, args.claim_ttl):
            continue                      # someone else is on it (or fresh claim)
        try:
            print(f"{tag} evaluating {combo.combo_id}", flush=True)
            feats, feature_names = sweep.build_vicreg_feature_cache(eval_args, ckpt, cdir)
            sweep.evaluate_combo_from_features(
                eval_args, ckpt, cdir, feats, feature_names,
                raw_cache, description_cache, text_cache)
            del feats, feature_names
            done_n += 1
            rebuild_grid_metrics(out_root, sweep_yaml, git_commit)   # STREAM after every combo
            print(f"{tag} done {combo.combo_id} (done={done_n} skip={skip_n} fail={fail_n})", flush=True)
        except BaseException as exc:  # noqa: BLE001 - one bad combo must not stop the worker
            fail_n += 1
            print(f"{tag} FAILED {combo.combo_id}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            release_claim(cdir)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    payload = rebuild_grid_metrics(out_root, sweep_yaml, git_commit)  # final publish
    print(f"{tag} battery worker exit: done={done_n} skip={skip_n} fail={fail_n}; "
          f"grid_metrics rows={payload['pool_size']} missing={len(payload['missing_reports'])}",
          flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", default=str(ROOT / "game_review_data/embedding_h5.h5"))
    p.add_argument("--out-dir", default="VICReg_review/heads/cloud_full_sweep_a100")
    p.add_argument("--sweep-yaml", default="VICReg_review/sweep/sweep.yaml")
    p.add_argument("--claim-ttl", type=float, default=7200.0,
                   help="Seconds before another worker may reclaim a stale eval claim.")
    p.add_argument("--local-data-dir", default="/root/data",
                   help="Machine-local dir holding the resident vectors .dat staged for "
                        "--h5. When present, feature extraction gathers vectors from "
                        "local NVMe instead of streaming the shared H5 per game.")
    p.add_argument("--warmup-only", action="store_true",
                   help="Build/load the shared raw/description/text caches, then exit. "
                        "Run once before fanning out workers so they don't all build "
                        "the same caches concurrently.")
    p.add_argument("--logout-address", default=None)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_with_optional_tee(args.logout_address, run_worker, args)


if __name__ == "__main__":
    main()
