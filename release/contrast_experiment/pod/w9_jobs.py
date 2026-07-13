# -*- coding: utf-8 -*-
"""Shared job table + queue machinery for the w9 campaign notebooks.

Single source of truth for w9_a100.ipynb (BIG jobs) and w9_l40.ipynb (SMALL
jobs). The job LABEL is the compatibility contract shared with the retired
w9_all.ipynb: it names the claim file (<out>/claims/<label>.claim) and the
result file (ft4var_<label>{_fp}_best.json), so machines running any of the
notebooks resume / skip / coordinate each other's work with no migration.

Class split (is_big): anchor_cap >= 2048, or epdb_* full-population arms.
Everything else -- all g512 fixed-split arms and ALL CV jobs -- is small.
A job tuple may carry an optional 7th field "big"/"small" to override.
"""
import json
import os
import queue
import socket
import subprocess
import threading
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent          # dir holding the worker .py's
FS_WORKER = str(PKG / "w9_a100_worker.py")
CV_WORKER = str(PKG / "w9_cv_worker.py")

# ---------------- campaign job table (fixed-split) ----------------
# (arm, anchor_cap, nsp, doc_lead[, wiki_src[, view_w[, size-override]]])
FS_JOBS = [
    ("wcle_cegate3_icetf", 512, False, 0),
    ("wcle_cegate4_icetf", 512, False, 0),
    ("wcle_cegate2_icetf", 2048, False, 0),
    ("wcle_cegate2_icetf", 512, True, 0),
    ("wcle_cegate1_icetf", 512, True, 0),
    ("wcle_i2ce_icetf", 512, True, 0),
    ("wcle_ice_icetf", 512, True, 0),
    ("wcle_ce_cetf", 512, True, 0),
    ("wcle_arc_arctf", 512, True, 0),
    ("wcle_byol_bytf", 512, True, 0),
    ("wcle_rgate2_icetf", 512, False, 0),
    ("wcle_nodoc_i2ce_icetf", 512, False, 0),
    ("wcle_cegate2_icetf", 512, False, 16),
    ("wcle_cegate2_icetf", 512, False, 64),
    ("wcle_cegate2_icetf", 512, False, 0, "llm"),  # pretraining-leak ablation:
    # champion with wiki_llm paraphrase docs instead of raw wiki_clean
    # single-constraint towers under the SAME paraphrase docs (user decree):
    # even leak-free wiki text cannot save CE-only / margin-only / align-only
    # -- the dual-constraint (CE+I) advantage is not a leakage artifact.
    ("wcle_ce_cetf", 512, False, 0, "llm"),
    ("wcle_arc_arctf", 512, False, 0, "llm"),
    ("wcle_byol_bytf", 512, False, 0, "llm"),
]

def _pad(j):
    j = j if len(j) >= 5 else (*j, "clean")
    j = j if len(j) >= 6 else (*j, 16)
    return j if len(j) == 7 else (*j, None)         # optional size override
FS_JOBS = [_pad(j) for j in FS_JOBS]

# ---------------- CV jobs ----------------
CV_RECIPES = ["wcle_cegate2_icetf", "wcle_i2ce_icetf", "wcle_ce_cetf",
              "wcle_rgate2_icetf", "wcle_nodoc_i2ce_icetf", "wcle_ice_icetf"]
N_FOLDS = 5

# ---------------- budgets / protocol constants ----------------
FS_EPOCHS, CV_EPOCHS = 1000, 600
FULL_POOL = True    # views drawn from the ENTIRE review corpus (host RAM);
                    # False = the 2048-sentence pool (local-protocol parity)
CKPT_EVERY, FS_CKPT_SEEDS, CV_CKPT_SEEDS, TOPUP_SEEDS = 50, 3, 2, 10
BEAT_SEC = 30            # worker heartbeat period (written into its claim)
DEAD_SEC = 120           # heartbeat silent for this long => host presumed dead
TAKEOVER_LOCK_STALE = 600  # a .takeover lock older than this = taker died too
VORD = ["neutral", "noname", "positive", "negative"]

# ---------------- labels (THE compatibility contract) ----------------
def fs_label(arm, cap, nsp, lead, wsrc, vw, size=None):
    return (f"w9_{arm}" + (f"_g{cap}" if cap != 512 else "")
            + ("_nsp" if nsp else "") + (f"_ld{lead}" if lead else "")
            + ("_wllm" if wsrc == "llm" else "") + (f"_w{vw}" if vw != 16 else ""))

def cv_label(recipe, fold):
    return f"w9cv_{recipe}_fold{fold}"

def result_name(label):
    return f"ft4var_{label}{'_fp' if FULL_POOL else ''}_best.json"

def is_big(job):
    arm, cap = job[0], job[1]
    if job[6] in ("big", "small"):
        return job[6] == "big"
    return cap >= 2048 or arm.startswith("wcle_epdb")

def summary():
    big = [j for j in FS_JOBS if is_big(j)]
    small = [j for j in FS_JOBS if not is_big(j)]
    print(f"fs jobs: {len(FS_JOBS)} = {len(big)} big + {len(small)} small | "
          f"cv jobs (all small): {len(CV_RECIPES) * N_FOLDS}")

# ---------------- multi-machine claims ----------------
HOST = socket.gethostname() + ":" + os.environ.get("RUNPOD_POD_ID", "?")

def try_claim(claim_dir, nm):
    """Claim = "<host> <heartbeat-ts>", beaten every BEAT_SEC by the OWNING
    worker (--claim-file). A heartbeat silent > DEAD_SEC means the host is
    presumed dead and the job may be taken over. Takeover is made atomic by
    an exclusive-create .takeover lock: two hosts racing for the same corpse
    cannot both succeed (one os.O_EXCL create must fail)."""
    claim_dir.mkdir(parents=True, exist_ok=True)
    cl = claim_dir / f"{nm}.claim"
    try:
        fd = os.open(cl, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{HOST} {time.time():.0f}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        pass
    try:
        owner, ts = cl.read_text().split()
    except Exception:
        return False                    # unreadable mid-rewrite; retry later
    if owner == HOST:
        return True                     # our own earlier claim (crash restart)
    if time.time() - float(ts) <= DEAD_SEC:
        return False                    # owner's heartbeat is alive
    # presumed dead -> atomic takeover via temp lock file
    tk = claim_dir / f"{nm}.claim.takeover"
    try:
        fd = os.open(tk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{HOST} {time.time():.0f}".encode())
        os.close(fd)
    except FileExistsError:
        try:                            # taker itself died mid-takeover?
            if time.time() - tk.stat().st_mtime > TAKEOVER_LOCK_STALE:
                tk.unlink()
        except OSError:
            pass
        return False
    try:
        try:
            owner2, ts2 = cl.read_text().split()
        except Exception:
            return False
        if owner2 != owner or ts2 != ts:
            return False                # owner beat again (or a 3rd host won)
        cl.write_text(f"{HOST} {time.time():.0f}")
        print(f"[claim] took over {nm} (was {owner}, heartbeat silent "
              f"{time.time() - float(ts):.0f}s)", flush=True)
        return True
    finally:
        tk.unlink(missing_ok=True)

def _monitor(log_dirs, stop_evt, period=180):
    seen = {}
    while not stop_evt.wait(period):
        for ld in log_dirs:
            for lg in sorted(Path(ld).glob("*.log")):
                try:
                    sz = lg.stat().st_size
                    if seen.get(str(lg)) == sz:
                        continue
                    seen[str(lg)] = sz
                    with open(lg, "rb") as fh:
                        fh.seek(max(0, sz - 400))
                        tail = fh.read().decode(errors="ignore").strip().splitlines()
                    if tail:
                        print(f"[beat] {lg.name}: {tail[-1]}", flush=True)
                except Exception:
                    pass

def detect_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        ids = [l.strip() for l in out.splitlines() if l.strip()]
        return ids if ids else ["0"]
    except Exception:
        return ["0"]

# ---------------- queue ----------------
def _pending(cls, out_fs, out_cv):
    """Yield ("fs", label, job6) / ("cv", label, recipe, fold) not yet done,
    filtered by class ("big" | "small" | "all"). CV jobs count as small."""
    items = []
    for job in FS_JOBS:
        if cls != "all" and (cls == "big") != is_big(job):
            continue
        nm = fs_label(*job)
        if (Path(out_fs) / result_name(nm)).exists():
            print(f"[skip] fs {nm}")
            continue
        items.append(("fs", nm, job))
    if cls in ("small", "all"):
        for r in CV_RECIPES:
            for k in range(N_FOLDS):
                nm = cv_label(r, k)
                if (Path(out_cv) / result_name(nm)).exists():
                    print(f"[skip] cv {r}/fold{k}")
                    continue
                items.append(("cv", nm, r, k))
    return items

def run_queue(cls, repo, data_dir, out_fs, out_cv, full_pool_path="",
              gpus=None, sweep_other=False):
    """Drain the <cls> job class across this machine's GPUs. Labels/claims/
    results are shared with every other w9 notebook. Returns the fail list."""
    gpus = gpus or detect_gpus()
    items = _pending(cls, out_fs, out_cv)
    if sweep_other and cls in ("big", "small"):
        items += _pending("small" if cls == "big" else "big", out_fs, out_cv)
    jobs = queue.Queue()
    for it in items:
        jobs.put(it)
    log_fs = Path(out_fs) / "logs"; log_fs.mkdir(parents=True, exist_ok=True)
    log_cv = Path(out_cv) / "logs"; log_cv.mkdir(parents=True, exist_ok=True)
    fails = []

    def run_job(gpu, it):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        if it[0] == "fs":
            _, nm, (arm, cap, nsp, lead, wsrc, vw, _sz) = it
            sfx = (("_nsp" if nsp else "") + (f"_ld{lead}" if lead else "")
                   + ("_wllm" if wsrc == "llm" else "")
                   + (f"_w{vw}" if vw != 16 else ""))
            log = log_fs / f"{arm}_g{cap}{sfx}.log"
            cmd = ["python", "-u", FS_WORKER,
                   "--data-dir", data_dir, "--out-dir", out_fs, "--repo", repo,
                   "--arm", arm, "--anchor-cap", str(cap),
                   "--epochs", str(FS_EPOCHS), "--ckpt-every", str(CKPT_EVERY),
                   "--ckpt-seeds", str(FS_CKPT_SEEDS),
                   "--topup-seeds", str(TOPUP_SEEDS)]
            if nsp:
                cmd.append("--no-sp-view")
            if lead:
                cmd += ["--doc-lead", str(lead)]
            if wsrc == "llm":
                cmd += ["--wiki-src", "llm"]
            if vw != 16:
                cmd += ["--view-w", str(vw)]
            tag = f"fs {arm}@g{cap}{sfx}"
        else:
            _, nm, r, k = it
            log = log_cv / f"{r}_fold{k}.log"
            cmd = ["python", "-u", CV_WORKER,
                   "--data-dir", data_dir, "--out-dir", out_cv, "--repo", repo,
                   "--arm", r, "--fold", str(k), "--n-folds", str(N_FOLDS),
                   "--epochs", str(CV_EPOCHS), "--ckpt-every", str(CKPT_EVERY),
                   "--ckpt-seeds", str(CV_CKPT_SEEDS),
                   "--topup-seeds", str(TOPUP_SEEDS)]
            tag = f"cv {r}/fold{k}"
        if FULL_POOL:
            cmd += ["--full-pool", "--full-pool-path", full_pool_path]
        cdir = (Path(out_fs) if it[0] == "fs" else Path(out_cv)) / "claims"
        cmd += ["--claim-file", str(cdir / f"{it[1]}.claim")]
        print(f"[gpu{gpu}] start {tag}", flush=True)
        t0 = time.time()
        with open(log, "w") as fh:
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        if p.returncode != 0:
            fails.append((tag, str(log)))
        print(f"[gpu{gpu}] " + ("ok" if p.returncode == 0 else "FAIL")
              + f" {tag} [{(time.time()-t0)/60:.1f} min]", flush=True)

    def worker(gpu):
        while True:
            try:
                it = jobs.get_nowait()
            except queue.Empty:
                return
            cdir = (Path(out_fs) if it[0] == "fs" else Path(out_cv)) / "claims"
            if not try_claim(cdir, it[1]):
                print(f"[claim] {it[1]} held by another machine -- skipped",
                      flush=True)
                continue
            run_job(gpu, it)

    stop_evt = threading.Event()
    mon = threading.Thread(target=_monitor, args=([log_fs, log_cv], stop_evt),
                           daemon=True)
    mon.start()
    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop_evt.set()
    print(f"QUEUE({cls}) drained in {(time.time()-t0)/3600:.1f} h; "
          f"{len(items)} queued, {len(fails)} failed")
    for tag, log in fails:
        print("  FAILED:", tag, "->", log)
    return fails

# ---------------- aggregate + audit ----------------
def aggregate(out_fs, out_cv):
    import numpy as np
    print("========= fixed-split bests =========")
    for j in sorted(Path(out_fs).glob("ft4var_*_best*.json")):
        d = json.loads(j.read_text())
        runs = d["per_seed"]
        m4 = np.mean([np.mean([r[v]["h1"] for r in runs]) for v in VORD])
        print(f"{j.stem} (ep{d.get('best_ep','?')}): "
              + " ".join(f"{v}:{np.mean([r[v]['h1'] for r in runs]):.3f}"
                         for v in VORD) + f" m4:{m4:.3f}")
    print("\n========= 5-fold CV =========")
    for f in sorted(Path(out_cv).glob("w9cv_frozen_fold*.json")):
        d = json.loads(f.read_text())
        print("  " + f.stem + ": "
              + " ".join(f"{v}:{d[v]['h1']:.3f}" for v in VORD))
    for r in CV_RECIPES:
        per_fold = []
        for k in range(N_FOLDS):
            j = Path(out_cv) / result_name(cv_label(r, k))
            if not j.exists():
                continue
            runs = json.loads(j.read_text())["per_seed"]
            row = {v: np.mean([x[v]["h1"] for x in runs]) for v in VORD}
            row["m4"] = np.mean(list(row.values()))
            per_fold.append(row)
        if not per_fold:
            continue
        print(f"\n{r} ({len(per_fold)} folds)")
        for kf in ("neutral", "noname", "m4"):
            vals = [pf[kf] for pf in per_fold]
            print(f"  {kf:8s} {np.mean(vals):.3f} +- {np.std(vals):.3f}")

def audit(out_fs, out_cv, cls="all"):
    """List unfinished jobs of <cls>. Missing jobs are either being run by
    ANOTHER machine right now, or stale claims of a dead pod (rerun the
    matching notebook later to sweep them up)."""
    missing = []
    for job in FS_JOBS:
        if cls != "all" and (cls == "big") != is_big(job):
            continue
        nm = fs_label(*job)
        if not (Path(out_fs) / result_name(nm)).exists():
            missing.append(nm)
    if cls in ("small", "all"):
        for r in CV_RECIPES:
            for k in range(N_FOLDS):
                nm = cv_label(r, k)
                if not (Path(out_cv) / result_name(nm)).exists():
                    missing.append(nm)
    if missing:
        print(f"AUDIT({cls}): {len(missing)} job(s) not finished on the volume:")
        for m in missing:
            print("   -", m)
    else:
        print(f"AUDIT({cls}): class COMPLETE -- every job has its result json.")
    return missing
