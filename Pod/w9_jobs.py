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
# (arm, anchor_cap, nsp, doc_lead[, wiki_src[, view_w[, size-override
#  [, epochs-override]]]]) -- epochs None = FS_EPOCHS
FS_JOBS = [
    # ---- FRONT OF THE BIG QUEUE (user priority: the CE scaling question) ----
    # pure-CE anchor curve 512(done)/2048/4096: does CE decline past 512 while
    # i2ce keeps scaling? Same-cap I-premium reference at the champion caps.
    ("wcle_ce_cetf", 2048, False, 0),
    ("wcle_ce_cetf", 4096, False, 0),
    # SCALE showdown (user, w9_scale.ipynb): ai2ce joins the cap ladder
    ("wcle_ai2ce_icetf", 2048, False, 0),
    ("wcle_ai2ce_icetf", 4096, False, 0),
    # USER TOP PRIORITY small row: CE decomposed (anchor-in-I + LSE(k-))
    ("wcle_ai2auni25_icetf", 512, False, 0),
    # W&I decomposition controls: 3*align + uniform(t=2, batch repulsion)
    ("wcle_ai2ce_icetf", 512, False, 0),    # both pulls: CE + anchor rope
    ("wcle_ai6auni2_icetf", 512, False, 0),  # W&I both-at-anchor (i6 = paper 3:1)
    ("wcle_ai4auni2_icetf", 512, False, 0),  # i4 rung
    ("wcle_i6uni2_icetf", 512, False, 0),    # pure W&I anchor-free (i6)
    ("wcle_i4uni2_icetf", 512, False, 0),    # i4 rung
    # bce family: CE push from BATCH views (bare ce stays anchored == ace)
    ("wcle_bce_cetf", 512, False, 0),        # pure SimCLR baseline
    ("wcle_i2bce_icetf", 512, False, 0),     # + I x2
    ("wcle_ai2bce_icetf", 512, False, 0),    # + anchor rope (I only)
    # SCALE @1024: brand-new cap point for the showdown trio
    ("wcle_ce_cetf", 1024, False, 0),
    ("wcle_i2ce_icetf", 1024, False, 0),
    ("wcle_ai2ce_icetf", 1024, False, 0),
    ("wcle_ai6uni2_icetf", 512, False, 0),   # mixed source (parked, not in flash)
    # CE-attachment matrix at the champion cap (user decree): where CE
    # attaches x anchor cap; A/B = i2ce@g2048 (done) / i2ce@512 (queued)
    ("wcle_i2expce_icetf", 2048, False, 0),
    ("wcle_i2expce_icetf", 512, False, 0),   # CE-attachment matrix @512
    ("wcle_i2poolce_icetf", 512, False, 0),
    ("wcle_expce_cetf", 2048, False, 0),   # pure expander-CE (no I)
    ("wcle_expce_cetf", 512, False, 0),
    # ---- @512 STRUCTURE BLITZ (user top priority): full 2x4 CE-attachment
    # matrix {no-I, I2} x {per-view, expander, pooled, pool->expander}.
    # ce@512 done (0.835); the other 7 cells below/above. Results decide
    # which structure the scaling theory is built on.
    ("wcle_ceexpi2_icetf", 512, False, 0),      # per-view CE@dep + I@E
    ("wcle_expi2expce_icetf", 512, False, 0),   # I@E + CE@E (new design)
    ("wcle_poolceexpi2_icetf", 512, False, 0),  # pooled CE@dep + I@E
    ("wcle_expi2poolexpce_icetf", 512, False, 0),  # I@E_I + pool->E_CE->CE (dual)
    ("wcle_shexpi2ce_icetf", 512, False, 0),    # SHARED E: I@E + CE@E
    ("wcle_shexpi2poolce_icetf", 512, False, 0),  # SHARED E: I@E + pool->E->CE
    ("wcle_cmpce_cetf", 512, False, 0),     # DOWN-projector (SimCLR dir) trio
    # direction-closure wave (user decree, terminal grid)
    ("wcle_i2poolexpce_icetf", 512, False, 0),
    ("wcle_i2poolcmpce_icetf", 512, False, 0),
    ("wcle_expi2cmpce_icetf", 512, False, 0),
    ("wcle_cmpi2expce_icetf", 512, False, 0),
    ("wcle_cmpi2cmpce_icetf", 512, False, 0),
    ("wcle_expi2poolcmpce_icetf", 512, False, 0),
    ("wcle_shexpi2poolexpce_icetf", 512, False, 0),
    ("wcle_shcmpi2poolcmpce_icetf", 512, False, 0),
    # symmetry completion (30-cell final grid)
    ("wcle_poolcmpce_cetf", 512, False, 0),
    ("wcle_cecmpi2_icetf", 512, False, 0),
    ("wcle_poolcecmpi2_icetf", 512, False, 0),
    ("wcle_shcmpi2poolce_icetf", 512, False, 0),
    ("wcle_cmpi2poolexpce_icetf", 512, False, 0),
    ("wcle_cmpi2poolcmpce_icetf", 512, False, 0),
    ("wcle_i2cmpce_icetf", 512, False, 0),
    ("wcle_shcmpi2ce_icetf", 512, False, 0),
    ("wcle_poolce_cetf", 512, False, 0),
    ("wcle_poolexpce_cetf", 512, False, 0),
    ("wcle_i2ce_icetf", 512, False, 0),  # clean@512 i2ce (only nsp existed):
    # fixes the i2ce anchor-curve ORIGIN (512 point was nsp-handicapped)
    # and completes the clean same-cap ce-vs-i2ce pair at 512
    ("wcle_cegate3_icetf", 512, False, 0),
    ("wcle_cegate4_icetf", 512, False, 0),
    ("wcle_cegate2_icetf", 2048, False, 0),
    ("wcle_i2ce_icetf", 2048, False, 0),
    ("wcle_i2ce_icetf", 4096, False, 0),
    ("wcle_cegate2_icetf", 512, True, 0),
    ("wcle_cegate1_icetf", 512, True, 0),
    ("wcle_i2ce_icetf", 512, True, 0),
    ("wcle_ice_icetf", 512, True, 0),
    ("wcle_ce_cetf", 512, True, 0),
    ("wcle_arc_arctf", 512, True, 0),
    ("wcle_byol_bytf", 512, True, 0),
    ("wcle_rgate2_icetf", 512, False, 0),
    ("wcle_nodoc_i2ce_icetf", 512, False, 0),
    ("wcle_vic_cetf", 512, False, 0),   # VICReg tower I=10 V=20 C=20:
    # negative-free like BYOL but with explicit V/C anti-collapse
    ("wcle_vic2_cetf", 512, False, 0),  # C-dose ablation (C 20 -> 15)
    ("wcle_epd_v25i25c1_cetf", 512, False, 0),  # CANONICAL VICReg (all 3 terms
    # on expander output, paper weights 25/25/1) -- the FAIR baseline; vic/vic2
    # died via the centroid-collapse loophole and are not citable as VICReg
    ("wcle_epd_v20i10c20_cetf", 512, False, 0),  # canonical, OUR weights
    ("wcle_epd_v20i10c15_cetf", 512, False, 0),  # canonical, OUR weights C15
    # TRUE batch=all: epd wiring, views for the entire train pool per step
    ("wcle_epdb_v25i25c1_cetf", 512, False, 0),
    ("wcle_epdb_v20i10c20_cetf", 512, False, 0),
    ("wcle_epdb_v20i10c15_cetf", 512, False, 0),
    ("wcle_byol2_bytf", 512, False, 0),  # BYOL + BN projector/3-layer BN predictor
    ("wcle_byol_bytf", 512, False, 0),   # plain BYOL clean fp W16: A/B partner
    # for byol2 (BN effect) AND for the local pool-2048 plain byol (fp effect)
    ("wcle_cegate2c_icetf", 2048, False, 0),  # champion + TRAIN-TIME centering
    # (mu-EMA subtract before L2-norm); A/B partner = cegate2_icetf@g2048
    ("wcle_i2cce_icetf", 2048, False, 0),   # I2CCE: CE + I x2 + C x1; A/B = i2ce@g2048
    ("wcle_i2ccec_icetf", 2048, False, 0),  # I2CCE + train-time centering
    ("wcle_i2ccec_icetf", 4096, False, 0),  # coronation test: does C+centering
    # stack with scale? (deployment story stays g2048 either way)
    # ---- view-W sweep (sentence budget per review view; 16 = historical) ----
    ("wcle_i2ce_icetf", 2048, False, 0, "clean", 48),
    ("wcle_i2ce_icetf", 2048, False, 0, "clean", 64),
    ("wcle_i2cce_icetf", 2048, False, 0, "clean", 48),
    ("wcle_i2cce_icetf", 2048, False, 0, "clean", 64),
    ("wcle_i2ccec_icetf", 2048, False, 0, "clean", 48),
    ("wcle_i2ccec_icetf", 2048, False, 0, "clean", 64),
    ("wcle_ce_cetf", 512, False, 0),               # pure-CE W16 baseline
    ("wcle_ce_cetf", 512, False, 0, "clean", 48),
    ("wcle_ce_cetf", 512, False, 0, "clean", 64),
    ("wcle_vic_cetf", 512, False, 0, "clean", 48),
    ("wcle_vic_cetf", 512, False, 0, "clean", 64),
    ("wcle_vic2_cetf", 512, False, 0, "clean", 48),
    ("wcle_vic2_cetf", 512, False, 0, "clean", 64),
    ("wcle_byol2_bytf", 512, False, 0, "clean", 48),
    ("wcle_byol2_bytf", 512, False, 0, "clean", 64),
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
    ("wcle_i2ccec_icetf", 2048, False, 0, "llm"),  # NEW-champion leak ablation:
    # the wllm firewall row must sit on the configuration the paper leads with
    # (the existing wllm rows cover the OLD champion cegate2@512 only)
    # ---- view-COMPOSITION grid (user; explicit grammar, see repo-root
    # model_history.md): [d<k>][w<k>][sp<k>]r<n>_i2ce -- d = tiered doc
    # slot (wiki -> sp -> review fallback, the protocol slot), w = wiki-
    # only slot, sp = store-page-only slot (wiki games NOT excluded), r =
    # review views; protocol i2ce == d1r3. d1r4/5/6 = v_review dose ladder
    # (baselines: i2ce@512 = d1r3, nodoc = 4R+0D); w1sp1r3 = wiki AND sp
    # views COEXIST in one step (tiering never lets a game see both);
    # missing docs fall back to review views. All @512 clean; per-view-sum
    # convention: CE terms and I edges grow with NV.
    # 2000ep (single-budget curve, matches the scale grid's @512 cells;
    # per-job override -- other pending rows stay at FS_EPOCHS).
    ("wcle_d1r4_i2ce_icetf", 512, False, 0, "clean", 16, None, 2000),
    ("wcle_d1r5_i2ce_icetf", 512, False, 0, "clean", 16, None, 2000),
    ("wcle_d1r6_i2ce_icetf", 512, False, 0, "clean", 16, None, 2000),
    ("wcle_w1sp1r3_i2ce_icetf", 512, False, 0, "clean", 16, None, 2000),
    # ---- anchor-supply scaling ladder (bank + MoCo queue), control = the
    # existing i2cce@g2048 row. Nominally g2048 but size-overridden "small":
    # they eliminate the full-gallery-with-grad pass that makes g2048 "big"
    # (per-step anchors 192*cap vs 1613*cap), static VRAM 18.6 GB << L40 48 GB.
    # mq FIRST (user priority).
    # mq3072i2cce cap curve (2048 done / 4096 / 8192) is OWNED by the
    # dedicated Pod/w9_mq_i2ce.ipynb (user decree); labels stay compatible.
    ("wcle_bkq192i2cce_icetf", 2048, False, 0, "clean", 16, "small"),
    ("wcle_bkq48i2cce_icetf", 2048, False, 0, "clean", 16, "small"),
    ("wcle_bkq12i2cce_icetf", 2048, False, 0, "clean", 16, "small"),
    ("wcle_bkbi2cce_icetf", 2048, False, 0, "clean", 16, "small"),
]

def _pad(j):
    j = j if len(j) >= 5 else (*j, "clean")
    j = j if len(j) >= 6 else (*j, 16)
    j = j if len(j) >= 7 else (*j, None)            # optional size override
    return j if len(j) == 8 else (*j, None)         # optional epochs override
FS_JOBS = [_pad(j) for j in FS_JOBS]

# ---------------- CV jobs ----------------
CV_RECIPES = ["wcle_ce_cetf", "wcle_i2ce_icetf"]   # core: I-CE vs CE (user)
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
def fs_label(arm, cap, nsp, lead, wsrc, vw, size=None, epochs=None):
    # NOTE: epochs is NOT in the label -- a label owns ONE budget; changing
    # a row's budget after results exist means extend/retrain, not a fork.
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
            _, nm, (arm, cap, nsp, lead, wsrc, vw, _sz, _ep) = it
            sfx = (("_nsp" if nsp else "") + (f"_ld{lead}" if lead else "")
                   + ("_wllm" if wsrc == "llm" else "")
                   + (f"_w{vw}" if vw != 16 else ""))
            log = log_fs / f"{arm}_g{cap}{sfx}.log"
            cmd = ["python", "-u", FS_WORKER,
                   "--data-dir", data_dir, "--out-dir", out_fs, "--repo", repo,
                   "--arm", arm, "--anchor-cap", str(cap),
                   "--epochs", str(_ep or FS_EPOCHS), "--ckpt-every", str(CKPT_EVERY),
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

# ---------------- extension (续训) ----------------
def extend_fs(job6, epochs, repo, data_dir, out_fs, full_pool_path="", gpu="0"):
    """EXTEND a finished fixed-split arm past its original budget. The worker
    rebuilds training state from its newest checkpoint (weights + mq shadow;
    fresh opt/rng; mq queue re-prefills from the shadow). Stale best-jsons
    are removed so the head phase re-picks over ALL checkpoints (old + new
    per-epoch probe jsons are reused, only new ones get computed)."""
    arm, cap, nsp, lead, wsrc, vw = job6
    nm = fs_label(arm, cap, nsp, lead, wsrc, vw)
    cdir = Path(out_fs) / "claims"
    if not try_claim(cdir, nm):
        print(f"[extend] {nm} held by another machine -- skipped")
        return None
    for f in Path(out_fs).glob(f"ft4var_{nm}{'_fp' if FULL_POOL else ''}_best*.json"):
        print("[extend] removing stale", f.name)
        f.unlink()
    log = Path(out_fs) / "logs" / f"{arm}_g{cap}_ext{epochs}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["python", "-u", FS_WORKER,
           "--data-dir", data_dir, "--out-dir", out_fs, "--repo", repo,
           "--arm", arm, "--anchor-cap", str(cap),
           "--epochs", str(epochs), "--ckpt-every", str(CKPT_EVERY),
           "--ckpt-seeds", str(FS_CKPT_SEEDS), "--topup-seeds", str(TOPUP_SEEDS)]
    if nsp:
        cmd.append("--no-sp-view")
    if lead:
        cmd += ["--doc-lead", str(lead)]
    if wsrc == "llm":
        cmd += ["--wiki-src", "llm"]
    if vw != 16:
        cmd += ["--view-w", str(vw)]
    if FULL_POOL:
        cmd += ["--full-pool", "--full-pool-path", full_pool_path]
    cmd += ["--claim-file", str(cdir / f"{nm}.claim")]
    print(f"[extend] {nm} -> {epochs} epochs", flush=True)
    t0 = time.time()
    with open(log, "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
    print(("[extend] ok" if p.returncode == 0 else "[extend] FAIL")
          + f" {nm} [{(time.time()-t0)/60:.1f} min] -> {log}", flush=True)
    return p.returncode

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
