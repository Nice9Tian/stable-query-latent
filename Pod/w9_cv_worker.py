# -*- coding: utf-8 -*-
"""R60 wcle-protocol 5-FOLD CV worker: one (ARM, FOLD) job = tower (600ep,
ckpt/50) + per-checkpoint heads (2 seeds) + post-hoc vsel pick topped to 10.

FOLD PROTOCOL: the 814-game clean wiki universe is permuted once (--cv-seed)
and split into 5 folds. Fold k = TEST (~163), fold (k+1)%5 = VAL (~163),
remaining 3 folds = TRAIN docs (~488). Every exclusion (doc bans, review
train_pool, pseudo-queries, head phase-2/selection) is recomputed per fold
under the same fully-inductive rules as the fixed split. A frozen-embedder
baseline for the fold is computed once into w9cv_frozen_fold<k>.json.

A100 adaptations vs the local (RTX3080) cells:
  * EVERYTHING lives in VRAM: the 2048-sentence review pool (8.5 GB), the
    flat pseudo-queries (8.5 GB), anchors, doc views, eval queries. View
    sampling gathers on-GPU in one indexing kernel (no host copies).
  * --anchor-cap 2048 rebuilds the FULL-budget anchor (sp_clean prefix +
    whole reviews to 2048) on the fly from the pool — the design the user
    wanted originally, affordable on 80 GB.
Protocol (identical to the local run): fully inductive (train gallery =
1,613 train games), frozen tower tau=0.02 for CE-family, rejection view
sampler a(L)=0.2+0.7*(L-Lmin)/(Lmax-Lmin) over WHOLE reviews (no truncation),
game-level 2:1 doc-gating in the head, selection score
vsel = max(S(non@1;0.45), S(non@5;0.65)) + S(neu@1;0.85).

Resume: tower checkpoints and every ft4var json are skipped when present.
Output: <out-dir>/<files mirroring the local naming>.
"""
import argparse
import json
import math
import re
import time
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _tuc

ARMS = {
    "wcle_ice_icetf": ("ice", "ice"),
    "wcle_i2ce_icetf": ("i2ce", "ice"),
    "wcle_slotc10i2ce_icetf": ("slotc10i2ce", "ice"),  # i2ce + slot-C x1.0
    "wcle_as8dc5i2ce_icetf": ("as8dc5i2ce", "ice"),    # DC-ASGD sim: 8 workers, lambda 0.5
    "wcle_ce_cetf": ("ce", "ce"),
    "wcle_byol_bytf": ("byol", "by"),
    "wcle_mv3_bytf": ("mv3", "by"),   # MoCo v3 analog: in-batch InfoNCE
    # (no queue, per v3), EMA target (m=0.99) + prediction head; the
    # symmetrized q/k pairing runs over all ordered view pairs.
    "wcle_arc_arctf": ("arc", "arc"),
    "wcle_cegate1_icetf": ("cegate1", "ice"),
    "wcle_cegate2_icetf": ("cegate2", "ice"),
    "wcle_cegate3_icetf": ("cegate3", "ice"),
    "wcle_cegate4_icetf": ("cegate4", "ice"),
    "wcle_cegate1w_icetf": ("cegate1w", "ice"),
    "wcle_cegate2w_icetf": ("cegate2w", "ice"),
    "wcle_igate1_icetf": ("igate1", "ice"),
    "wcle_igate1w_icetf": ("igate1w", "ice"),
    "wcle_rgate2_icetf": ("rgate2", "ice"),        # CE on RANDOM coverage-matched set
    "wcle_nodoc_i2ce_icetf": ("nodoc", "ice"),     # zero doc views, CE all + I x2
    # ---- 5fold-wave2 (user 2026-07-19, w9_experiment_5fold_2.ipynb) ----
    "wcle_slot8i2cemean_icetf": ("i2ce", "ice"),   # 8-slot tower, plain i2ce
    #   loss (slot count parsed from the ARM NAME -> NSLOT; loss untouched)
    "wcle_mq3072i2ce_icetf": ("mq3072i2ce", "ice"),   # MoCo queue + view-I x2
    "wcle_mq3072ce_cetf": ("mq3072ce", "ce"),         # MoCo queue, CE only
    "wcle_epdb_v20i10c20_cetf": ("epdb_v20i10c20", "ce"),   # VICReg epd b=all
    "wcle_epd_v20i10c20_cetf": ("epd_v20i10c20", "ce"),     # VICReg epd bs=192
    "wcle_vfai2ce_icetf": ("vfai2ce", "ice"),   # views-first anchors + I x2
    "wcle_ma2i2ce_icetf": ("ma2i2ce", "ice"),   # MULTI-ANCHOR (user
    #   2026-07-23): 2 independent teacher packs per game, BOTH in the
    #   gallery as separate entries; CE is multi-positive over them; no
    #   pack averaging and no pack-pack pull. Eval takes per-game max.   # views-first anchors + I x2
    #   (user 2026-07-23: the step's 4 student views open the teacher pack,
    #   fixed pack fills the rest; teacher = superset of student evidence)
    #   Audited note: store-tier games' doc view duplicates the pack's store
    #   prefix (2x weight); wiki-tier gains new doc evidence. By design --
    #   review views likewise duplicate pack reviews; eval packs unchanged.
    #   (user 2026-07-22: batch=all retired -- 22h/fold; bs=192 is the 5fold arm)
    "wcle_i2au2_icetf": ("i2au2", "ice"),    # I2CE, CE->align+uniform (t=2 soft)
    "wcle_i2au25_icetf": ("i2au25", "ice"),  # I2CE, CE->align+uniform (t=25 = CE tau)
    #   keeps i2ce's view-view I (IW=2); ONLY the CE term is replaced by
    #   explicit Wang&Isola align(view->own anchor) + Gaussian uniformity
    #   (view vs WRONG anchors, kernel exp(2t(cos-1))). Tests whether I
    #   stabilizes align+uniform the way it stabilizes CE (anchor-only a+u
    #   without a separate view-view I collapsed to .005).
    "wcle_i4uni2_icetf": ("i4uni2", "ice"),  # pure Wang&Isola align+uniform
    "wcle_i6uni2_icetf": ("i6uni2", "ice"),  # (views only, anchor-free): view-view
    #   align (IW=4/6 x mean(1-cos)) + batch-view Gaussian uniformity (t=2).
    #   Controlled loss ablation vs CE -- SAME tower/views/2000ep/cvsel/readout,
    #   ONLY the loss changes (no anchor softmax). i4=align_w2, i6=align_w3.
    "wcle_bce_cetf": ("bce", "ce"),   # SimCLR-style in-batch NT-Xent over
    # the step's 4*bs view encodings (ring-sibling positives, no anchors)
    "wcle_swin168step84loop2i2ce_icetf": ("swin168step84loop2i2ce", "ice"),
    "wcle_swin84step42loop2i2ce_icetf": ("swin84step42loop2i2ce", "ice"),
    #   20G recipe (user 2026-07-22): W=84 window x cap 2048 -- both proven
    #   free (A4 window sweep + A2 saturation); store 8.5G, fits a 20G card.
    # ---- tau sweep (user 2026-07-19, w9_i2ce_t.ipynb) ----
    "wcle_i2cet05_icetf": ("i2ce", "ice"),   # tau = 0.05 (from ARM name)
    "wcle_i2cet10_icetf": ("i2ce", "ice"),   # tau = 0.10
    "wcle_i2ce_icetl": ("i2ce", "ice"),      # LEARNABLE tau (init 0.02)
    # sliding fresh-window i2ce (user): CE field = now-batch 192 anchors +
    # L ring windows of W (fresh, grad, no cache); micro-pass backward.
}
_IW = {"ice": 1.0, "i2ce": 2.0, "ma2i2ce": 2.0, "vfai2ce": 2.0, "mq3072i2ce": 2.0, "swin168step84loop2i2ce": 2.0, "swin84step42loop2i2ce": 2.0, "cegate1": 1.0, "cegate2": 2.0, "cegate3": 3.0,
       "cegate4": 4.0, "cegate1w": 1.0, "cegate2w": 2.0, "igate1": 1.0,
       "igate1w": 1.0, "rgate2": 2.0, "nodoc": 2.0,
       "i4uni2": 4.0, "i6uni2": 6.0, "i2au2": 2.0, "i2au25": 2.0}
SPLIT_SEED = 20260711
DM, HEADS, NV = 128, 4, 4
ARC_S_T, ARC_M_T = 50.0, 0.2       # tower ArcFace
K1, K2, S_A, S_B = 1.0, 1.0, 10.0, 1.0   # vsel piecewise score


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--anchor-cap", type=int, default=512)
    ap.add_argument("--pfc-shards", type=int, default=0,
                    help="distributed sharded softmax (Partial-FC): split "
                         "the ring into K shards, each sweeps a swin window "
                         "inside its 1/K, CE partition = exact union of "
                         "eNow + all K shard windows (single-proc concat == "
                         "multi-node per-step all_reduce)")
    ap.add_argument("--pfc-window", type=int, default=0,
                    help="per-shard negatives/step (0 = SWIN_W); set below "
                         "the shard size to probe low per-step coverage")
    ap.add_argument("--view-w", type=int, default=16,
                    help="STUDENT view size: soft sentence-stop threshold for "
                         "each small view (whole reviews never split). fs "
                         "worker default 16; sweep for student scaling.")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--cv-seed", type=int, default=20260711)
    ap.add_argument("--no-sp-view", action="store_true",
                    help="doc tier = wiki_clean ONLY (drop the sp_raw fallback)")
    ap.add_argument("--doc-lead", type=int, default=0,
                    help=">0: truncate doc VIEWS to the first N sentences "
                         "(length-attribution ablation)")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training/projection; just refresh the "
                         "trajectory from existing tower npz and rewrite "
                         "zsbest (used to backfill test_tag post-hoc)")
    ap.add_argument("--full-pool-path", default="",
                    help="path of full_pool_fp16.npy (meta npz expected beside "
                         "it); empty = <data-dir>/full_pool_fp16.npy")
    ap.add_argument("--full-pool", action="store_true",
                    help="draw training views from the FULL review corpus "
                         "(host-RAM flat npy) instead of the 2048-sent pool")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--ckpt-seeds", type=int, default=2)
    ap.add_argument("--topup-seeds", type=int, default=10)
    ap.add_argument("--claim-file", default="",
                    help="claim file on the shared volume; when set, a 30s "
                         "heartbeat thread keeps it fresh (silent >2 min = "
                         "host presumed dead, job claimable by other hosts)")
    ap.add_argument("--head", action="store_true", default=False,
                    help="legacy FT head phase (ZS-only is the default)")
    ap.add_argument("--init-ckpt", default="",
                    help="warm-start tower weights from this ckpt (dict with "
                         "a model key or plain state dict); name suffix _bw")
    ap.add_argument("--measure-vram", default="",
                    help="run 3 real steps, write peak max_memory_allocated "
                         "bytes to this file, exit(0). Scheduler warmup.")
    ap.add_argument("--wiki-src", choices=("clean", "llm"), default="clean",
                    help="wiki doc source: clean=raw wiki text; llm=sentence-wise "
                         "faithful LLM rewrite (name suffix _wllm)")
    return ap.parse_args()


class SetPoolN(nn.Module):
    def __init__(s, N):
        super().__init__()
        s.q0 = nn.Parameter(torch.randn(1, N, DM) * 0.02)
        s.attn = nn.MultiheadAttention(DM, HEADS, kdim=1024, vdim=1024,
                                       batch_first=True)
        s.head = nn.Sequential(nn.Linear(DM, 256), nn.GELU(), nn.Linear(256, DM))

    def forward(s, S, m=None):
        a, _ = s.attn(s.q0.expand(S.shape[0], -1, -1), S.float(), S.float(),
                      key_padding_mask=m, need_weights=False)
        if getattr(s, "slot_buf", None) is not None:
            s.slot_buf.append(a)      # pre-pool slots, graph kept (slot-C)
        return F.normalize(s.head(a.mean(1)), dim=-1)


def rown(x, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    s = x.std(-1, keepdims=True)
    return (x - m) / (s + eps)


def S_fn(x, x0):
    return (-K1 * (math.exp(S_A * (x0 - x)) - 1) if x < x0
            else K2 * (x - x0) ** S_B)


def start_claim_beat(path):
    """Heartbeat thread: every BEAT(30)s rewrite the claim file with
    "<host> <now>". If the claim changes owner (this host was presumed dead
    after 2 min of silence and another machine took the job over), YIELD by
    exiting immediately -- two workers on one job would corrupt checkpoints."""
    import os
    import socket
    import threading
    import time
    from pathlib import Path
    host = socket.gethostname() + ":" + os.environ.get("RUNPOD_POD_ID", "?")
    p = Path(path)

    def _beat():
        while True:
            try:
                parts = p.read_text().split()
                if parts and parts[0] != host:
                    print(f"[beat] claim {p.name} now owned by {parts[0]} "
                          f"-- yielding (this host was presumed dead)",
                          flush=True)
                    os._exit(3)
                tmp = p.with_name(p.name + ".beat_tmp")
                tmp.write_text(f"{host} {time.time():.0f}")
                os.replace(tmp, p)                       # atomic on the volume
            except Exception:
                pass
            time.sleep(30)

    threading.Thread(target=_beat, daemon=True).start()


def main():
    args = parse_args()
    if args.claim_file:
        start_claim_beat(args.claim_file)
    import sys
    sys.path.insert(0, args.repo)
    from VICReg_review.text_variant_eval import (train_anchor_ridge,
                                                 make_or_load_split, micro_prf)
    from VICReg_review.model import GameCentroidExpander, vicreg_loss

    tower_kind, FT = ARMS[args.arm]
    slot_am = re.match(r"wcle_slot(\d+)i2cemean_", args.arm)
    NSLOT = int(slot_am.group(1)) if slot_am else 4   # slot count from ARM name
    mq_m = re.match(r"mq(\d+)(i2ce|ce)$", tower_kind)
    MQ_LEN = int(mq_m.group(1)) if mq_m else 0        # MoCo FIFO ring length
    MQ_M = 0.99                                       # shadow weight-EMA momentum
    BCE = tower_kind == "bce"          # in-batch views, no gallery
    UNI = tower_kind in ("i4uni2", "i6uni2")   # W&I: batch uniformity, no anchor
    UNI_T = 2.0                        # W&I uniformity Gaussian t (repo default)
    _au = re.match(r"i2au(\d+)$", tower_kind)   # I2CE with CE->align+uniform
    AU = bool(_au)                     # (anchor align + anchor Gaussian uniform)
    AU_T = int(_au.group(1)) if _au else 0
    VFA = tower_kind == "vfai2ce"      # views-first anchors (user 2026-07-23)
    ma_m = re.match(r"ma(\d+)i2ce$", tower_kind)
    MA_N = int(ma_m.group(1)) if ma_m else 0   # multi-anchor pack count
    as_m = re.match(r"as(\d+)(?:dc(\d+))?i2ce$", tower_kind)
    AS_M = int(as_m.group(1)) if as_m else 0          # async workers (sim)
    DC_L = (int(as_m.group(2)) / 10.0) if (as_m and as_m.group(2)) else 0.0
    sc_m = re.match(r"slotc(\d+)i2ce$", tower_kind)
    SC_W = (int(sc_m.group(1)) / 10.0) if sc_m else 0.0   # slot-C weight
    swin_m = re.match(r"swin(\d+)step(\d+)loop(\d+)i2ce$", tower_kind)
    SWIN = bool(swin_m)                # fresh sliding-window CE field
    SWIN_W = int(swin_m.group(1)) if swin_m else 0
    SWIN_S = int(swin_m.group(2)) if swin_m else 0
    SWIN_L = int(swin_m.group(3)) if swin_m else 0
    IW = _IW.get(tower_kind, 0.0)
    if re.match(r"slotc\d+i2ce$", tower_kind):
        IW = 2.0                       # slotc = i2ce family: I x2
    if AS_M:
        IW = 2.0                       # as*dc* = i2ce family: I x2
    HIW = _IW.get(tower_kind, 1.0)
    CE_GATED = tower_kind.startswith("cegate") or tower_kind == "rgate2"
    I_GATED = tower_kind.startswith("igate")
    name = (f"w9cv_{args.arm}_fold{args.fold}"
            + (f"_g{args.anchor_cap}" if args.anchor_cap != 512 else "")
            + ("_nsp" if args.no_sp_view else "")
            + (f"_ld{args.doc_lead}" if args.doc_lead else "")
            + ("_wllm" if args.wiki_src == "llm" else "")
            + ("_bw" if args.init_ckpt else "")
            + ((f"_pfc{args.pfc_shards}"
                + (f"w{args.pfc_window}" if args.pfc_window else ""))
               if args.pfc_shards else "")
            + (f"_w{args.view_w}" if args.view_w != 16 else "")
            + ("_fp" if args.full_pool else ""))
    dev = torch.device("cuda")
    C, OUT = Path(args.data_dir), Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] tower={tower_kind} ft={FT} IW={IW} anchor_cap={args.anchor_cap}",
          flush=True)

    # ---------------- corpus: ALL of it onto the GPU ----------------
    G = np.load(C / "games.npz", allow_pickle=True)
    names = [str(x) for x in G["names"]]
    NG = len(names)
    n2i = {n: i for i, n in enumerate(names)}
    appid2name = {n.split("_")[0]: n for n in names}

    RIDp = np.load(C / "wscan_pool_rev_rid.npy")
    plen = np.load(C / "wscan_pool_rev_len.npy")
    need_pool = (not args.full_pool) or args.anchor_cap != 512
    POOL = (torch.tensor(np.load(C / "wscan_pool_rev.npy"), device=dev)
            if need_pool else None)                                      # fp16
    if args.full_pool:
        fp = Path(args.full_pool_path) if args.full_pool_path             else C / "full_pool_fp16.npy"
        FULLV = np.load(fp, mmap_mode="r")
        FMETA = np.load(fp.with_name("full_pool_meta.npz"), allow_pickle=True)
        print(f"full pool source: {fp}", flush=True)
        f_gro = FMETA["game_review_offsets"]
        f_ro = FMETA["review_offsets"]
        f_e2i = {str(n): i for i, n in enumerate(FMETA["game_names"])}
        print(f"full pool: {FULLV.shape[0]:,} sentences (host RAM/page cache)",
              flush=True)
    A = np.load(C / "wiki_eval.npz", allow_pickle=True)
    SA = rown(torch.tensor(A["S"]).to(dev).float()).half()
    mA = torch.arange(SA.shape[1], device=dev)[None, :] >= \
        torch.tensor(A["S_len"]).to(dev)[:, None]
    gA_t = torch.tensor(A["gidx"]).to(dev)
    variants = [str(x) for x in A["variants"]]
    art_games = [str(x) for x in A["names"]]
    Qs = np.load(C / "ss_queries_rev.npz")
    # pseudo-queries as an mmap (page cache SHARED across all GPU workers on
    # the machine -- 9 workers cost the same RAM as one; only touched at
    # checkpoint projection). Saves 8.5 GB VRAM and 8.5 GB RAM per worker.
    QS_G = np.load(C / "ss_queries_rev_S.npy", mmap_mode="r")            # fp16 mmap
    y = np.load(C / "tag_labels.npz", allow_pickle=True)["y"]

    def load_views(fname):
        d = np.load(C / fname, allow_pickle=True)
        Sx = rown(torch.tensor(d["S"]).to(dev).float()).half()
        mx = torch.arange(Sx.shape[1], device=dev)[None, :] >= \
            torch.tensor(d["S_len"]).to(dev)[:, None]
        return d, Sx, mx

    WK, SW, mW = load_views(f"wiki_{args.wiki_src}_views.npz")
    ST, SS, mS = load_views("sp_raw_views.npz")
    if args.doc_lead:
        mW = mW | (torch.arange(SW.shape[1], device=dev)[None, :] >= args.doc_lead)
        mS = mS | (torch.arange(SS.shape[1], device=dev)[None, :] >= args.doc_lead)
        print(f"doc views truncated to lead-{args.doc_lead}", flush=True)

    # ---------------- split + inductive machinery ----------------
    sp = json.loads((C / "wiki_eval_split.json").read_text())
    universe = sorted(set(sp["test"]) | set(sp["val"]) | set(sp["train"]))
    rngF = np.random.default_rng(args.cv_seed)
    perm = rngF.permutation(len(universe))
    folds = np.array_split(perm, args.n_folds)
    te = {universe[i] for i in folds[args.fold]}
    va = {universe[i] for i in folds[(args.fold + 1) % args.n_folds]}
    test_g = {appid2name[a] for a in te}
    val_g = {appid2name[a] for a in va}
    traing = {appid2name[a] for a in universe} - test_g - val_g
    print(f"fold {args.fold}/{args.n_folds}: test {len(test_g)} / val {len(val_g)} / "
          f"train-doc {len(traing)}", flush=True)
    excl = test_g | val_g
    train_pool_games = np.array([i for i in range(NG) if names[i] not in excl])
    pos_of_g = np.full(NG, -1, dtype=np.int64)
    pos_of_g[train_pool_games] = np.arange(len(train_pool_games))
    pos_of_g_t = torch.tensor(pos_of_g)
    tp_t = torch.tensor(train_pool_games).to(dev)

    g2wiki = {int(WK["gidx"][i]): i for i in range(len(WK["gidx"]))
              if str(WK["names"][i]) not in excl}
    g2store = {int(ST["gidx"][i]): i for i in range(len(ST["gidx"]))
               if str(ST["names"][i]) not in excl and int(ST["gidx"][i]) not in g2wiki}
    n_doc_cover = len(g2wiki) + len(g2store)
    if args.no_sp_view:
        g2store = {}                    # wiki-pure tower views (user decree)
        tiers = [(g2wiki, SW, mW)]
    else:
        tiers = [(g2wiki, SW, mW), (g2store, SS, mS)]
    if tower_kind == "nodoc":
        tiers = []                      # ZERO doc views: all four views = reviews
    if CE_GATED or I_GATED:
        if tower_kind == "rgate2":
            # coverage-matched RANDOM gate (control for "CE dose reduction"):
            # same #games as the doc-bearing set, drawn independently of docs.
            rngG = np.random.default_rng(SPLIT_SEED + 7)
            gate_games = set(int(g) for g in rngG.choice(
                train_pool_games, size=min(n_doc_cover, len(train_pool_games)),
                replace=False))
            print(f"gate(CE) scope=RANDOM: {len(gate_games)} games "
                  f"(coverage-matched)", flush=True)
        else:
            scope_wiki = tower_kind.endswith("w")
            gate_games = set(g2wiki) if scope_wiki else set(g2wiki) | set(g2store)
            print(f"gate({'CE' if CE_GATED else 'I'}) scope="
                  f"{'wiki' if scope_wiki else 'doc'}: {len(gate_games)} games",
                  flush=True)

    va_neu = [i for i, g in enumerate(art_games) if g in val_g and variants[i] == "neutral"]
    va_non = [i for i, g in enumerate(art_games) if g in val_g and variants[i] == "noname"]
    hgi = {n2i[g] for g in excl}
    q_train = np.where(~np.isin(Qs["gidx"], list(hgi)))[0]
    qpos_t = pos_of_g_t[Qs["gidx"]].to(dev)

    targs = SimpleNamespace(tag_text_train_frac=0.7, tag_text_val_frac=0.15,
                            tag_text_split_seed=42, seed=42,
                            tag_text_threshold_steps=33)
    tag_split = make_or_load_split(C / "_tag_splitM.json", names, targs)

    # ---------------- anchors (512 shipped / 2048 built here) ----------------
    sp_row = {int(ST["gidx"][i]): i for i in range(len(ST["gidx"]))}
    if args.anchor_cap == 512:
        GALd = np.load(C / "wscan_gal_rev.npz")
        SGal = torch.tensor(GALd["gal"]).to(dev)
        gal_len = torch.tensor(GALd["gal_len"]).to(dev)
        gal_doc = torch.tensor(GALd["gal_doc_len"]).to(dev)
    elif (C / f"wscan_gal_rev_g{args.anchor_cap}.npz").exists():
        # PREBUILT pack from the volume. Built from the full corpus, so it
        # actually reaches the cap (med 4096 at cap 4096) instead of the
        # ~2068 the 2048-sentence POOL can supply. This matters for any arm
        # that SUB-SAMPLES the pack (multi-anchor): with a POOL-built pack
        # the per-game review supply equals the per-pack budget and every
        # pack covers the whole pool.
        GALd = np.load(C / f"wscan_gal_rev_g{args.anchor_cap}.npz")
        SGal = torch.tensor(GALd["gal"]).to(dev)
        gal_len = torch.tensor(np.asarray(GALd["gal_len"], np.int64)).to(dev)
        gal_doc = torch.tensor(np.asarray(GALd["gal_doc_len"], np.int64)).to(dev)
        print(f"anchors: PREBUILT g{args.anchor_cap} pack, used med "
              f"{int(gal_len.float().median())}", flush=True)
    else:
        GCAP = args.anchor_cap
        print(f"building {GCAP}-sentence anchors on GPU "
              f"(POOL source, caps at {POOL.shape[1] if POOL is not None else '?'} "
              f"review sentences) ...", flush=True)
        SGal = torch.zeros(NG, GCAP, 1024, dtype=torch.float16, device=dev)
        gal_len = torch.zeros(NG, dtype=torch.long, device=dev)
        gal_doc = torch.zeros(NG, dtype=torch.long, device=dev)
        rngA = np.random.default_rng(SPLIT_SEED + 5)
        for g in range(NG):
            row = 0
            if g in sp_row:
                dl = int(ST["S_len"][sp_row[g]])
                SGal[g, :dl] = SS[sp_row[g], :dl]
                row = dl
                gal_doc[g] = dl
            r_lens = np.bincount(RIDp[g, :plen[g]])
            r_starts = np.zeros(len(r_lens), np.int64)
            r_starts[1:] = np.cumsum(r_lens)[:-1]
            for j in rngA.permutation(len(r_lens)):
                L = int(r_lens[j])
                if row + L <= GCAP:
                    SGal[g, row:row + L] = POOL[g, r_starts[j]:r_starts[j] + L]
                    row += L
            gal_len[g] = row
        print(f"anchors: used med {int(gal_len.float().median())}", flush=True)
    mGal = torch.arange(SGal.shape[1], device=dev)[None, :] >= gal_len[:, None]
    mGal_nd = mGal | (torch.arange(SGal.shape[1], device=dev)[None, :] <
                      gal_doc[:, None])
    idxP = mGalP = aliveW = ma_dead = None
    if MA_N:
        # N independent random samples of each game's anchor sentences.
        # Packs MAY overlap (independent draws, not disjoint slices); a
        # per-game fixed seed makes them bit-identical across restarts and
        # between training and evaluation. The doc prefix is pinned to the
        # head of pack 0 only. Games with no reviews keep pack 0 and mark
        # the rest dead (aliveW 0) so they never enter a numerator and are
        # masked out of every denominator.
        MA_SEED = 20260723
        _h = SGal.shape[1] // MA_N
        assert _h >= 512, f"ma{MA_N} needs cap//{MA_N} >= 512, got {_h}"
        idxP = torch.zeros(MA_N, NG, _h, dtype=torch.long)
        _mP = torch.ones(MA_N, NG, _h, dtype=torch.bool)
        aliveW = torch.ones(NG, MA_N)
        for _g in range(NG):
            _l, _d = int(gal_len[_g]), int(gal_doc[_g])
            _R = _l - _d
            _gen = torch.Generator().manual_seed(MA_SEED + _g)
            for _k in range(MA_N):
                _dk = min(_d, _h) if _k == 0 else 0
                _need = _h - _dk
                if _need == 0:                      # pack 0 all doc prefix
                    idxP[_k, _g] = torch.arange(_h)
                    _mP[_k, _g] = False
                elif _R > 0:
                    _pm = torch.cat([torch.randperm(_R, generator=_gen)
                                     for _ in range(-(-_need // _R))])[:_need]
                    idxP[_k, _g] = (torch.cat([torch.arange(_dk), _d + _pm])
                                    if _dk else _d + _pm)
                    _mP[_k, _g] = False
                elif _k == 0:                       # doc-only, short prefix
                    idxP[_k, _g, :_dk] = torch.arange(_dk)
                    _mP[_k, _g, :_dk] = False
                else:                               # dead pack
                    _mP[_k, _g, 0] = False
                    aliveW[_g, _k] = 0.0
        idxP = idxP.to(dev)
        aliveW = aliveW.to(dev)
        mGalP = [_mP[k].to(dev) for k in range(MA_N)]
        print(f"multi-anchor: {MA_N} packs x {_h} sentences; "
              f"{int((aliveW[:, 1:] == 0).any(1).sum())} games with dead packs",
              flush=True)
        # gallery rows that are dead packs, in the SAME k-major order
        # gallery_train_ma emits them (pack k of train-pos p at k*n_train+p).
        _tp = torch.as_tensor(train_pool_games, device=dev)
        ma_dead = torch.cat([aliveW[_tp, k] == 0 for k in range(MA_N)])
        # PACK INDEPENDENCE GUARD: a pack is a random half of the game's
        # review sentences. If a game has fewer than _h review sentences the
        # sampler tiles a permutation, so every pack covers the whole pool
        # and the packs collapse onto each other -- multi-anchor degenerates
        # to single-anchor. That happens when the anchors are built from the
        # 2048-sentence POOL instead of --full-pool.
        # Two packs of _h drawn from R review sentences overlap by about
        # _h/R. R == _h is TOTAL degeneracy (each pack is the whole pool),
        # not the boundary case -- so test the overlap itself, not R < _h.
        _Rv = (gal_len - gal_doc).clamp(min=1)
        _ov = (torch.full_like(_Rv, _h, dtype=torch.float) / _Rv).clamp(max=1.0)
        _mo = float(_ov.median())
        print(f"pack overlap: median {_mo:.0%} (target ~50%; review supply "
              f"median {int(_Rv.median())} vs per-pack {_h})", flush=True)
        assert _mo <= 0.75, (
            f"packs would overlap {_mo:.0%} of their sentences (review supply "
            f"median {int(_Rv.median())} vs per-pack budget {_h}) -- the two "
            f"anchors would be near-identical and multi-anchor is meaningless. "
            f"Build/upload the prebuilt wscan_gal_rev_g{args.anchor_cap}.npz "
            f"(full-corpus anchors) or lower MA_N / raise --anchor-cap.")

    def ma_sg(k, rows):
        """Pack k's sentences for game rows -> (len(rows), _h, D)."""
        return SGal[rows[:, None], idxP[k][rows]]
    if tower_kind.startswith(("epd", "vic")):
        # VICReg family: training never reads the anchors (they are eval-
        # only), and batch=all view activations need the room -- park the
        # anchor tensors on pinned host RAM; the gallery encoders move
        # chunk slices back on the fly.
        SGal = SGal.cpu().pin_memory()
        mGal = mGal.cpu()
        mGal_nd = mGal_nd.cpu()
        torch.cuda.empty_cache()
        print("anchors parked on host RAM (VICReg family)", flush=True)
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    # ---------------- review table + GPU view sampler ----------------
    REV_TAB = []
    for g in range(NG):
        if args.full_pool:
            i = f_e2i[names[g]]
            r0, r1 = int(f_gro[i]), int(f_gro[i + 1])
            starts = f_ro[r0:r1].astype(np.int64)          # ABSOLUTE sent idx
            lens = (f_ro[r0 + 1:r1 + 1] - f_ro[r0:r1]).astype(np.int64)
        else:
            r = RIDp[g]
            n = int(r.max()) + 1
            lens = np.bincount(r[r >= 0], minlength=n).astype(np.int64)
            starts = np.zeros(n, np.int64)
            starts[1:] = np.cumsum(lens)[:-1]
        if lens.max() > lens.min():
            a = 0.2 + 0.7 * (lens - lens.min()) / (lens.max() - lens.min())
        else:
            a = np.full(len(lens), 0.9)
        REV_TAB.append((starts, lens, a))

    def sample_views(gids, W, rng):
        idx_rows, lens = [], []
        for g in gids:
            st, ln, a = REV_TAB[int(g)]
            n = len(ln)
            tot, chosen = 0, []
            taken = np.zeros(n, bool)
            while tot < W and not taken.all():
                i = int(rng.integers(n))
                if taken[i]:
                    continue
                if rng.random() < a[i]:
                    taken[i] = True
                    chosen.append(i)
                    tot += int(ln[i])
            seg = np.concatenate([np.arange(st[i], st[i] + ln[i]) for i in chosen])
            idx_rows.append(seg)
            lens.append(len(seg))
        LM = max(lens)
        if args.full_pool:
            flat = np.concatenate(idx_rows)
            order = np.argsort(flat, kind="stable")          # sorted reads
            vec = np.empty((len(flat), 1024), np.float32)
            vec[order] = FULLV[flat[order]].astype(np.float32)
            vec = (vec - vec.mean(-1, keepdims=True)) / \
                (vec.std(-1, keepdims=True) + 1e-6)          # rown on the fly
            X = np.zeros((len(gids), LM, 1024), np.float16)
            pos = 0
            for k, seg in enumerate(idx_rows):
                X[k, :len(seg)] = vec[pos:pos + len(seg)]
                pos += len(seg)
            S = torch.from_numpy(X).to(dev, non_blocking=True)
            L = torch.as_tensor(lens, device=dev)
            m = torch.arange(LM, device=dev)[None, :] >= L[:, None]
            return S, m
        idx = np.zeros((len(gids), LM), np.int64)
        for k, seg in enumerate(idx_rows):
            idx[k, :len(seg)] = seg
        gid_t = torch.as_tensor(np.asarray(gids), device=dev).view(-1, 1).expand(-1, LM)
        S = POOL[gid_t, torch.as_tensor(idx, device=dev)]      # one gather kernel
        L = torch.as_tensor(lens, device=dev)
        m = torch.arange(LM, device=dev)[None, :] >= L[:, None]
        return S.masked_fill(m.unsqueeze(-1), 0), m

    def pad_flat(off, idx):
        segs = [(int(off[j]), int(off[j + 1])) for j in idx]
        LM = max(e - s for s, e in segs)
        X = np.zeros((len(segs), LM, 1024), np.float16)
        L = torch.zeros(len(segs), dtype=torch.long, device=dev)
        for k, (s, e) in enumerate(segs):
            X[k, :e - s] = QS_G[s:e]
            L[k] = e - s
        m = torch.arange(LM, device=dev)[None, :] >= L[:, None]
        return torch.from_numpy(X).to(dev, non_blocking=True), m

    def gallery(model, chunk=128, grad=False):
        outs = []
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            for i in range(0, NG, chunk):
                outs.append(model(
                    SGal[i:i + chunk].to(dev, non_blocking=True),
                    mGal[i:i + chunk].to(dev, non_blocking=True)))
        return torch.cat(outs)

    def gallery_nodoc(model, chunk=128):
        outs = []
        with torch.no_grad():
            for i in range(0, NG, chunk):
                outs.append(model(
                    SGal[i:i + chunk].to(dev, non_blocking=True),
                    mGal_nd[i:i + chunk].to(dev, non_blocking=True)))
        return torch.cat(outs)

    def gallery_train_ma(model, chunk=128):
        """[MA_N * n_train, D]: pack k of train-pool position p at
        k*n_train + p. Every entry carries gradient."""
        rows = train_pool_games
        outs = []
        for k in range(MA_N):
            for i in range(0, len(rows), chunk):
                r = torch.as_tensor(rows[i:i + chunk], device=dev)
                outs.append(model(ma_sg(k, r), mGalP[k][r]))
        return torch.cat(outs)

    def gallery_ma(model, chunk=128):
        """[MA_N, NG, D] for evaluation (no grad)."""
        outs = []
        with torch.no_grad():
            for k in range(MA_N):
                ok = []
                for i in range(0, NG, chunk):
                    r = torch.arange(i, min(i + chunk, NG), device=dev)
                    ok.append(model(ma_sg(k, r), mGalP[k][r]))
                outs.append(torch.cat(ok))
        return torch.stack(outs)

    def gallery_train(model, chunk=128):
        rows = train_pool_games
        outs = []
        for i in range(0, len(rows), chunk):
            r = torch.as_tensor(rows[i:i + chunk], device=SGal.device)
            outs.append(model(SGal[r].to(dev, non_blocking=True),
                              mGal[r].to(dev, non_blocking=True)))
        return torch.cat(outs)

    try:
        amp_cls = lambda: torch.amp.GradScaler("cuda")
        amp_cls()
    except Exception:
        amp_cls = lambda: torch.cuda.amp.GradScaler()

    def arcface_ce(logits, tgt):
        logits = logits.float()
        cos_t = logits.gather(1, tgt[:, None]).clamp(-1 + 1e-7, 1 - 1e-7)
        phi = torch.cos(torch.acos(cos_t) + ARC_M_T)
        phi = torch.where(cos_t > math.cos(math.pi - ARC_M_T),
                          phi, cos_t - ARC_M_T * math.sin(ARC_M_T))
        return F.cross_entropy(logits.scatter(1, tgt[:, None], phi) * ARC_S_T, tgt)

    def assemble_doc_view(model, gids, W, rng, bs):
        Zlast = torch.empty(bs, DM, device=dev, dtype=torch.float16)
        assigned = np.zeros(bs, bool)
        for g2x, Sx, mx in tiers:
            msk = np.array([(not a) and (g in g2x) for a, g in zip(assigned, gids)])
            if msk.any():
                rows = [g2x[g] for g in gids[msk]]
                Zlast[torch.tensor(msk).to(dev)] = model(Sx[rows], mx[rows]).half()
                assigned |= msk
        rest = ~assigned
        if rest.any():
            Zlast[torch.tensor(rest).to(dev)] = model(*sample_views(gids[rest], W, rng)).half()
        return Zlast

    _TIER_LENS = [(~mx).sum(1).cpu().numpy() for _, __, mx in tiers]

    def doc_view_raw(gids, W, rng):
        # raw twin of assemble_doc_view: each game's doc-view SENTENCES
        # (wiki tier, then store tier, then a review-view fallback),
        # reusable by both the view encode and the vfa pack build.
        out = [None] * len(gids)
        for ti, (g2x, Sx, mx) in enumerate(tiers):
            for k, g in enumerate(gids):
                if out[k] is None and g in g2x:
                    r = g2x[g]
                    out[k] = Sx[r, :int(_TIER_LENS[ti][r])]
        rest = [k for k in range(len(gids)) if out[k] is None]
        if rest:
            S2, m2 = sample_views(np.asarray(gids)[rest], W, rng)
            L2 = (~m2).sum(1).cpu().numpy()
            for j, k in enumerate(rest):
                out[k] = S2[j, :int(L2[j])]
        return out

    def encode_doc_rows(model, rows):
        LM = max(r.shape[0] for r in rows)
        S = torch.zeros(len(rows), LM, 1024, dtype=torch.float16, device=dev)
        for k, r in enumerate(rows):
            S[k, :r.shape[0]] = r
        L = torch.as_tensor([r.shape[0] for r in rows], device=dev)
        m = torch.arange(LM, device=dev)[None, :] >= L[:, None]
        return model(S, m)

    def vfa_packs(gids, vraw, dparts):
        # views-first teacher packs for the batch games: the step's three
        # review views + the doc view open the pack, the game's fixed
        # anchor pack fills the remainder, truncated at the cap.
        GC = SGal.shape[1]
        gl = gal_len.detach().cpu().numpy()
        vlens = [(~mv).sum(1).cpu().numpy() for _, mv in vraw]
        S = torch.zeros(len(gids), GC, 1024, dtype=torch.float16, device=dev)
        lens = []
        for k, g in enumerate(gids):
            parts = [Sv[k, :int(vlens[v][k])] for v, (Sv, _) in enumerate(vraw)]
            parts.append(dparts[k])
            parts.append(SGal[int(g), :int(gl[int(g)])])
            row = torch.cat(parts)[:GC]
            S[k, :row.shape[0]] = row
            lens.append(row.shape[0])
        L = torch.as_tensor(lens, device=dev)
        m = torch.arange(GC, device=dev)[None, :] >= L[:, None]
        return S, m

    pairs = list(combinations(range(NV), 2))
    t_m = re.match(r"wcle_i2cet(\d{2})_", args.arm)
    TAU = int(t_m.group(1)) / 100.0 if t_m else 0.02   # from ARM name
    TAU_LEARN = args.arm.endswith("_icetl")            # learnable tau arm
    inv_t = 1.0 / TAU
    TAUP = {}                     # filled by train_v4doc for the tl arm

    def _invt():
        # CE logit scale: constant for fixed-tau arms; exp(log_invt)
        # (clamped to tau in [0.005, 0.2]) for the learnable arm.
        return (torch.exp(TAUP["p"]).clamp(5.0, 200.0) if "p" in TAUP
                else inv_t)

    def train_v4doc(seed=0, W=16, bs=192, per_epoch=3072):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = SetPoolN(NSLOT).to(dev)
        if TAU_LEARN:
            # log-parameterized inverse temperature, NO weight decay (wd
            # would pull tau toward 1.0); joins the optimizer + grad clip.
            TAUP["p"] = torch.nn.Parameter(
                torch.tensor(math.log(1.0 / TAU), device=dev))
        params_all = list(model.parameters()) + (
            [TAUP["p"]] if TAU_LEARN else [])
        opt = torch.optim.AdamW(
            [{"params": model.parameters()},
             {"params": [TAUP["p"]], "weight_decay": 0.0}] if TAU_LEARN
            else model.parameters(), lr=5e-4, weight_decay=1e-4)
        amp = amp_cls()
        ckpts = {}
        shadow, mqueue, mq_gid, mq_ptr = None, None, None, 0
        swin_ptr = 0
        PFC = args.pfc_shards
        if PFC:
            _ntr0 = len(train_pool_games)
            _pfc_rows = [np.sort(np.arange(_ntr0)[_k::PFC])
                         for _k in range(PFC)]
            pfc_ptr = [(_k * 997) % max(len(_pfc_rows[_k]), 1)
                       for _k in range(PFC)]   # stagger shard sweeps
            _tpg = np.asarray(train_pool_games)

            def _pfc_enc(_wt):
                # gradient checkpoint: keep only the [W,128] output; the big
                # [W,4096,1024] activation is recomputed one window at a
                # time in backward, so K shard windows never sit live
                # simultaneously (the OOM fix). Exact same output.
                with torch.amp.autocast("cuda"):
                    return model(SGal[_wt], mGal[_wt]).float()
        if MQ_LEN:
            # true MoCo: frozen weight-EMA twin encodes the keys
            import copy
            shadow = copy.deepcopy(model).to(dev)
            for _pp in shadow.parameters():
                _pp.requires_grad_(False)
        RES = OUT / f"resume_{name}.pt"
        start_ep = 0
        if RES.exists():
            st = torch.load(RES, map_location="cpu")
            model.load_state_dict({k: v.to(dev) for k, v in st["model"].items()})
            opt.load_state_dict(st["opt"])
            amp.load_state_dict(st["amp"])
            torch.set_rng_state(st["cpu_rng"])
            torch.cuda.set_rng_state(st["cuda_rng"])
            rng.bit_generator.state = st["np_rng"]
            start_ep = int(st["ep"])
            print(f"RESUME from ep{start_ep}", flush=True)
            if SWIN:
                swin_ptr = int(st.get("swin_ptr", 0))
                if args.pfc_shards and "pfc_ptr" in st:
                    pfc_ptr = list(st["pfc_ptr"])
            if TAU_LEARN and "log_invt" in st:
                TAUP["p"].data = st["log_invt"].to(dev)
            if MQ_LEN and "mqueue" in st:
                shadow.load_state_dict({k: v.to(dev)
                                        for k, v in st["shadow"].items()})
                mqueue = st["mqueue"].to(dev)
                mq_gid = st["mq_gid"].to(dev)
                mq_ptr = int(st["mq_ptr"])
        if start_ep == 0 and not RES.exists():
            # EXTEND fallback (ported from the fs worker for w9_i2ce_continue):
            # the resume bundle is deleted when a run completes; to train PAST
            # the old budget, rebuild from the newest checkpoint. Weights come
            # back exactly; opt/amp/rng restart fresh (rng re-seeded by
            # seed=fold, same semantics as the fs EXTEND).
            cands = [c for c in OUT.glob(f"ckpt_{name}_ep*.pt")
                     if int(c.stem.split("_ep")[-1]) < args.epochs]
            if cands:
                ck = max(cands, key=lambda c: int(c.stem.split("_ep")[-1]))
                st = torch.load(ck, map_location="cpu")
                sd = st["model"] if isinstance(st, dict) and "model" in st else st
                model.load_state_dict({k: v.to(dev) for k, v in sd.items()})
                start_ep = int(ck.stem.split("_ep")[-1])
                print(f"EXTEND from ckpt ep{start_ep} (fresh opt/amp/rng)",
                      flush=True)
                if MQ_LEN and isinstance(st, dict) and "shadow" in st:
                    shadow.load_state_dict({k: v.to(dev)
                                            for k, v in st["shadow"].items()})
        if start_ep == 0 and args.init_ckpt:
            # STAGE-1 handover (concat): warm-start the tower only;
            # opt/amp/rng start fresh. BYOL cv ckpts are dicts with a
            # model key (pred/target ignored).
            st0 = torch.load(args.init_ckpt, map_location="cpu")
            sd0 = st0["model"] if isinstance(st0, dict) and "model" in st0 else st0
            model.load_state_dict({k: v.to(dev) for k, v in sd0.items()})
            print(f"WARM INIT from {args.init_ckpt}", flush=True)
        if MQ_LEN and mqueue is None:
            # prefill the FIFO ring with shadow(=main at t0) keys
            mqueue = torch.zeros(MQ_LEN, DM, device=dev)
            mq_gid = torch.full((MQ_LEN,), -1, dtype=torch.long, device=dev)
            fill = rng.choice(train_pool_games, MQ_LEN,
                              replace=MQ_LEN > len(train_pool_games))
            with torch.no_grad():
                for _i in range(0, MQ_LEN, 256):
                    sub = torch.as_tensor(fill[_i:_i + 256], device=dev)
                    mqueue[_i:_i + len(sub)] = shadow(SGal[sub], mGal[sub]).float()
            mq_gid[:] = torch.as_tensor(fill, device=dev)
        as_step, as_pulls = 0, None
        if AS_M:
            assert not (SWIN or MQ_LEN or BCE), "async sim: plain gallery only"
            _v0 = torch.nn.utils.parameters_to_vector(
                model.parameters()).detach().clone()
            as_pulls = [_v0.clone() for _ in range(AS_M)]
            # resume: pulls re-init at current weights -> first M steps
            # transiently synchronous.
            print(f"DC-ASGD sim: {AS_M} workers, staleness {AS_M - 1}, "
                  f"lambda {DC_L}", flush=True)
        t0 = time.time()
        mv_steps = 0
        if args.measure_vram and dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for ep in range(start_ep, args.epochs):
            model.train()
            for _ in range(per_epoch // bs):
                gids = rng.choice(train_pool_games, bs, replace=False)
                tgt = pos_of_g_t[gids].to(dev)
                if AS_M:
                    # scheduled worker computes THIS step at the weights it
                    # pulled M-1 PS-steps ago (swap-in: gallery, views, doc
                    # tiers below all see the stale pull).
                    _ps_vec = torch.nn.utils.parameters_to_vector(
                        model.parameters()).detach().clone()
                    _wk = as_step % AS_M
                    torch.nn.utils.vector_to_parameters(
                        as_pulls[_wk], model.parameters())
                with torch.amp.autocast("cuda"):
                    Zg = (None if (MQ_LEN or SWIN or BCE or UNI)
                          else (gallery_train_ma(model) if MA_N
                                else gallery_train(model)))
                    if VFA:
                        # views-first anchors: draw the raw views ONCE, put
                        # them at the head of this step's teacher packs for
                        # the batch games, and swap those teachers into the
                        # gallery. Eval anchors stay the fixed packs.
                        vraw = [sample_views(gids, W, rng)
                                for _ in range(NV - 1)]
                        dparts = doc_view_raw(gids, W, rng)
                        S_vfa, m_vfa = vfa_packs(gids, vraw, dparts)
                        Zg = Zg.clone()
                        Zg[tgt] = model(S_vfa, m_vfa)
                    if SWIN:
                        # NOW anchors: fresh, grad, always in the field.
                        gids_t = torch.as_tensor(gids, device=dev)
                        eNow = model(SGal[gids_t], mGal[gids_t]).float()
                    model.slot_buf = [] if SC_W else None
                    if VFA:
                        Zs = [model(Sv, mv) for (Sv, mv) in vraw]
                        Zs.append(encode_doc_rows(model, dparts))
                    else:
                        Zs = [model(*sample_views(gids, W, rng))
                              for _ in range(NV - 1)]
                        Zs.append(assemble_doc_view(model, gids, W, rng, bs))
                    if MQ_LEN:
                        # current keys enter the ring head and ARE this
                        # step's positives; the rest of the ring = negatives;
                        # entity-ID mask kills stale same-game keys.
                        with torch.no_grad():
                            _r = torch.as_tensor(gids, device=dev)
                            keys = shadow(SGal[_r], mGal[_r]).float()
                        gid_t = torch.as_tensor(gids, device=dev)
                        slot = (torch.arange(bs, device=dev) + mq_ptr) % MQ_LEN
                        mqueue[slot] = keys
                        mq_gid[slot] = gid_t
                        mq_ptr = int((mq_ptr + bs) % MQ_LEN)
                        fmask = mq_gid[None, :] == gid_t[:, None]
                        fmask[torch.arange(bs, device=dev), slot] = False
                        loss = 0
                        for Z in Zs:
                            lg = Z.float() @ mqueue.T * _invt()
                            loss = loss + F.cross_entropy(
                                lg.masked_fill(fmask, -1e4), slot)
                    elif SWIN:
                        # CE lives in the window micro-passes at the
                        # backward stage; I chain still runs on the views.
                        loss = torch.zeros((), device=dev)
                    elif BCE:
                        # in-batch NT-Xent: logits vs ALL 4*bs views of this
                        # step; target = ring sibling; self + other siblings
                        # masked (false negatives). Ported verbatim from the
                        # fs worker.
                        keys = torch.cat([Z.float() for Z in Zs])
                        _ar = torch.arange(bs, device=dev)
                        loss = 0.0
                        for v in range(NV):
                            lg = Zs[v].float() @ keys.T * _invt()
                            for u in range(NV):
                                if u != (v + 1) % NV:
                                    lg[_ar, u * bs + _ar] = -1e4
                            loss = loss + F.cross_entropy(
                                lg, ((v + 1) % NV) * bs + _ar)
                    elif AU:
                        # I2CE with CE replaced by Wang&Isola align+uniformity
                        # on the anchor field (view-view I stays, IW block).
                        # align = pull each view to its OWN anchor (1-cos);
                        # uniform = Gaussian kernel exp(2t(cos-1)) vs the
                        # WRONG anchors (own column zeroed). Summed per view
                        # like the CE it replaces.
                        loss = 0.0
                        for Z in Zs:
                            sim = Z.float() @ Zg.T.float()
                            own = sim.gather(1, tgt[:, None]).squeeze(1)
                            ker = ((sim - 1.0) * (2 * AU_T)).exp().scatter(
                                1, tgt[:, None], 0.0)
                            loss = loss + (1 - own).mean() + ker.sum(1).div(
                                ker.shape[1] - 1).log().mean()
                    elif UNI:
                        # W&I uniformity (official-repo form): log mean
                        # exp(-t*pdist^2) over the batch views per branch;
                        # the align term is the IW block below (no anchor CE).
                        loss = 0.0
                        for Z in Zs:
                            d2 = torch.pdist(Z.float()).pow(2)
                            loss = loss + d2.mul(-UNI_T).exp().mean().log() / len(Zs)
                    elif tower_kind == "arc":
                        loss = sum(arcface_ce(Z.float() @ Zg.T.float(), tgt) for Z in Zs)
                    elif CE_GATED:
                        hd = torch.tensor(np.array([g in gate_games for g in gids])
                                          ).to(dev).nonzero(as_tuple=True)[0]
                        loss = (sum(F.cross_entropy(Z.float()[hd] @ Zg.T.float() * _invt(),
                                                    tgt[hd]) for Z in Zs)
                                if len(hd) else torch.zeros((), device=dev))
                    elif MA_N:
                        # MULTI-POSITIVE CE (SupCon L_out): every student view
                        # is classified against all MA_N*n_train packs; the
                        # game's own packs are ALL positives, scored once each
                        # and averaged. Dead packs are alive-weighted out of
                        # the numerator and -inf-masked out of the denominator.
                        _nt = len(train_pool_games)
                        _wg = aliveW[torch.as_tensor(gids, device=dev)]  # [bs,MA_N]
                        loss = 0.0
                        for Z in Zs:
                            lg = (Z.float() @ Zg.T.float() * _invt()
                                  ).masked_fill(ma_dead[None, :], -1e4)
                            _num = torch.zeros(len(gids), device=dev)
                            for k in range(MA_N):
                                _num = _num + _wg[:, k] * F.cross_entropy(
                                    lg, tgt + k * _nt, reduction="none")
                            loss = loss + (_num / _wg.sum(1).clamp(min=1e-6)).mean()
                    else:
                        loss = sum(F.cross_entropy(Z.float() @ Zg.T.float() * _invt(), tgt)
                                   for Z in Zs)
                    if IW > 0:
                        if I_GATED:
                            hd = torch.tensor(np.array([g in gate_games for g in gids],
                                                       dtype=np.float32)).to(dev)
                            loss = loss + IW * sum(
                                ((1 - (Zs[i].float() * Zs[j].float()).sum(-1)) * hd).sum()
                                / hd.sum().clamp(min=1) for i, j in pairs) / len(pairs)
                        else:
                            loss = loss + IW * sum(
                                (1 - (Zs[i].float() * Zs[j].float()).sum(-1)).mean()
                                for i, j in pairs) / len(pairs)
                    if SC_W and model.slot_buf:
                        # slot-C (user): orthogonality pressure on the 4
                        # slot vectors of every view forward -- mean squared
                        # off-diagonal of the unit-slot Gram matrix.
                        _sp = 0.0
                        for _sl in model.slot_buf:
                            _u = F.normalize(_sl.float(), dim=-1)
                            _g = _u @ _u.transpose(1, 2)
                            _n = _g.shape[1]
                            _off = _g - torch.eye(_n, device=_g.device)
                            _sp = _sp + (_off ** 2).sum((1, 2)).mean() \
                                / (_n * (_n - 1))
                        loss = loss + SC_W * _sp / len(model.slot_buf)
                        model.slot_buf = None   # NOT []: eval must not append
                opt.zero_grad()
                if SWIN:
                    # L window micro-passes: fresh-encode the next W ring
                    # games (grad), own CE partition [now | window] per
                    # view, IMMEDIATE backward (window activations freed
                    # between passes). Ring positions index train_pool_games.
                    _arb = torch.arange(bs, device=dev)
                    _ntr = len(train_pool_games)
                    if PFC:
                        # DISTRIBUTED SHARDED SOFTMAX (Partial-FC): K shards
                        # each sweep a swin window inside their own 1/K; the
                        # CE partition is the EXACT union of eNow + all K
                        # shard windows (concat == multi-node all_reduce).
                        # Each window encode is gradient-checkpointed.
                        _ews, _dups = [], []
                        for _k in range(PFC):
                            _sr = _pfc_rows[_k]
                            _wl = min(args.pfc_window or SWIN_W, len(_sr))
                            _wpos = _sr[(pfc_ptr[_k]
                                         + np.arange(_wl)) % len(_sr)]
                            _wt = torch.as_tensor(_tpg[_wpos], device=dev)
                            _dups.append(torch.isin(_wt, gids_t))
                            _ews.append(_tuc.checkpoint(
                                _pfc_enc, _wt, use_reentrant=False))
                            pfc_ptr[_k] = int((pfc_ptr[_k] + SWIN_S)
                                              % len(_sr))
                        with torch.amp.autocast("cuda"):
                            lw = 0.0
                            for Z in Zs:
                                _cols = [Z.float() @ eNow.T * _invt()]
                                for _e, _d in zip(_ews, _dups):
                                    _cols.append(
                                        (Z.float() @ _e.T * _invt())
                                        .masked_fill(_d[None, :], -1e4))
                                lw = lw + F.cross_entropy(
                                    torch.cat(_cols, 1), _arb)
                        amp.scale(lw).backward(retain_graph=True)
                    else:
                        for _l in range(SWIN_L):
                            _wpos = (swin_ptr + SWIN_S * _l
                                     + np.arange(SWIN_W)) % _ntr
                            _wt = torch.as_tensor(
                                np.asarray(train_pool_games)[_wpos], device=dev)
                            _dup = torch.isin(_wt, gids_t)
                            with torch.amp.autocast("cuda"):
                                eWin = model(SGal[_wt], mGal[_wt]).float()
                                lw = 0.0
                                for Z in Zs:
                                    lg = torch.cat(
                                        [Z.float() @ eNow.T * _invt(),
                                         (Z.float() @ eWin.T * _invt()
                                          ).masked_fill(_dup[None, :], -1e4)], 1)
                                    lw = lw + F.cross_entropy(lg, _arb)
                            amp.scale(lw).backward(retain_graph=True)
                        swin_ptr = int((swin_ptr + SWIN_L * SWIN_S) % _ntr)
                amp.scale(loss).backward()
                amp.unscale_(opt)
                if AS_M:
                    # DC-ASGD (user): compensate the stale gradient with the
                    # diagonal-Hessian Taylor term lambda*g*g*(W_t - W_pull),
                    # then apply at the CURRENT weights via the shared AdamW.
                    _g = torch.cat([
                        (q.grad if q.grad is not None else
                         torch.zeros_like(q)).reshape(-1)
                        for q in model.parameters()])
                    if DC_L:
                        _g = _g + DC_L * _g * _g * (_ps_vec - as_pulls[_wk])
                    torch.nn.utils.vector_to_parameters(
                        _ps_vec, model.parameters())
                    _o = 0
                    for q in model.parameters():
                        _n = q.numel()
                        if q.grad is not None:
                            q.grad.copy_(_g[_o:_o + _n].view_as(q))
                        _o += _n
                torch.nn.utils.clip_grad_norm_(params_all, 5.0)
                amp.step(opt)
                amp.update()
                if AS_M:
                    as_pulls[_wk] = torch.nn.utils.parameters_to_vector(
                        model.parameters()).detach().clone()
                    as_step += 1
                if MQ_LEN:
                    with torch.no_grad():
                        for _pk, _pq in zip(shadow.parameters(),
                                            model.parameters()):
                            _pk.data.mul_(MQ_M).add_(_pq.data, alpha=1 - MQ_M)
                if args.measure_vram:
                    mv_steps += 1
                    if mv_steps >= 3:
                        torch.cuda.synchronize()
                        peak = int(torch.cuda.max_memory_allocated())
                        Path(args.measure_vram).write_text(str(peak))
                        print(f"[measure-vram] cap={args.anchor_cap} "
                              f"peak={peak / 2**30:.2f}GiB", flush=True)
                        raise SystemExit(0)
            if (ep + 1) % args.ckpt_every == 0:
                sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if MQ_LEN:
                    torch.save(dict(model=sd,
                                    shadow={k: v.detach().cpu().clone()
                                            for k, v in shadow.state_dict().items()}),
                               OUT / f"ckpt_{name}_ep{ep+1}.pt")
                else:
                    torch.save(sd, OUT / f"ckpt_{name}_ep{ep+1}.pt")   # persist NOW
                bundle = dict(model=sd, opt=opt.state_dict(), amp=amp.state_dict(),
                              cpu_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state(),
                              np_rng=rng.bit_generator.state, ep=ep + 1)
                if SWIN:
                    bundle["swin_ptr"] = swin_ptr
                    if PFC:
                        bundle["pfc_ptr"] = pfc_ptr
                if TAU_LEARN:
                    bundle["log_invt"] = TAUP["p"].detach().cpu()
                    print(f"  tau={1.0 / float(torch.exp(TAUP['p'])):.4f}",
                          flush=True)
                if MQ_LEN:
                    bundle["shadow"] = {k: v.detach().cpu().clone()
                                        for k, v in shadow.state_dict().items()}
                    bundle["mqueue"] = mqueue.detach().cpu()
                    bundle["mq_gid"] = mq_gid.detach().cpu()
                    bundle["mq_ptr"] = mq_ptr
                tmp = RES.with_suffix(".tmp")
                torch.save(bundle, tmp)
                tmp.replace(RES)
            if ep % 100 == 99:
                print(f"  [ep{ep+1}] {time.time()-t0:.0f}s", flush=True)
        model.eval()
        return model


    def train_vicreg(seed=0, W=16, bs=192, per_epoch=3072):
        # Negative-free (no gallery CE); epd wiring = all three VICReg terms
        # on the expander OUTPUT pair (ported verbatim from the fs worker).
        # epdb: views drawn for EVERY train game per step (batch=all moments).
        epd = re.match(r"(epd[bg]?)_v(\d+)i(\d+)c(\d+)$", tower_kind)
        all_batch = bool(epd) and epd.group(1) == "epdb"
        vic_v, vic_i, vic_c = (float(x) for x in epd.groups()[1:])
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = SetPoolN(NSLOT).to(dev)
        expander = GameCentroidExpander(input_dim=DM).to(dev)
        opt = torch.optim.AdamW(
            list(model.parameters()) + list(expander.parameters()),
            lr=5e-4, weight_decay=1e-4)
        amp = amp_cls()
        RES = OUT / f"resume_{name}.pt"
        start_ep = 0
        if RES.exists():
            st = torch.load(RES, map_location="cpu")
            model.load_state_dict({k: v.to(dev) for k, v in st["model"].items()})
            expander.load_state_dict({k: v.to(dev)
                                      for k, v in st["expander"].items()})
            opt.load_state_dict(st["opt"])
            amp.load_state_dict(st["amp"])
            torch.set_rng_state(st["cpu_rng"])
            torch.cuda.set_rng_state(st["cuda_rng"])
            rng.bit_generator.state = st["np_rng"]
            start_ep = int(st["ep"])
            print(f"RESUME from ep{start_ep}", flush=True)
        t0 = time.time()
        mv_steps = 0
        if args.measure_vram and dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for ep in range(start_ep, args.epochs):
            model.train(); expander.train()
            for _ in range(per_epoch // bs):
                gids = (train_pool_games if all_batch else
                        rng.choice(train_pool_games, bs, replace=False))
                with torch.amp.autocast("cuda"):
                    Zs = [model(*sample_views(gids, W, rng))
                          for _ in range(NV - 1)]
                    Zs.append(assemble_doc_view(model, gids, W, rng, len(gids)))
                    Es = [expander(Z.float()) for Z in Zs]
                    loss = sum(vicreg_loss(
                        Es[i], Es[j], invariance_weight=vic_i,
                        variance_weight=vic_v,
                        covariance_weight=vic_c)["loss"]
                        for i, j in pairs) / len(pairs)
                opt.zero_grad()
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(expander.parameters()), 5.0)
                amp.step(opt)
                amp.update()
                if args.measure_vram:
                    mv_steps += 1
                    if mv_steps >= 3:
                        torch.cuda.synchronize()
                        peak = int(torch.cuda.max_memory_allocated())
                        Path(args.measure_vram).write_text(str(peak))
                        print(f"[measure-vram] cap={args.anchor_cap} "
                              f"peak={peak / 2**30:.2f}GiB", flush=True)
                        raise SystemExit(0)
            if (ep + 1) % args.ckpt_every == 0:
                sd = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
                xd = {k: v.detach().cpu().clone()
                      for k, v in expander.state_dict().items()}
                torch.save(dict(model=sd, expander=xd),
                           OUT / f"ckpt_{name}_ep{ep+1}.pt")
                bundle = dict(model=sd, expander=xd,
                              opt=opt.state_dict(), amp=amp.state_dict(),
                              cpu_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state(),
                              np_rng=rng.bit_generator.state, ep=ep + 1)
                tmp = RES.with_suffix(".tmp")
                torch.save(bundle, tmp)
                tmp.replace(RES)
            if ep % 100 == 99:
                print(f"  [vic ep{ep+1}] {time.time()-t0:.0f}s", flush=True)
        model.eval()
        return model

    def train_byol(seed=0, W=16, bs=192, per_epoch=3072):
        import copy
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        MV3 = tower_kind == "mv3"       # v3: in-batch CE keys, m=0.99
        mom = 0.99 if MV3 else 0.996
        _ar = torch.arange(bs, device=dev)
        model = SetPoolN(NSLOT).to(dev)
        pred = nn.Sequential(nn.Linear(DM, 256), nn.GELU(), nn.Linear(256, DM)).to(dev)
        target = copy.deepcopy(model).to(dev)
        for p in target.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(list(model.parameters()) + list(pred.parameters()),
                                lr=5e-4, weight_decay=1e-4)
        amp = amp_cls()
        P = lambda z: F.normalize(pred(z.float()), dim=-1)
        ckpts = {}
        RES = OUT / f"resume_{name}.pt"
        start_ep = 0
        if RES.exists():
            st = torch.load(RES, map_location="cpu")
            model.load_state_dict({k: v.to(dev) for k, v in st["model"].items()})
            pred.load_state_dict({k: v.to(dev) for k, v in st["pred"].items()})
            target.load_state_dict({k: v.to(dev) for k, v in st["target"].items()})
            opt.load_state_dict(st["opt"])
            amp.load_state_dict(st["amp"])
            torch.set_rng_state(st["cpu_rng"])
            torch.cuda.set_rng_state(st["cuda_rng"])
            rng.bit_generator.state = st["np_rng"]
            start_ep = int(st["ep"])
            print(f"RESUME from ep{start_ep}", flush=True)
        t0 = time.time()
        mv_steps = 0
        if args.measure_vram and dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for ep in range(start_ep, args.epochs):
            model.train(); pred.train()
            for _ in range(per_epoch // bs):
                gids = rng.choice(train_pool_games, bs, replace=False)
                with torch.amp.autocast("cuda"):
                    views = [sample_views(gids, W, rng) for _ in range(NV - 1)]
                    Zo = [model(S, m) for S, m in views]
                    with torch.no_grad():
                        Zt = [target(S, m) for S, m in views]
                    Zo.append(assemble_doc_view(model, gids, W, rng, bs))
                    with torch.no_grad():
                        Zlast_t = torch.empty(bs, DM, device=dev, dtype=torch.float16)
                        assigned = np.zeros(bs, bool)
                        for g2x, Sx, mx in tiers:
                            msk = np.array([(not a) and (g in g2x)
                                            for a, g in zip(assigned, gids)])
                            if msk.any():
                                rows = [g2x[g] for g in gids[msk]]
                                Zlast_t[torch.tensor(msk).to(dev)] = \
                                    target(Sx[rows], mx[rows]).half()
                                assigned |= msk
                        rest = ~assigned
                        if rest.any():
                            Zlast_t[torch.tensor(rest).to(dev)] = \
                                target(*sample_views(gids[rest], W, rng)).half()
                        Zt.append(Zlast_t)
                    loss, npairs = 0.0, 0
                    for i in range(NV):
                        for j in range(NV):
                            if i == j:
                                continue
                            if MV3:
                                # v3: symmetrized in-batch InfoNCE -- query
                                # pred(online view i) vs EMA keys of view j
                                # for the whole batch; positive = own column.
                                lg = P(Zo[i]) @ Zt[j].float().detach().T \
                                    * _invt()
                                loss = loss + F.cross_entropy(lg, _ar)
                            else:
                                loss = loss + (1 - (P(Zo[i]) * Zt[j].float().detach()).sum(-1)).mean()
                            npairs += 1
                    loss = loss / npairs
                opt.zero_grad()
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(pred.parameters()), 5.0)
                amp.step(opt)
                amp.update()
                with torch.no_grad():
                    for pt, po in zip(target.parameters(), model.parameters()):
                        pt.data.mul_(mom).add_(po.data, alpha=1 - mom)
                if args.measure_vram:
                    mv_steps += 1
                    if mv_steps >= 3:
                        torch.cuda.synchronize()
                        peak = int(torch.cuda.max_memory_allocated())
                        Path(args.measure_vram).write_text(str(peak))
                        print(f"[measure-vram] cap={args.anchor_cap} "
                              f"peak={peak / 2**30:.2f}GiB", flush=True)
                        raise SystemExit(0)
            if (ep + 1) % args.ckpt_every == 0:
                sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                pd = {k: v.detach().cpu().clone() for k, v in pred.state_dict().items()}
                td = {k: v.detach().cpu().clone() for k, v in target.state_dict().items()}
                # archive predictor + EMA target PER CHECKPOINT (user decree)
                torch.save(dict(model=sd, pred=pd, target=td),
                           OUT / f"ckpt_{name}_ep{ep+1}.pt")
                bundle = dict(model=sd,
                              pred=pd,
                              target=td,
                              opt=opt.state_dict(), amp=amp.state_dict(),
                              cpu_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state(),
                              np_rng=rng.bit_generator.state, ep=ep + 1)
                tmp = RES.with_suffix(".tmp")
                torch.save(bundle, tmp)
                tmp.replace(RES)
            if ep % 100 == 99:
                print(f"  [byol ep{ep+1}] {time.time()-t0:.0f}s", flush=True)
        model.eval()
        return model

    # ---------------- eval machinery ----------------
    VORDER = ["neutral", "noname", "positive", "negative"]

    def zs_from_arrays(Zg, Za, Zq):
        # ZS-only CV metrics + selection (user 2026-07-23): cvsel = noname_h1
        # + noname_h5 + 2*val_tag, all on the VAL fold, where val_tag is the
        # CLEAN-inductive tag (probe on the train pool, scored on val_g
        # neutral) -- the old 2*noname_tagF1 used the leaky tag_split probe.
        # Also all-4 test variants (h1+h5) and test_tag for the readout.
        # multi-anchor: Zg is [K, NG, D]. Retrieval scores a game by the
        # MAX over its packs (a query matching any facet counts); the tag
        # probe reads the concatenation (the deployed representation is all
        # K vectors, so the probe is entitled to all of them).
        _MA = (Zg.ndim == 3)
        if _MA:
            gz = Zg / (np.linalg.norm(Zg, axis=2, keepdims=True) + 1e-8)
            # RETRIEVAL takes the per-game max over packs (above). The TAG
            # PROBE instead reads a MEAN-POOLED 128-d vector: the same
            # structure and dimensionality every other arm feeds its probe,
            # so the tag number is directly comparable to BYOL / VICReg /
            # SimCLR. Averaging is wrong for retrieval (it is exactly what
            # this arm was built to remove) but right here -- the probe must
            # not collect a dimensionality bonus. A concat read is kept
            # alongside as a diagnostic only.
            _mp = Zg.mean(0)
            Zg_probe = _mp / (np.linalg.norm(_mp, axis=1, keepdims=True) + 1e-8)
        else:
            gz = Zg / (np.linalg.norm(Zg, axis=1, keepdims=True) + 1e-8)
            Zg_probe = Zg
        az = Za / (np.linalg.norm(Za, axis=1, keepdims=True) + 1e-8)
        out = {}

        def _sim(a):
            if _MA:
                return np.stack([a @ gz[k].T for k in range(gz.shape[0])]).max(0)
            return a @ gz.T

        def _rk(idx):
            sim = _sim(az[idx])
            tgt = A["gidx"][idx]
            return (sim > sim[np.arange(len(idx)), tgt][:, None]).sum(1) + 1

        for var in VORDER:
            ii = [i for i, g in enumerate(art_games)
                  if g in test_g and variants[i] == var]
            rk = _rk(ii)
            out["nm_" + var] = float((rk == 1).mean())
            out["h5_" + var] = float((rk <= 5).mean())
        rkv = _rk(va_neu)
        out["v_neu"] = float((rkv == 1).mean())
        rkv = _rk(va_non)
        out["v_non"] = float((rkv == 1).mean())
        out["v_non5"] = float((rkv <= 5).mean())
        sc, rg, al, th, _ = train_anchor_ridge(targs, Zg_probe, y, n2i, tag_split)
        for var in ("neutral", "noname"):
            idx = [i for i in range(len(art_games))
                   if variants[i] == var and art_games[i] in test_g]
            s = rg.predict(sc.transform(np.stack([Za[i] for i in idx]).astype(np.float32)))
            labs = np.stack([y[n2i[art_games[i]]] for i in idx])
            out["tag_" + var] = micro_prf(labs, s, th)["micro_f1"]
        # CLEAN INDUCTIVE TAG (user 2026-07-23): the default tag probe fits
        # on tag_split (70% of ALL games -> it sees most held-out anchors).
        # test_tag refits the ridge on the RETRIEVAL train pool only, picks
        # the threshold on val_g, and scores the test games' NEUTRAL queries
        # -- a fully held-out tag-generalization number matching the
        # retrieval eval's rigor. noname is unnecessary: the tower never saw
        # these games, so a name in the neutral text cannot be memorized.
        _ctag = {"train": [names[i] for i in train_pool_games],
                 "val": sorted(val_g), "test": sorted(test_g)}
        _sc2, _rg2, _, _th2, _ = train_anchor_ridge(targs, Zg_probe, y, n2i, _ctag)
        _ti = [i for i in range(len(art_games))
               if variants[i] == "neutral" and art_games[i] in test_g]
        _s2 = _rg2.predict(_sc2.transform(
            np.stack([Za[i] for i in _ti]).astype(np.float32)))
        _lab2 = np.stack([y[n2i[art_games[i]]] for i in _ti])
        out["test_tag"] = micro_prf(_lab2, _s2, _th2)["micro_f1"]
        if _MA:
            # A concat probe would need a 2*D query side that does not exist
            # (queries are single 128-d views), so it is not computed. The
            # useful diagnostic is whether the packs stayed DIFFERENT: if
            # pack_cos approaches 1 the arm has degenerated to single-anchor
            # and any comparison against I-CE is meaningless.
            _p = Zg / (np.linalg.norm(Zg, axis=2, keepdims=True) + 1e-8)
            _cs = [float((_p[a] * _p[b]).sum(1).mean())
                   for a in range(_p.shape[0]) for b in range(a + 1, _p.shape[0])]
            out["pack_cos"] = float(np.mean(_cs))
        # val-side twin of test_tag (same clean probe, scored on val_g
        # neutral) -- the LEAK-FREE selection tag; replaces v_non_tag, whose
        # tag_split probe had seen most held-out anchors (user 2026-07-23).
        _vi = [i for i in range(len(art_games))
               if variants[i] == "neutral" and art_games[i] in val_g]
        _vs = _rg2.predict(_sc2.transform(
            np.stack([Za[i] for i in _vi]).astype(np.float32)))
        _vlab = np.stack([y[n2i[art_games[i]]] for i in _vi])
        out["val_tag"] = micro_prf(_vlab, _vs, _th2)["micro_f1"]
        # VAL noname tag F1 (same ridge, predicted on val noname queries)
        s = rg.predict(sc.transform(np.stack([Za[i] for i in va_non]).astype(np.float32)))
        labs = np.stack([y[n2i[art_games[i]]] for i in va_non])
        out["v_non_tag"] = micro_prf(labs, s, th)["micro_f1"]
        out["cvsel"] = out["v_non"] + out["v_non5"] + 2.0 * out["val_tag"]
        out["cvsel_leaky"] = (out["v_non"] + out["v_non5"]
                              + 2.0 * out["v_non_tag"])   # old, report-only
        # --- REVIEW-based selection (user 2026-07-18): deployment has NO
        # rewrites, so the checkpoint pick uses the val fold's REVIEW
        # pseudo-queries (ss_queries): rvsel = q@1 + q@RSEL_K + 2*q_tagF1.
        # (User once wrote '@2'; the canonical formula was @5 -> RSEL_K=5,
        # single-constant change if @2 was intended.) Rewrite metrics stay
        # as REPORT-ONLY columns.
        RSEL_K = 5
        qz = Zq / (np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-8)
        qg = np.asarray(Qs["gidx"])
        for pref, gset in (("v", val_g), ("t", test_g)):
            qi = [i for i in range(len(qg)) if names[qg[i]] in gset]
            sim = _sim(qz[qi])
            tgtq = qg[qi]
            rkq = (sim > sim[np.arange(len(qi)), tgtq][:, None]).sum(1) + 1
            out[pref + "_q1"] = float((rkq == 1).mean())
            out[pref + "_q5"] = float((rkq <= RSEL_K).mean())
            sq = rg.predict(sc.transform(qz[qi].astype(np.float32)))
            labq = np.stack([y[qg[i]] for i in qi])
            out[pref + "_qtag"] = micro_prf(labq, sq, th)["micro_f1"]
        out["rvsel"] = out["v_q1"] + out["v_q5"] + 2.0 * out["v_qtag"]
        return out

    def zs_metrics(model):
        with torch.no_grad():
            Zg = ((gallery_ma(model) if MA_N else gallery(model))
                  .float().cpu().numpy())
            Za = torch.cat([model(SA[i:i+256], mA[i:i+256])
                            for i in range(0, SA.shape[0], 256)]).float().cpu().numpy()
        return zs_from_arrays(Zg, Za)

    def metrics4(gal, art):
        gz = gal / (np.linalg.norm(gal, axis=1, keepdims=True) + 1e-8)
        az = art / (np.linalg.norm(art, axis=1, keepdims=True) + 1e-8)
        out = {}
        sc, rg, al, th, _ = train_anchor_ridge(targs, gal, y, n2i, tag_split)
        for var in VORDER:
            ii = [i for i, g in enumerate(art_games) if g in test_g and variants[i] == var]
            sim = az[ii] @ gz.T
            tgt = A["gidx"][ii]
            rk = (sim > sim[np.arange(len(ii)), tgt][:, None]).sum(1) + 1
            s = rg.predict(sc.transform(np.stack([art[i] for i in ii]).astype(np.float32)))
            labs = np.stack([y[n2i[art_games[i]]] for i in ii])
            out[var] = dict(h1=float((rk == 1).mean()), h5=float((rk <= 5).mean()),
                            med=float(np.median(rk)),
                            tag=micro_prf(labs, s, th)["micro_f1"])
        # clean-inductive tag on the frozen (or head) embeddings: probe on
        # the train pool only, threshold on val_g, score test_g neutral.
        _ctag = {"train": [names[i] for i in train_pool_games],
                 "val": sorted(val_g), "test": sorted(test_g)}
        _sc2, _rg2, _, _th2, _ = train_anchor_ridge(targs, gal, y, n2i, _ctag)
        _ti = [i for i, g in enumerate(art_games)
               if g in test_g and variants[i] == "neutral"]
        _s2 = _rg2.predict(_sc2.transform(
            np.stack([art[i] for i in _ti]).astype(np.float32)))
        _lab2 = np.stack([y[n2i[art_games[i]]] for i in _ti])
        out["test_tag"] = micro_prf(_lab2, _s2, _th2)["micro_f1"]
        return out

    d_rows = [g2wiki[g] for g in sorted(g2wiki)]
    d_gidx = np.array(sorted(g2wiki), np.int64)

    def project_cache(model, path):
        NQ = len(Qs["gidx"])
        with torch.no_grad():
            SPg = ((gallery_ma(model) if MA_N else gallery(model))
                   .float().cpu().numpy())
            SPg_nd = gallery_nodoc(model).float().cpu().numpy()
            SPa = torch.cat([model(SA[i:i+256], mA[i:i+256])
                             for i in range(0, SA.shape[0], 256)]).float().cpu().numpy()
            SPq = torch.cat([model(*pad_flat(Qs["off"], range(i, min(i+64, NQ)))).float()
                             for i in range(0, NQ, 64)]).cpu().numpy()
            SPd = torch.cat([model(SW[d_rows[i:i+256]], mW[d_rows[i:i+256]])
                             for i in range(0, len(d_rows), 256)]).float().cpu().numpy()
        np.savez(path, SPg=SPg, SPg_nd=SPg_nd, SPa=SPa, SPq=SPq,
                 SPd=SPd, SPd_gidx=d_gidx)

    def train_userft_mrr(Xg, Xg_nd, Xa, Xq, Xd, d_pos, seed, ep2=600,
                         p1="ce", p2=None, ls=0.0, iw=1.0):
        p2 = p2 or FT
        ARC_S, ARC_M = 30.0, 0.2
        torch.manual_seed(seed)
        np.random.seed(seed)
        o_g = torch.tensor(np.random.randint(0, 3, Xg.shape[0]), device=dev)
        gstep = 0

        def gated_gallery():
            nonlocal gstep
            use_doc = ((gstep + o_g) % 3) < 2
            gstep += 1
            return torch.where(use_doc[:, None], Xg, Xg_nd)

        head = nn.Linear(Xg.shape[1], 128).to(dev)
        fwd = lambda x: F.normalize(head(x), dim=-1)
        logt = nn.Parameter(torch.tensor(np.log(1/0.07), dtype=torch.float32, device=dev))
        opt = torch.optim.AdamW(list(head.parameters()) + [logt], lr=1e-3,
                                weight_decay=1e-4)

        def vsel():
            with torch.no_grad():
                Zg = fwd(Xg)
                sim = fwd(Xa[va_neu]) @ Zg.T
                rk = (sim > sim.gather(1, gA_t[va_neu][:, None])).sum(1) + 1
                h_neu = float((rk == 1).float().mean())
                sim = fwd(Xa[va_non]) @ Zg.T
                rk = (sim > sim.gather(1, gA_t[va_non][:, None])).sum(1) + 1
                h_non = float((rk == 1).float().mean())
                h_non5 = float((rk <= 5).float().mean())
            sc = max(S_fn(h_non, 0.45), S_fn(h_non5, 0.65)) + S_fn(h_neu, 0.85)
            return sc, h_neu, h_non, h_non5

        best, pat = -float("inf"), 0
        bs_ = {k2: v.detach().clone() for k2, v in head.state_dict().items()}
        for ep in range(80):
            head.train()
            order = np.random.choice(q_train, 6144, replace=True)
            for k in range(0, 6144, 256):
                b = torch.tensor(order[k:k+256]).to(dev)
                Zg = fwd(gated_gallery()[tp_t])
                if p1 == "by":
                    loss = (1 - (fwd(Xq[b]) * Zg[qpos_t[b]].detach()).sum(-1)).mean()
                elif p1 == "ice":
                    zq = fwd(Xq[b])
                    loss = F.cross_entropy(zq @ Zg.T * logt.exp(), qpos_t[b],
                                           label_smoothing=ls)
                    loss = loss + iw * (1 - (zq * Zg[qpos_t[b]]).sum(-1)).mean()
                else:
                    loss = F.cross_entropy(fwd(Xq[b]) @ Zg.T * logt.exp(), qpos_t[b],
                                           label_smoothing=ls)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                opt.step()
                with torch.no_grad():
                    logt.clamp_(max=float(np.log(100.0)))
            if ep % 5 == 4:
                m = vsel()[0]
                if m > best:
                    best, pat = m, 0
                    bs_ = {k2: v.detach().clone() for k2, v in head.state_dict().items()}
                else:
                    pat += 1
                if pat >= 8:
                    break
        head.load_state_dict(bs_)

        opt = torch.optim.AdamW(list(head.parameters()) + [logt], lr=1e-4,
                                weight_decay=1e-4)
        bsel = vsel()
        best, pat = bsel[0], 0
        bs_ = {k2: v.detach().clone() for k2, v in head.state_dict().items()}
        nd = Xd.shape[0]
        for ep in range(ep2):
            head.train()
            order = np.random.permutation(nd)
            for k in range(0, nd, 128):
                b = torch.tensor(order[k:k+128]).to(dev)
                z = fwd(Xd[b])
                Zg = fwd(gated_gallery()[tp_t])
                tgt = d_pos[b]
                if p2 == "by":
                    loss = (1 - (z * Zg[tgt].detach()).sum(-1)).mean()
                elif p2 == "arc":
                    la = z @ Zg.T
                    lt = z @ z.T - torch.eye(z.shape[0], device=dev) * 1e4
                    logits = torch.cat([la, lt], 1)
                    cos_t = logits.gather(1, tgt[:, None]).clamp(-1 + 1e-7, 1 - 1e-7)
                    phi = torch.cos(torch.acos(cos_t) + ARC_M)
                    phi = torch.where(cos_t > math.cos(math.pi - ARC_M),
                                      phi, cos_t - ARC_M * math.sin(ARC_M))
                    logits = logits.scatter(1, tgt[:, None], phi)
                    loss = F.cross_entropy(logits * ARC_S, tgt)
                else:
                    la = z @ Zg.T
                    lt = z @ z.T - torch.eye(z.shape[0], device=dev) * 1e4
                    loss = F.cross_entropy(torch.cat([la, lt], 1) * logt.exp(), tgt,
                                           label_smoothing=ls)
                    if p2 == "ice":
                        loss = loss + iw * (1 - (z * Zg[tgt]).sum(-1)).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                opt.step()
                with torch.no_grad():
                    logt.clamp_(max=float(np.log(100.0)))
            if ep % 5 == 4:
                cur = vsel()
                if cur[0] > best:
                    best, pat, bsel = cur[0], 0, cur
                    bs_ = {k2: v.detach().clone() for k2, v in head.state_dict().items()}
                else:
                    pat += 1
                if pat >= 24:
                    break
        head.load_state_dict(bs_)
        head.eval()
        with torch.no_grad():
            return (fwd(Xg).cpu().numpy(), fwd(Xa).cpu().numpy(),
                    {"vscore": float(bsel[0]), "v_neu": float(bsel[1]),
                     "v_non": float(bsel[2]), "v_non5": float(bsel[3])})

    # ---------------- fold frozen-embedder baseline (once per fold) ----------
    fb = OUT / f"w9cv_frozen_fold{args.fold}.json"
    if not fb.exists():
        with torch.no_grad():
            # CHUNKED: SGal.float() on the whole [NG, cap, 1024] tensor is
            # 31.6 GiB in one allocation at cap 4096 and OOMs an 80G card
            # that already holds the fp16 gallery.
            Zg0 = np.zeros((NG, SGal.shape[-1]), np.float32)
            for _i in range(0, NG, 64):
                _s = SGal[_i:_i + 64].float()
                _w = (~mGal[_i:_i + 64]).float().unsqueeze(-1)
                Zg0[_i:_i + 64] = ((_s * _w).sum(1)
                                   / _w.sum(1).clamp(min=1)).cpu().numpy()
                del _s, _w
            Za0 = np.zeros((SA.shape[0], SA.shape[-1]), np.float32)
            for _i in range(0, SA.shape[0], 256):
                _s = SA[_i:_i + 256].float()
                _w = (~mA[_i:_i + 256]).float().unsqueeze(-1)
                Za0[_i:_i + 256] = ((_s * _w).sum(1)
                                    / _w.sum(1).clamp(min=1)).cpu().numpy()
                del _s, _w
            torch.cuda.empty_cache()
        json.dump(metrics4(Zg0, Za0), open(fb, "w"), indent=2)
        print(f"frozen baseline written: {fb.name}", flush=True)

    # ---------------- tower + checkpoints ----------------
    DONE_FLAG = OUT / f"tower_{name}_ep{args.epochs}.npz"
    if not DONE_FLAG.exists() and not args.eval_only:
        t0 = time.time()
        # PER-FOLD TRAIN SEED (user design 2026-07-17): seed = fold index.
        # The 5 folds then jointly sample split variance AND seed variance,
        # so the across-fold std is the TOTAL error bar; within a fold both
        # recipes share the seed (and the split), keeping ce-vs-i2ce PAIRED.
        # Fold identity is untouched (--cv-seed drives the permutation).
        if tower_kind in ("byol", "mv3"):
            train_byol(seed=args.fold, W=args.view_w)
        elif tower_kind.startswith("epd"):
            train_vicreg(seed=args.fold, W=args.view_w)
        else:
            train_v4doc(seed=args.fold, W=args.view_w)
        print(f"tower {name} train phase done in {time.time()-t0:.0f}s", flush=True)
        traj_p = OUT / f"zs_traj_{name}.json"
        zs_traj = json.loads(traj_p.read_text()) if traj_p.exists() else {}
        cks = sorted(OUT.glob(f"ckpt_{name}_ep*.pt"),
                     key=lambda q: int(q.stem.split("_ep")[-1]))
        for ck in cks:
            ek = int(ck.stem.split("_ep")[-1])
            npz = OUT / f"tower_{name}_ep{ek}.npz"
            if npz.exists():
                continue                                # projection-level resume
            st = torch.load(ck, map_location="cpu")
            sd = st["model"] if "model" in st else st     # byol ckpts are nested
            m2 = SetPoolN(NSLOT).to(dev)
            m2.load_state_dict({k: v.to(dev) for k, v in sd.items()})
            m2.eval()
            project_cache(m2, npz)          # SPq/SPg/SPa saved once ...
            T0 = np.load(npz)
            zk = zs_from_arrays(T0["SPg"], T0["SPa"], T0["SPq"])
            zs_traj[f"ep{ek}"] = zk
            print(f"ZS(ep{ek}): {dict((k, round(v, 3)) for k, v in zk.items())}",
                  flush=True)
            json.dump(zs_traj, open(traj_p, "w"), indent=2)
            del m2
        (OUT / f"resume_{name}.pt").unlink(missing_ok=True)

    # ---------------- ZS-primary: refresh traj + select zsbest by cvsel ----
    # cvsel = noname_h1 + noname_h5 + 2*val_tag (val fold), val_tag = clean-
    # inductive tag (train-pool probe, val_g neutral). Old traj entries
    # refresh from the projection npz (no GPU) to gain test_tag + val_tag.
    traj_p = OUT / f"zs_traj_{name}.json"
    zs_traj = json.loads(traj_p.read_text()) if traj_p.exists() else {}
    for npzp in sorted(OUT.glob(f"tower_{name}_ep*.npz"),
                       key=lambda q: int(q.stem.split("_ep")[-1])):
        ek = npzp.stem.split("_ep")[-1]
        if "val_tag" in zs_traj.get(f"ep{ek}", {}):
            continue   # gate on val_tag (the field ONLY the clean-selection
        # code writes): entries scored by an earlier version carry test_tag +
        # the OLD leaky cvsel but no val_tag, so they MUST be recomputed to
        # gain val_tag + the clean cvsel -- keying on test_tag would skip them
        # and silently re-select on the leaky metric.
        T0 = np.load(npzp)
        zs_traj[f"ep{ek}"] = zs_from_arrays(T0["SPg"], T0["SPa"], T0["SPq"])
        print(f"ZS-refresh(ep{ek}) cvsel={zs_traj[f'ep{ek}']['cvsel']:.3f}",
              flush=True)
        json.dump(zs_traj, open(traj_p, "w"), indent=2)
    cand = {k: v for k, v in zs_traj.items() if "val_tag" in v}   # only
    # entries with the CLEAN cvsel (val_tag present) are selection candidates
    if cand:
        bk = max(cand, key=lambda k: (cand[k]["cvsel"], -int(k[2:])))
        json.dump(dict(best_ep=int(bk[2:]), **cand[bk]),
                  open(OUT / f"zsbest_{name}.json", "w"), indent=2)
        b = cand[bk]
        print(f"ZSBEST {name}: ep{bk[2:]} cvsel={b['cvsel']:.3f} "
              f"test_tag={b['test_tag']:.3f} "
              f"(q1 {b['v_q1']:.3f} q5 {b['v_q5']:.3f} qtag {b['v_qtag']:.3f}) "
              f"non={b['nm_noname']:.3f}/{b['h5_noname']:.3f} "
              f"tag={b['tag_noname']:.3f}", flush=True)

    # ---------------- heads (LEGACY, --head only) ----------------------------
    if args.head:
        if tower_kind == "byol":
            HEAD_CFGS = [("", "by", "by", 0.0), ("_ce", "ce", "ce", 0.0),
                         ("_cetf", "ce", "ce", 0.1)]
        elif FT == "ice":
            HEAD_CFGS = [("", "ice", "ice", 0.0), ("_p1ce", "ce", "ice", 0.0)] \
                if tower_kind in ("ice", "i2ce") else [("", "ice", "ice", 0.0)]
        else:
            HEAD_CFGS = [("", "ce", FT, 0.0)]

        VORD = VORDER

        def head_runs(path, seeds, p1, p2, ls, tag_):
            T = np.load(path)
            g0 = T["SPg"]
            d_pos = pos_of_g_t[T["SPd_gidx"]].to(dev)
            mu, sd = g0.mean(0, keepdims=True), g0.std(0, keepdims=True) + 1e-6
            tt = lambda x: torch.tensor((x - mu) / sd, dtype=torch.float32).to(dev)
            Xg, Xa, Xq, Xd = tt(g0), tt(T["SPa"]), tt(T["SPq"]), tt(T["SPd"])
            Xg_nd = tt(T["SPg_nd"])
            runs = []
            for seed in seeds:
                gal, art, vs = train_userft_mrr(Xg, Xg_nd, Xa, Xq, Xd, d_pos, seed,
                                                p1=p1, p2=p2, ls=ls, iw=HIW)
                m = metrics4(gal, art)
                m.update(vs)
                runs.append(m)
                print(f"{tag_} seed{seed}: neu {m['neutral']['h1']:.3f} "
                      f"non {m['noname']['h1']:.3f} vsel {vs['vscore']:.3f}", flush=True)
            return runs

        def agg_print(tag_, runs):
            for var in VORD:
                h1m = np.mean([r[var]["h1"] for r in runs])
                h5m = np.mean([r[var]["h5"] for r in runs])
                tm = np.mean([r[var]["tag"] for r in runs])
                print(f"AGG {tag_} {var:9s} h1={h1m:.3f} h5={h5m:.3f} tag={tm:.3f}",
                      flush=True)
            m4 = np.mean([np.mean([r[v]["h1"] for r in runs]) for v in VORD])
            vm = np.mean([r["vscore"] for r in runs])
            print(f"AGG {tag_} mean-of-4 = {m4:.3f} | val-score = {vm:.3f}", flush=True)

        ck_paths = sorted(OUT.glob(f"tower_{name}_ep*.npz"),
                          key=lambda p: int(p.stem.split("_ep")[-1]))
        for hsuf, p1, p2, ls in HEAD_CFGS:
            for path in ck_paths:
                ek = path.stem.split("_ep")[-1]
                tag_ = f"{name}_ep{ek}{hsuf}"
                outj = OUT / f"ft4var_{tag_}.json"
                if outj.exists():
                    continue
                runs = head_runs(path, range(args.ckpt_seeds), p1, p2, ls, tag_)
                agg_print(tag_, runs)
                json.dump({"per_seed": runs}, open(outj, "w"), indent=2)
            outb = OUT / f"ft4var_{name}_best{hsuf}.json"
            if outb.exists():
                continue
            vms = {}
            for path in ck_paths:
                ek = path.stem.split("_ep")[-1]
                rs = json.loads((OUT / f"ft4var_{name}_ep{ek}{hsuf}.json").read_text())["per_seed"]
                vms[ek] = float(np.mean([r["vscore"] for r in rs]))
            bek = max(vms, key=vms.get)
            bpath = OUT / f"tower_{name}_ep{bek}.npz"
            tag_ = f"{name}_best{hsuf}(ep{bek})"
            print(f"POST-HOC pick {name}{hsuf}: ep{bek} (val-score {vms[bek]:.3f})",
                  flush=True)
            prev = json.loads((OUT / f"ft4var_{name}_ep{bek}{hsuf}.json").read_text())["per_seed"]
            runs = prev + head_runs(bpath, range(args.ckpt_seeds, args.topup_seeds),
                                    p1, p2, ls, tag_)
            agg_print(tag_, runs)
            json.dump({"best_ep": int(bek), "val_score_by_ep": vms, "per_seed": runs},
                      open(outb, "w"), indent=2)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
