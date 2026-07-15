# -*- coding: utf-8 -*-
"""R60 wcle-protocol worker for A100 pods: one ARM = tower (1000ep, ckpt/50)
+ per-checkpoint heads (3 seeds) + post-hoc vsel pick topped to 10 seeds.

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

ARMS = {
    "wcle_ice_icetf": ("ice", "ice"),
    "wcle_i2ce_icetf": ("i2ce", "ice"),
    "wcle_ce_cetf": ("ce", "ce"),
    "wcle_byol_bytf": ("byol", "by"),
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
    "wcle_vic_cetf": ("vic", "ce"),                # VICReg tower: negative-free
    # like BYOL, but V/C terms supply the anti-collapse force BYOL lacks
    "wcle_vic2_cetf": ("vic2", "ce"),              # C-dose ablation (C 20 -> 15)
    "wcle_byol2_bytf": ("byol2", "by"),            # BYOL + BN: BatchNorm1d in
    # projector head and a 3-layer BN predictor (implicit-contrast channel)
    "wcle_cegate2c_icetf": ("cegate2c", "ice"),    # champion recipe with
    # TRAIN-TIME centering: outputs get mu-EMA subtracted BEFORE the L2
    # normalize, so CE/I never spend capacity on a common direction
    "wcle_i2expce_icetf": ("i2expce", "ice"),
    # NAMING GRAMMAR (user decree): [I position][CE path]. "i2exp..." = I in
    # the DEPLOYED space (the ORIGINAL n4expce design); "expi2..." = I after
    # the expander (E-space, the NEW design).
    "wcle_ceexpi2_icetf": ("ceexpi2", "ice"),            # per-view CE@dep + I@E
    # (with poolceexpi2: the "expander serves ONLY I" pair; deployed-CE
    # terms lead the name, expi2 trails = I after the expander)
    # E-sharing grammar (user decree): "expi2expce"/"expi2poolexpce" carry TWO
    # exp tokens = I and CE use SEPARATE expanders (E_I, E_CE). The "shexp"
    # prefix = ONE SHARED E serves every E-space loss of the arm.
    "wcle_expi2expce_icetf": ("expi2expce", "ice"),      # I@E_I + CE@E_CE (dual)
    "wcle_shexpi2ce_icetf": ("shexpi2ce", "ice"),        # I@E + CE@E, SHARED E
    "wcle_shexpi2poolce_icetf": ("shexpi2poolce", "ice"),  # I@E + pool-AFTER-E
    # CE = CE(mean(E(views))) -- REWIRED per the pool-position grammar,
    # SHARED E
    "wcle_poolceexpi2_icetf": ("poolceexpi2", "ice"),    # pooled CE@dep + I@E
    # (the expander exists ONLY to give I its tax space)
    "wcle_expi2poolexpce_icetf": ("expi2poolexpce", "ice"),  # I@E + pool->E->CE
    "wcle_poolce_cetf": ("poolce", "ce"),          # pooled CE, NO I
    "wcle_poolexpce_cetf": ("poolexpce", "ce"),    # pool -> E -> CE, NO I
    # cmp = COMPRESSOR (SimCLR-direction gate, user callout): same lifecycle
    # as exp but DOWN-projecting (128->128->64) -- an information bottleneck
    # for the loss instead of head-room. exp arms stay up (128->256->512).
    "wcle_cmpce_cetf": ("cmpce", "ce"),            # no-I, per-view CE@cmp
    "wcle_i2cmpce_icetf": ("i2cmpce", "ice"),      # I@dep + per-view CE@cmp
    "wcle_shcmpi2ce_icetf": ("shcmpi2ce", "ice"),  # SHARED cmp: I@cmp + CE@cmp
    # direction-closure wave (user decree): every template gets its exp/cmp
    # variants; DUAL mixed directions allowed; "pool...E" order encodes pool
    # position (poolexpce = pool BEFORE E; shexpi2poolce = pool AFTER E).
    "wcle_i2poolexpce_icetf": ("i2poolexpce", "ice"),    # I@dep + pool->exp->CE
    "wcle_i2poolcmpce_icetf": ("i2poolcmpce", "ice"),    # I@dep + pool->cmp->CE
    "wcle_expi2cmpce_icetf": ("expi2cmpce", "ice"),      # DUAL: I@exp + CE@cmp
    "wcle_cmpi2expce_icetf": ("cmpi2expce", "ice"),      # DUAL: I@cmp + CE@exp
    "wcle_cmpi2cmpce_icetf": ("cmpi2cmpce", "ice"),      # DUAL: I@cmp + CE@cmp
    "wcle_expi2poolcmpce_icetf": ("expi2poolcmpce", "ice"),  # DUAL: I@exp +
    # pool->cmp->CE (poolexpceexpi2 would equal expi2poolexpce -- not duped)
    "wcle_shexpi2poolexpce_icetf": ("shexpi2poolexpce", "ice"),  # SHARED exp:
    # I@E + pool-BEFORE-E CE (the old shexpi2poolce wiring, renamed)
    "wcle_shcmpi2poolcmpce_icetf": ("shcmpi2poolcmpce", "ice"),  # SHARED cmp:
    # I@E + pool-BEFORE-E CE (user wrote shexpi2poolcmpce; shared module is
    # single-direction, cmp assumed)
    # symmetry-completion wave (user decree: full 30-cell grid)
    "wcle_poolcmpce_cetf": ("poolcmpce", "ce"),        # no-I, pool->cmp->CE
    "wcle_cecmpi2_icetf": ("cecmpi2", "ice"),          # per-view CE@dep + I@cmp
    "wcle_poolcecmpi2_icetf": ("poolcecmpi2", "ice"),  # pooled CE@dep + I@cmp
    "wcle_shcmpi2poolce_icetf": ("shcmpi2poolce", "ice"),  # SHARED cmp,
    # pool-AFTER-E (CE on mean of E-outputs) -- twin of shexpi2poolce
    "wcle_cmpi2poolexpce_icetf": ("cmpi2poolexpce", "ice"),  # DUAL: I@cmp +
    # pool->exp->CE
    "wcle_cmpi2poolcmpce_icetf": ("cmpi2poolcmpce", "ice"),  # DUAL: I@cmp +
    # pool->cmp->CE
    "wcle_expce_cetf": ("expce", "ce"),            # PURE expander CE (user
    # design): NO I anywhere -- the deployed space receives no direct loss,
    # all shaping arrives via backprop through E. SimCLR-orthodox member of
    # the attachment matrix; A/B = ce (deployed CE, no I) at the same cap.          # n4expce revival (w9 cert; I x2 kept):
    # CE paid in a DISPOSABLE expander space (E = norm(MLP 128->256->512),
    # CE on E(view) @ E(gallery)); I stays in the deployed space; deploy =
    # pre-expander tower output. A/B = i2ce at the same cap.
    "wcle_i2poolce_icetf": ("i2poolce", "ice"),        # CE on the normalized MEAN
    # of the 4 views (x4 weight keeps the effective 4:2 CE:I ratio); I
    # unchanged -- w9 re-cert of the old 6-direction pooled-CE verdict.
    "wcle_ai2lse_icetf": ("ai2lse", "ice"),        # ANCHOR-IN-I + LSE (user):
    # I pairs span {4 views + OWN ANCHOR} (10 edges) so alignment itself ties
    # anchor_g to its view cluster; repulsion = same negative-only LSE.
    # = CE fully decomposed W&I-style: symmetric constant-weight attraction
    # (incl. the positive edge) + DCL uniformity. (i2lse, the no-anchor-rope
    # variant, was purged pre-run: desert equilibria made it foreseeable.)
    # vs ai2ce isolates dropping CE's adaptive pull (LSE keeps the push).
    "wcle_ai2ce_icetf": ("ai2ce", "ice"),          # BOTH pulls (user, 2x2 corner):
    # full per-view CE (adaptive pull + push, tau .02) AND anchor-in-I
    # (constant rope, 10 edges). vs i2ce = the anchor edge's worth ON TOP
    # of CE; vs ai2lse = CE's adaptive pull's worth GIVEN the rope.
    # Completes {CE-pull x anchor-in-I}: i2ce(Y/N) ai2ce(Y/Y) ai2lse(N/Y),
    # i2lse(N/N) purged.
    "wcle_aali3uni_icetf": ("aali3uni", "ice"),        # W&I uniformity swap:
    # same anchor-in-I attraction as ai2lse, but repulsion = Wang&Isola
    # batch uniformity log mean exp(-t*d^2) (t=2) over the 192 batch views
    # per branch -- NOT the anchor field. On the sphere that kernel is
    # exp(2t*cos) => LSE over cos at tau=1/(2t)=0.25. vs aali3auni isolates
    # repulsion SOURCE alone (paper recipe fixed, anchors -> batch).
    "wcle_ali3uni_icetf": ("ali3uni", "ice"),          # PURE Wang&Isola cell:
    # 3*align (= IW 6 x mean(1-cos), alpha=2 identity) + 1*uniformity,
    # official-repo STL-10 flagship weights. NO anchor edge anywhere =
    # first anchor-free NEGATIVE-based arm (BYOL/VICReg are anchor-free
    # but negative-free). No desert trap: batch self-repulsion has no
    # fixed field to hide from, unlike the purged i2lse. Doc VIEW stays
    # (protocol constant NV=4), so anchor-free != doc-free.
    "wcle_aali3auni_icetf": ("aali3auni", "ice"),      # W&I BOTH-AT-ANCHOR (user):
    # paper recipe (3*align IW6 + 1*uniform t=2) with BOTH forces aimed at
    # the anchor field: pull = anchor-in-I, push = Gaussian kernel vs the
    # WRONG anchors. Gradient-identical to ai2lse's LSE at tau=1/(2t)=0.25
    # (log-mean vs log-sum = additive const), so the ladder factorizes:
    # ai2lse -> aali3auni = kernel softness + pull:push budget (source fixed);
    # aali3auni -> aali3uni = repulsion source only (recipe fixed).
    "wcle_aali2auni_icetf": ("aali2auni", "ice"),  # align_w 2 rung of aali3auni
    "wcle_ali2uni_icetf": ("ali2uni", "ice"),      # align_w 2 rung of ali3uni
    # digit grammar (user decree): number after aali/ali = PAPER align weight
    # (alpha=2 units); internal IW = 2x digit (3 -> IW6 = repo flagship 3:1,
    # 2 -> IW4). Our i2 family sits at IW2 == paper align_w 1.
    "wcle_i2cce_icetf": ("i2cce", "ice"),          # I2CCE: CE all + I x2 + C x1
    # (VICReg-style off-diag covariance penalty ON the 128-d outputs --
    # decorrelated dims = feature-richness constraint, no expander needed)
    "wcle_i2ccec_icetf": ("i2ccec", "ice"),        # I2CCE + train-time centering
    # epd_v{V}i{I}c{C}: CANONICAL VICReg (weights parsed from the codename).
    # All three terms on the expander OUTPUT pair (paper wiring) -- kills the
    # vic/vic2 loophole where centroid collapse zeroed the invariance MSE
    # while the expander's input LayerNorm faked V/C from noise.
    "wcle_epd_v25i25c1_cetf": ("epd_v25i25c1", "ce"),   # paper (image) weights
    "wcle_epd_v20i10c20_cetf": ("epd_v20i10c20", "ce"),  # OUR allocation
    "wcle_epd_v20i10c15_cetf": ("epd_v20i10c15", "ce"),  # OUR allocation, C backoff
    # epdb_*: TRUE batch=all -- epd wiring untouched (all three terms on the
    # view embeddings), but every step draws views for the ENTIRE train pool
    # (~1613 games) instead of 192; steps/epoch unchanged (pure batch-size
    # effect on the moment estimates).
    # epdg_*: GRID-HARMONIZED canonical VICReg (dedicated flash_byol_vicreg
    # notebook): epd wiring, but the expander is the 30-cell grid's exp module
    "wcle_epdg_v25i25c1_cetf": ("epdg_v25i25c1", "ce"),
    "wcle_epdg_v20i10c15_cetf": ("epdg_v20i10c15", "ce"),
    "wcle_epdb_v25i25c1_cetf": ("epdb_v25i25c1", "ce"),
    "wcle_epdb_v20i10c20_cetf": ("epdb_v20i10c20", "ce"),
    "wcle_epdb_v20i10c15_cetf": ("epdb_v20i10c15", "ce"),
    # ---- anchor MEMORY-BANK family (InstDisc/XBM-style; NOT MoCo -- online
    # encoder snapshots, no momentum encoder). Bank = one 128-d row per train
    # entity; each step only a small set of rows is re-encoded, the rest keep
    # their last snapshot. Per-step anchor cost = k*cap, DECOUPLED from N.
    #   bkq{k}i2cce: QUEUE rotation -- refresh the next k rows in cyclic
    #                order; every anchor (positives included) is at most
    #                ceil(N/k) steps stale; NO gradient through any anchor.
    #   bkbi2cce:    refresh = the CURRENT BATCH's 192 entities -- positive
    #                columns are always fresh AND carry gradient (as in the
    #                full recipe); only the cold columns are stale snapshots.
    "wcle_bkq192i2cce_icetf": ("bkq192i2cce", "ice"),
    "wcle_bkq48i2cce_icetf": ("bkq48i2cce", "ice"),
    "wcle_bkq12i2cce_icetf": ("bkq12i2cce", "ice"),
    "wcle_bkbi2cce_icetf": ("bkbi2cce", "ice"),
    # ---- true MoCo queue (user design): shadow tower (weight-EMA of the
    # main tower, m=0.99) encodes the current batch's anchors each step;
    # they enter at the queue HEAD and serve as this step's positives, the
    # rest of the FIFO ring (historical keys) are the negatives; the write
    # pointer shifts by bs per step, evicting the oldest. Entity-ID mask
    # kills same-game stale keys (small-N false-negative collisions).
    # Negatives coverage = last Q/bs steps' samples, NOT the full catalog --
    # that is the trade this arm measures. Gradient: query side only.
    "wcle_mq3072i2cce_icetf": ("mq3072i2cce", "ice"),
}
CENTER_ARMS = {"cegate2c", "i2ccec"}
_CW = {"i2cce": 1.0, "i2ccec": 1.0,
       "bkq192i2cce": 1.0, "bkq48i2cce": 1.0, "bkq12i2cce": 1.0,
       "bkbi2cce": 1.0, "mq3072i2cce": 1.0}           # covariance weight
_IW = {"ice": 1.0, "i2ce": 2.0, "cegate1": 1.0, "cegate2": 2.0, "cegate3": 3.0,
       "cegate4": 4.0, "cegate1w": 1.0, "cegate2w": 2.0, "igate1": 1.0,
       "igate1w": 1.0, "rgate2": 2.0, "nodoc": 2.0, "cegate2c": 2.0,
       "i2cce": 2.0, "i2ccec": 2.0, "ai2lse": 2.0, "ai2ce": 2.0,
       "aali3uni": 6.0, "ali3uni": 6.0, "aali3auni": 6.0, "aali2auni": 4.0, "ali2uni": 4.0,   # = 2x W&I align_w 3 (alpha=2 = 2*(1-cos)) "i2expce": 2.0, "i2poolce": 2.0,
       "ceexpi2": 2.0, "expi2expce": 2.0, "poolceexpi2": 2.0, "expi2poolexpce": 2.0,
       "shexpi2ce": 2.0, "shexpi2poolce": 2.0, "i2cmpce": 2.0, "shcmpi2ce": 2.0,
       "i2poolexpce": 2.0, "i2poolcmpce": 2.0, "expi2cmpce": 2.0,
       "cmpi2expce": 2.0, "cmpi2cmpce": 2.0, "expi2poolcmpce": 2.0,
       "shexpi2poolexpce": 2.0, "shcmpi2poolcmpce": 2.0,
       "cecmpi2": 2.0, "poolcecmpi2": 2.0, "shcmpi2poolce": 2.0,
       "cmpi2poolexpce": 2.0, "cmpi2poolcmpce": 2.0,
       "bkq192i2cce": 2.0, "bkq48i2cce": 2.0, "bkq12i2cce": 2.0,
       "bkbi2cce": 2.0, "mq3072i2cce": 2.0}
SPLIT_SEED = 20260711
DM, HEADS, NV = 128, 4, 4
ARC_S_T, ARC_M_T = 50.0, 0.2       # tower ArcFace
UNI_T = 2.0                        # W&I uniformity Gaussian t (repo default)
# VICReg tower weights (I, V, C) per arm. The paper's 25/25/1 assumes
# unnormalized high-dim features; on unit-norm 128-d game centroids the budget
# shifts away from invariance toward spread/decorrelation (user-set).
# vic2 backs C off to 15 in case C=20 over-decorrelates.
VIC_W = {"vic": (10.0, 20.0, 20.0), "vic2": (10.0, 20.0, 15.0)}
K1, K2, S_A, S_B = 1.0, 1.0, 10.0, 1.0   # vsel piecewise score


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--anchor-cap", type=int, default=512)
    ap.add_argument("--no-sp-view", action="store_true",
                    help="doc tier = wiki_clean ONLY (drop the sp_raw fallback)")
    ap.add_argument("--wiki-src", choices=("clean", "llm"), default="clean",
                    help="wiki doc source: clean=raw wiki text; llm=sentence-wise "
                         "paraphrase (pretraining-leak ablation)")
    ap.add_argument("--doc-lead", type=int, default=0,
                    help=">0: truncate doc VIEWS to the first N sentences "
                         "(length-attribution ablation)")
    ap.add_argument("--full-pool-path", default="",
                    help="path of full_pool_fp16.npy (meta npz expected beside "
                         "it); empty = <data-dir>/full_pool_fp16.npy")
    ap.add_argument("--full-pool", action="store_true",
                    help="draw training views from the FULL review corpus "
                         "(host-RAM flat npy) instead of the 2048-sent pool")
    ap.add_argument("--view-w", type=int, default=16,
                    help="sentence budget per training review view (whole "
                         "reviews accumulated until >= W); 16 = historical")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--ckpt-seeds", type=int, default=3)
    ap.add_argument("--topup-seeds", type=int, default=10)
    ap.add_argument("--claim-file", default="",
                    help="claim file on the shared volume; when set, a 30s "
                         "heartbeat thread keeps it fresh (silent >2 min = "
                         "host presumed dead, job claimable by other hosts)")
    return ap.parse_args()


class SetPoolN(nn.Module):
    def __init__(s, N, bn=False, center=False):
        super().__init__()
        s.q0 = nn.Parameter(torch.randn(1, N, DM) * 0.02)
        s.attn = nn.MultiheadAttention(DM, HEADS, kdim=1024, vdim=1024,
                                       batch_first=True)
        # bn=True (byol2 arm): BatchNorm1d in the projector head -- BN's
        # cross-sample statistics are BYOL's implicit anti-collapse channel.
        s.head = (nn.Sequential(nn.Linear(DM, 256), nn.BatchNorm1d(256),
                                nn.GELU(), nn.Linear(256, DM)) if bn else
                  nn.Sequential(nn.Linear(DM, 256), nn.GELU(), nn.Linear(256, DM)))
        # center=True (cegate2c arm): running-mean centering BEFORE the L2
        # normalize (mu is a buffer: EMA'd on train forwards, frozen at eval,
        # archived in every checkpoint).
        s.center = center
        if center:
            s.register_buffer("mu", torch.zeros(DM))

    def forward(s, S, m=None):
        a, _ = s.attn(s.q0.expand(S.shape[0], -1, -1), S.float(), S.float(),
                      key_padding_mask=m, need_weights=False)
        h = s.head(a.mean(1))
        if s.center:
            if s.training:
                with torch.no_grad():
                    s.mu.mul_(0.99).add_(h.detach().float().mean(0), alpha=0.01)
            h = h - s.mu.to(h.dtype)
        return F.normalize(h, dim=-1)


def rown(x, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    s = x.std(-1, keepdims=True)
    return (x - m) / (s + eps)


def cov_pen(z):
    """VICReg-style covariance penalty: mean squared off-diagonal of the
    batch covariance, normalized by dim. Sample axis = games in the batch."""
    z = z - z.mean(0, keepdim=True)
    cov = (z.T @ z) / (z.shape[0] - 1)
    off = cov.flatten()[:-1].view(cov.shape[0] - 1, cov.shape[0] + 1)[:, 1:]
    return off.pow(2).sum() / z.shape[1]


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
    from VICReg_review.model import (GameCentroidExpander, vicreg_loss,
                                     vicreg_centroid_loss)

    tower_kind, FT = ARMS[args.arm]
    IW = _IW.get(tower_kind, 0.0)
    HIW = _IW.get(tower_kind, 1.0)
    CE_GATED = tower_kind.startswith("cegate") or tower_kind == "rgate2"
    I_GATED = tower_kind.startswith("igate")
    CENTERED = tower_kind in CENTER_ARMS
    _CE_E = ("i2expce", "expce", "expi2expce", "poolexpce", "expi2poolexpce",
             "shexpi2ce", "shexpi2poolce", "cmpce", "i2cmpce", "shcmpi2ce",
             "i2poolexpce", "i2poolcmpce", "expi2cmpce", "cmpi2expce",
             "cmpi2cmpce", "expi2poolcmpce", "shexpi2poolexpce",
             "shcmpi2poolcmpce", "poolcmpce", "shcmpi2poolce",
             "cmpi2poolexpce", "cmpi2poolcmpce")
    _CE_POOL = ("i2poolce", "poolce", "poolceexpi2", "poolexpce",
                "expi2poolexpce", "shexpi2poolce", "i2poolexpce",
                "i2poolcmpce", "expi2poolcmpce", "shexpi2poolexpce",
                "shcmpi2poolcmpce", "poolcmpce", "poolcecmpi2",
                "shcmpi2poolce", "cmpi2poolexpce", "cmpi2poolcmpce")
    _I_E = ("ceexpi2", "expi2expce", "poolceexpi2", "expi2poolexpce",
            "shexpi2ce", "shexpi2poolce", "shcmpi2ce", "expi2cmpce",
            "cmpi2expce", "cmpi2cmpce", "expi2poolcmpce",
            "shexpi2poolexpce", "shcmpi2poolcmpce", "cecmpi2",
            "poolcecmpi2", "shcmpi2poolce", "cmpi2poolexpce",
            "cmpi2poolcmpce")
    _DUAL = ("expi2expce", "expi2poolexpce", "expi2cmpce", "cmpi2expce",
             "cmpi2cmpce", "expi2poolcmpce", "cmpi2poolexpce",
             "cmpi2poolcmpce")   # I and CE use SEPARATE E's
    _POOL_AFTER = ("shexpi2poolce", "shcmpi2poolce")   # CE pools E-outputs
    # per-module E direction: (CE's E, I's E); None = that loss not in E
    _EDIR = {"expce": ("exp", None), "cmpce": ("cmp", None),
             "poolexpce": ("exp", None), "i2expce": ("exp", None),
             "i2cmpce": ("cmp", None), "i2poolexpce": ("exp", None),
             "i2poolcmpce": ("cmp", None), "ceexpi2": (None, "exp"),
             "poolceexpi2": (None, "exp"), "shexpi2ce": ("exp", "exp"),
             "shcmpi2ce": ("cmp", "cmp"), "shexpi2poolce": ("exp", "exp"),
             "shexpi2poolexpce": ("exp", "exp"),
             "shcmpi2poolcmpce": ("cmp", "cmp"),
             "expi2expce": ("exp", "exp"), "expi2cmpce": ("cmp", "exp"),
             "cmpi2expce": ("exp", "cmp"), "cmpi2cmpce": ("cmp", "cmp"),
             "expi2poolexpce": ("exp", "exp"),
             "expi2poolcmpce": ("cmp", "exp"), "poolcmpce": ("cmp", None),
             "cecmpi2": (None, "cmp"), "poolcecmpi2": (None, "cmp"),
             "shcmpi2poolce": ("cmp", "cmp"),
             "cmpi2poolexpce": ("exp", "cmp"),
             "cmpi2poolcmpce": ("cmp", "cmp")}
    XCE = tower_kind in _CE_E          # CE computed in expander space
    PCE = tower_kind in _CE_POOL       # CE pools the views first
    IE = tower_kind in _I_E            # I pairs live in expander space
    XPD = XCE or IE                    # arm carries the expander module
    DUAL = tower_kind in _DUAL         # separate E_I (xpd2) and E_CE (xpd)
    POOL_AFTER = tower_kind in _POOL_AFTER
    EDIR = _EDIR.get(tower_kind, (None, None))
    # naming grammar (user decree): i2exp* = I in DEPLOYED space (original
    # n4expce design); expi2* = I after the expander (new design)
    CW = _CW.get(tower_kind, 0.0)
    bank_m = re.match(r"bk(?:q(\d+)|b)i2cce$", tower_kind)
    BANK_POLICY = (("q" if bank_m.group(1) else "b") if bank_m else None)
    BANK_K = int(bank_m.group(1)) if bank_m and bank_m.group(1) else 0
    mq_m = re.match(r"mq(\d+)i2cce$", tower_kind)
    MQ_LEN = int(mq_m.group(1)) if mq_m else 0
    MQ_M = 0.99                        # shadow-tower weight-EMA momentum
    name = (f"w9_{args.arm}"
            + (f"_g{args.anchor_cap}" if args.anchor_cap != 512 else "")
            + ("_nsp" if args.no_sp_view else "")
            + (f"_ld{args.doc_lead}" if args.doc_lead else "")
            + ("_wllm" if args.wiki_src == "llm" else "")
            + (f"_w{args.view_w}" if args.view_w != 16 else "")
            + ("_fp" if args.full_pool else ""))
    dev = torch.device("cuda")
    C, OUT = Path(args.data_dir), Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] tower={tower_kind} ft={FT} IW={IW} anchor_cap={args.anchor_cap}"
          f" view_w={args.view_w}", flush=True)

    # ---------------- corpus: ALL of it onto the GPU ----------------
    G = np.load(C / "games.npz", allow_pickle=True)
    names = [str(x) for x in G["names"]]
    NG = len(names)
    n2i = {n: i for i, n in enumerate(names)}
    appid2name = {n.split("_")[0]: n for n in names}

    RIDp = np.load(C / "wscan_pool_rev_rid.npy")
    plen = np.load(C / "wscan_pool_rev_len.npy")
    need_pool = ((not args.full_pool)
                 or (args.anchor_cap != 512 and args.anchor_cap <= int(plen.max())
                     and not (C / f"wscan_gal_rev_g{args.anchor_cap}.npz").exists()))
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
    assert sp["seed"] == SPLIT_SEED
    test_g = {appid2name[a] for a in sp["test"]}
    val_g = {appid2name[a] for a in sp["val"]}
    traing = {appid2name[a] for a in sp["train"]}
    assert len(test_g) + len(val_g) == 407
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
        # prebuilt pack (built locally from embedding_h5, uploaded to the volume)
        GALd = np.load(C / f"wscan_gal_rev_g{args.anchor_cap}.npz")
        SGal = torch.tensor(GALd["gal"]).to(dev)
        gal_len = torch.tensor(np.asarray(GALd["gal_len"], np.int64)).to(dev)
        gal_doc = torch.tensor(np.asarray(GALd["gal_doc_len"], np.int64)).to(dev)
        print(f"anchors: prebuilt g{args.anchor_cap} pack, "
              f"used med {int(gal_len.float().median())}", flush=True)
    else:
        GCAP = args.anchor_cap
        full_src = GCAP > int(plen.max())     # the 2048 pool cannot fill past itself
        if full_src:
            assert args.full_pool, (
                f"anchor_cap {GCAP} exceeds the {int(plen.max())}-sentence pool; "
                "run with --full-pool so anchors can come from the full corpus")
        print(f"building {GCAP}-sentence anchors on GPU "
              f"(source: {'FULL corpus' if full_src else '2048 pool'}) ...", flush=True)
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
            if full_src:
                fi = f_e2i[names[g]]
                r0, r1 = int(f_gro[fi]), int(f_gro[fi + 1])
                r_starts = f_ro[r0:r1].astype(np.int64)      # ABSOLUTE sent idx
                r_lens = (f_ro[r0 + 1:r1 + 1] - f_ro[r0:r1]).astype(np.int64)
                segs = []
                for j in rngA.permutation(len(r_lens)):
                    L = int(r_lens[j])
                    if row + L <= GCAP:
                        segs.append((int(r_starts[j]), L))
                        row += L
                if segs:
                    flat = np.concatenate([np.arange(s, s + L) for s, L in segs])
                    order = np.argsort(flat, kind="stable")  # sorted mmap reads
                    vec = np.empty((len(flat), 1024), np.float32)
                    vec[order] = FULLV[flat[order]].astype(np.float32)
                    vec = (vec - vec.mean(-1, keepdims=True)) / \
                        (vec.std(-1, keepdims=True) + 1e-6)  # rown (pool is pre-rown)
                    SGal[g, row - len(flat):row] = \
                        torch.from_numpy(vec.astype(np.float16)).to(dev)
            else:
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
                outs.append(model(SGal[i:i + chunk], mGal[i:i + chunk]))
        return torch.cat(outs)

    def gallery_nodoc(model, chunk=128):
        outs = []
        with torch.no_grad():
            for i in range(0, NG, chunk):
                outs.append(model(SGal[i:i + chunk], mGal_nd[i:i + chunk]))
        return torch.cat(outs)

    def gallery_train(model, chunk=128):
        rows = train_pool_games
        outs = []
        for i in range(0, len(rows), chunk):
            r = torch.as_tensor(rows[i:i + chunk], device=dev)
            outs.append(model(SGal[r], mGal[r]))
        return torch.cat(outs)

    def gallery_rows(model, pos_rows, chunk=128):
        """Encode the anchors of train-pool POSITIONS pos_rows (bank refresh)."""
        outs = []
        for i in range(0, len(pos_rows), chunk):
            r = torch.as_tensor(train_pool_games[pos_rows[i:i + chunk]], device=dev)
            outs.append(model(SGal[r], mGal[r]))
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

    pairs = list(combinations(range(NV), 2))
    inv_t = 1.0 / 0.02

    def train_v4doc(seed=0, W=16, bs=192, per_epoch=3072):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = SetPoolN(4, center=CENTERED).to(dev)
        def _mkE(direction):
            # disposable loss space, discarded at eval -- deploy = pre-E.
            # exp goes UP (VICReg heritage), cmp goes DOWN (SimCLR bottleneck).
            d = (256, 512) if direction == "exp" else (128, 64)
            return nn.Sequential(nn.Linear(DM, d[0]), nn.GELU(),
                                 nn.Linear(d[0], d[1])).to(dev)
        xpd, xpd2 = None, None
        if DUAL:
            # I gets its OWN E (direction independent of CE's)
            xpd2 = _mkE(EDIR[1])
        if XPD:
            xpd = _mkE(EDIR[0] or EDIR[1])   # CE's E (or I's when CE not in E)
        params = (list(model.parameters())
                  + (list(xpd.parameters()) if xpd else [])
                  + (list(xpd2.parameters()) if xpd2 else []))
        opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-4)
        amp = amp_cls()
        ckpts = {}
        n_train = len(train_pool_games)
        bank, bank_ptr = None, 0
        shadow, mqueue, mq_gid, mq_ptr = None, None, None, 0
        if MQ_LEN:
            import copy
            shadow = copy.deepcopy(model).to(dev)
            for p in shadow.parameters():
                p.requires_grad_(False)
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
            if BANK_POLICY and "bank" in st:
                bank = st["bank"].to(dev)
                bank_ptr = int(st.get("bank_ptr", 0))
            if MQ_LEN and "mqueue" in st:
                shadow.load_state_dict({k: v.to(dev) for k, v in st["shadow"].items()})
                mqueue = st["mqueue"].to(dev)
                mq_gid = st["mq_gid"].to(dev)
                mq_ptr = int(st["mq_ptr"])
            if XPD and "xpd" in st:
                xpd.load_state_dict({k: v.to(dev) for k, v in st["xpd"].items()})
            if DUAL and "xpd2" in st:
                xpd2.load_state_dict({k: v.to(dev) for k, v in st["xpd2"].items()})
            print(f"RESUME from ep{start_ep}", flush=True)
        if start_ep == 0 and not RES.exists():
            # EXTEND fallback: the resume bundle is deleted when a run
            # completes; to train PAST the old budget, rebuild from the
            # newest checkpoint. Weights (+ mq shadow) come back exactly;
            # opt/amp/rng restart fresh; bank and mqueue fall through to
            # their cold-start inits below (mq prefills from the shadow).
            cands = [c for c in OUT.glob(f"ckpt_{name}_ep*.pt")
                     if int(c.stem.split("_ep")[-1]) < args.epochs]
            if cands:
                ck = max(cands, key=lambda c: int(c.stem.split("_ep")[-1]))
                st = torch.load(ck, map_location="cpu")
                sd = st["model"] if isinstance(st, dict) and "model" in st else st
                model.load_state_dict({k: v.to(dev) for k, v in sd.items()})
                if MQ_LEN and isinstance(st, dict) and "shadow" in st:
                    shadow.load_state_dict({k: v.to(dev)
                                            for k, v in st["shadow"].items()})
                if XPD and isinstance(st, dict) and "xpd" in st:
                    xpd.load_state_dict({k: v.to(dev)
                                         for k, v in st["xpd"].items()})
                if DUAL and isinstance(st, dict) and "xpd2" in st:
                    xpd2.load_state_dict({k: v.to(dev)
                                          for k, v in st["xpd2"].items()})
                start_ep = int(ck.stem.split("_ep")[-1])
                print(f"EXTEND from ckpt ep{start_ep} (fresh opt/amp/rng)",
                      flush=True)
        if BANK_POLICY and bank is None:
            with torch.no_grad():                       # fresh full init (age 0)
                bank = gallery_train(model).float().clone()
        if MQ_LEN and mqueue is None:
            # prefill the FIFO ring with shadow(=main at t0) keys of random
            # entities so the first steps see a full negative set.
            mqueue = torch.zeros(MQ_LEN, DM, device=dev)
            mq_gid = torch.full((MQ_LEN,), -1, dtype=torch.long, device=dev)
            fill = rng.choice(train_pool_games, MQ_LEN,
                              replace=MQ_LEN > n_train)
            with torch.no_grad():
                for i in range(0, MQ_LEN, 256):
                    sub = fill[i:i + 256]
                    mqueue[i:i + len(sub)] = gallery_rows(shadow, pos_of_g[sub]).float()
                    mq_gid[i:i + len(sub)] = torch.as_tensor(sub, device=dev)
        t0 = time.time()
        for ep in range(start_ep, args.epochs):
            model.train()
            for _ in range(per_epoch // bs):
                gids = rng.choice(train_pool_games, bs, replace=False)
                tgt = pos_of_g_t[gids].to(dev)
                with torch.amp.autocast("cuda"):
                    if MQ_LEN:
                        Zg = None                       # queue replaces gallery
                    elif BANK_POLICY == "q":
                        # queue rotation: refresh the next BANK_K rows (no
                        # grad), then the whole anchor matrix is the bank.
                        rows = np.arange(bank_ptr, bank_ptr + BANK_K) % n_train
                        bank_ptr = int((bank_ptr + BANK_K) % n_train)
                        with torch.no_grad():
                            bank[torch.as_tensor(rows, device=dev)] = \
                                gallery_rows(model, rows).float()
                        Zg = bank
                    elif BANK_POLICY == "b":
                        # refresh = batch: encode THIS batch's anchors fresh
                        # (grad flows, as in the full recipe); cold columns
                        # come from the bank; write the fresh ones back.
                        rows = pos_of_g[gids]
                        rows_t = torch.as_tensor(rows, device=dev)
                        fresh = gallery_rows(model, rows).float()
                        Zg = bank.detach().clone()
                        Zg[rows_t] = fresh              # autograd via index put
                        with torch.no_grad():
                            bank[rows_t] = fresh.detach()
                    else:
                        Zg = gallery_train(model)
                    Zs = [model(*sample_views(gids, W, rng)) for _ in range(NV - 1)]
                    Zs.append(assemble_doc_view(model, gids, W, rng, bs))
                    if MQ_LEN:
                        # user design: current keys enter at the ring head and
                        # ARE this step's positives; the rest of the ring
                        # (historical keys) are the negatives; pointer shifts
                        # by bs, evicting the oldest. Entity-ID mask removes
                        # stale same-game keys (false negatives at small N).
                        with torch.no_grad():
                            keys = gallery_rows(shadow, pos_of_g[gids]).float()
                        gid_t = torch.as_tensor(gids, device=dev)
                        slot = (torch.arange(bs, device=dev) + mq_ptr) % MQ_LEN
                        mqueue[slot] = keys
                        mq_gid[slot] = gid_t
                        mq_ptr = int((mq_ptr + bs) % MQ_LEN)
                        fmask = mq_gid[None, :] == gid_t[:, None]
                        fmask[torch.arange(bs, device=dev), slot] = False
                        loss = 0
                        for Z in Zs:
                            lg = Z.float() @ mqueue.T * inv_t
                            loss = loss + F.cross_entropy(
                                lg.masked_fill(fmask, -1e4), slot)
                    elif XCE and PCE and POOL_AFTER:
                        # shexpi2poolce: E each view FIRST, then pool the
                        # E-outputs (pool-position grammar: pool AFTER E)
                        Eg = F.normalize(xpd(Zg.float()), dim=-1)
                        em = F.normalize(torch.stack(
                            [F.normalize(xpd(Z.float()), dim=-1)
                             for Z in Zs]).mean(0), dim=-1)
                        loss = 4.0 * F.cross_entropy(em @ Eg.T * inv_t, tgt)
                    elif XCE and PCE:
                        # pool->E->CE: pool the views, then CE in E-space
                        Eg = F.normalize(xpd(Zg.float()), dim=-1)
                        zm = F.normalize(
                            torch.stack([Z.float() for Z in Zs]).mean(0), dim=-1)
                        loss = 4.0 * F.cross_entropy(
                            F.normalize(xpd(zm), dim=-1) @ Eg.T * inv_t, tgt)
                    elif XCE:
                        Eg = F.normalize(xpd(Zg.float()), dim=-1)
                        loss = sum(F.cross_entropy(
                            F.normalize(xpd(Z.float()), dim=-1) @ Eg.T * inv_t,
                            tgt) for Z in Zs)
                    elif PCE:
                        zm = F.normalize(
                            torch.stack([Z.float() for Z in Zs]).mean(0), dim=-1)
                        loss = 4.0 * F.cross_entropy(zm @ Zg.T.float() * inv_t, tgt)
                    elif tower_kind == "ai2lse":
                        # uniformity-only: repel from every WRONG anchor; no
                        # positive term (attraction is entirely the I block)
                        loss = 0.0
                        for Z in Zs:
                            lg = (Z.float() @ Zg.T.float() * inv_t).scatter(
                                1, tgt[:, None], -1e4)      # k^- only
                            loss = loss + torch.logsumexp(lg, dim=1).mean()
                    elif tower_kind in ("aali3auni", "aali2auni"):
                        # W&I uniformity vs the WRONG anchors (kernel zeroed
                        # at the own column) -- soft ai2lse, tau_eff 0.25.
                        loss = 0.0
                        for Z in Zs:
                            # exp(-t*||z-a||^2) == exp(2t*(cos-1)) on the
                            # sphere; computed via cos because cdist's
                            # BACKWARD is numerically wrong (~20x) here.
                            sim = Z.float() @ Zg.T.float()
                            ker = ((sim - 1.0) * (2 * UNI_T)).exp()
                            ker = ker.scatter(1, tgt[:, None], 0.0)
                            loss = loss + ker.sum(1).div(ker.shape[1] - 1)                                .log().mean() / len(Zs)
                    elif tower_kind in ("ali3uni", "ali2uni", "aali3uni"):
                        # W&I uniformity, official-repo form per view branch:
                        # log mean exp(-t*pdist^2) over the batch. Repulsion
                        # source = BATCH samples (self-organizing), never the
                        # anchor field; on the sphere == LSE(cos) at tau 0.25.
                        loss = 0.0
                        for Z in Zs:
                            d2 = torch.pdist(Z.float()).pow(2)
                            loss = loss + d2.mul(-UNI_T).exp().mean().log() / len(Zs)
                    elif tower_kind == "arc":
                        loss = sum(arcface_ce(Z.float() @ Zg.T.float(), tgt) for Z in Zs)
                    elif CE_GATED:
                        hd = torch.tensor(np.array([g in gate_games for g in gids])
                                          ).to(dev).nonzero(as_tuple=True)[0]
                        loss = (sum(F.cross_entropy(Z.float()[hd] @ Zg.T.float() * inv_t,
                                                    tgt[hd]) for Z in Zs)
                                if len(hd) else torch.zeros((), device=dev))
                    else:
                        loss = sum(F.cross_entropy(Z.float() @ Zg.T.float() * inv_t, tgt)
                                   for Z in Zs)
                    if IW > 0 and tower_kind in ("ai2ce", "ai2lse", "aali3uni", "aali3auni", "aali2auni"):
                        # anchor joins the alignment set: 4 views + own anchor
                        objs = [Z.float() for Z in Zs] + [Zg[tgt].float()]
                        loss = loss + IW * sum(
                            (1 - (objs[i] * objs[j]).sum(-1)).mean()
                            for i in range(5) for j in range(i + 1, 5)) / 10.0
                    elif IW > 0 and IE:
                        # E-space I; dual arms pay I into their OWN expander
                        _ei = xpd2 if DUAL else xpd
                        Ev = [F.normalize(_ei(Z.float()), dim=-1) for Z in Zs]
                        loss = loss + IW * sum(
                            (1 - (Ev[i] * Ev[j]).sum(-1)).mean()
                            for i, j in pairs) / len(pairs)
                    elif IW > 0:
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
                    if CW > 0:
                        loss = loss + CW * sum(cov_pen(Z.float())
                                               for Z in Zs) / len(Zs)
                opt.zero_grad()
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                amp.step(opt)
                amp.update()
                if MQ_LEN:
                    with torch.no_grad():
                        for pk, pq in zip(shadow.parameters(), model.parameters()):
                            pk.data.mul_(MQ_M).add_(pq.data, alpha=1 - MQ_M)
            if (ep + 1) % args.ckpt_every == 0:
                sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if MQ_LEN:
                    torch.save(dict(model=sd,
                                    shadow={k: v.detach().cpu().clone()
                                            for k, v in shadow.state_dict().items()}),
                               OUT / f"ckpt_{name}_ep{ep+1}.pt")
                elif XPD:
                    _pay = dict(model=sd,
                                xpd={k: v.detach().cpu().clone()
                                     for k, v in xpd.state_dict().items()})
                    if DUAL:
                        _pay["xpd2"] = {k: v.detach().cpu().clone()
                                        for k, v in xpd2.state_dict().items()}
                    torch.save(_pay, OUT / f"ckpt_{name}_ep{ep+1}.pt")
                else:
                    torch.save(sd, OUT / f"ckpt_{name}_ep{ep+1}.pt")   # persist NOW
                bundle = dict(model=sd, opt=opt.state_dict(), amp=amp.state_dict(),
                              cpu_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state(),
                              np_rng=rng.bit_generator.state, ep=ep + 1)
                if DUAL:
                    bundle["xpd2"] = {k: v.detach().cpu().clone()
                                      for k, v in xpd2.state_dict().items()}
                if XPD:
                    bundle["xpd"] = {k: v.detach().cpu().clone()
                                     for k, v in xpd.state_dict().items()}
                if BANK_POLICY:
                    bundle["bank"] = bank.detach().cpu()
                    bundle["bank_ptr"] = bank_ptr
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

    def train_byol(seed=0, W=16, bs=192, per_epoch=3072):
        import copy
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        bn = tower_kind == "byol2"
        model = SetPoolN(4, bn=bn).to(dev)
        # byol2: 3-layer BN predictor (BYOL-paper shape scaled to DM=128) --
        # more capacity to break online/target symmetry, BN to push the batch
        # apart. Plain byol keeps the original 2-layer GELU predictor.
        pred = (nn.Sequential(nn.Linear(DM, 512), nn.BatchNorm1d(512), nn.GELU(),
                              nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(),
                              nn.Linear(512, DM)) if bn else
                nn.Sequential(nn.Linear(DM, 256), nn.GELU(),
                              nn.Linear(256, DM))).to(dev)
        target = copy.deepcopy(model).to(dev)
        for p in target.parameters():
            p.requires_grad_(False)

        def _safe(net, S, m):
            # BN in train mode crashes on 1-row sub-batches (doc-view tiers
            # can select a single game); fall back to running stats for those.
            if bn and S.shape[0] == 1:
                was = net.training
                net.eval()
                out = net(S, m)
                if was:
                    net.train()
                return out
            return net(S, m)
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
        for ep in range(start_ep, args.epochs):
            model.train(); pred.train()
            for _ in range(per_epoch // bs):
                gids = rng.choice(train_pool_games, bs, replace=False)
                with torch.amp.autocast("cuda"):
                    views = [sample_views(gids, W, rng) for _ in range(NV - 1)]
                    Zo = [model(S, m) for S, m in views]
                    with torch.no_grad():
                        Zt = [target(S, m) for S, m in views]
                    Zo.append(assemble_doc_view(
                        lambda S, m: _safe(model, S, m), gids, W, rng, bs))
                    with torch.no_grad():
                        Zlast_t = torch.empty(bs, DM, device=dev, dtype=torch.float16)
                        assigned = np.zeros(bs, bool)
                        for g2x, Sx, mx in tiers:
                            msk = np.array([(not a) and (g in g2x)
                                            for a, g in zip(assigned, gids)])
                            if msk.any():
                                rows = [g2x[g] for g in gids[msk]]
                                Zlast_t[torch.tensor(msk).to(dev)] = \
                                    _safe(target, Sx[rows], mx[rows]).half()
                                assigned |= msk
                        rest = ~assigned
                        if rest.any():
                            Zlast_t[torch.tensor(rest).to(dev)] = \
                                _safe(target, *sample_views(gids[rest], W, rng)).half()
                        Zt.append(Zlast_t)
                    loss, npairs = 0.0, 0
                    for i in range(NV):
                        for j in range(NV):
                            if i == j:
                                continue
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
                        pt.data.mul_(0.996).add_(po.data, alpha=0.004)
                    for bt, bo in zip(target.buffers(), model.buffers()):
                        bt.data.copy_(bo.data)     # BN running stats: copy, not EMA
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

    def train_vicreg(seed=0, W=16, bs=192, per_epoch=3072):
        # Negative-free like BYOL (same 4 views, no gallery CE), but VICReg's
        # variance/covariance terms provide the explicit anti-collapse force
        # BYOL lacks. Invariance is MSE between unit-norm centroids; V/C act
        # on expander(centroid) where std>=1 is actually reachable.
        # epd_* arms use the CANONICAL wiring instead: all three terms on the
        # expander OUTPUT pair -- centroid collapse then violates V (constant
        # expander output), so the vic/vic2 degenerate solution is closed.
        epd = re.match(r"(epd[bg]?)_v(\d+)i(\d+)c(\d+)$", tower_kind)
        all_batch = bool(epd) and epd.group(1) == "epdb"  # views for ALL games
        grid_e = bool(epd) and epd.group(1) == "epdg"     # GRID-harmonized E:
        # same module as the 30-cell grid's exp (128->256->512, NO LayerNorm)
        # so "expce vs epdg" isolates the loss functional in an identical room
        if epd:
            vic_v, vic_i, vic_c = (float(g) for g in epd.groups()[1:])
        else:
            vic_i, vic_v, vic_c = VIC_W[tower_kind]
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = SetPoolN(4).to(dev)
        expander = (nn.Sequential(nn.Linear(DM, 256), nn.GELU(),
                                  nn.Linear(256, 512)).to(dev) if grid_e else
                    GameCentroidExpander(input_dim=DM).to(dev))
        opt = torch.optim.AdamW(list(model.parameters()) + list(expander.parameters()),
                                lr=5e-4, weight_decay=1e-4)
        amp = amp_cls()
        RES = OUT / f"resume_{name}.pt"
        start_ep = 0
        if RES.exists():
            st = torch.load(RES, map_location="cpu")
            model.load_state_dict({k: v.to(dev) for k, v in st["model"].items()})
            expander.load_state_dict({k: v.to(dev) for k, v in st["expander"].items()})
            opt.load_state_dict(st["opt"])
            amp.load_state_dict(st["amp"])
            torch.set_rng_state(st["cpu_rng"])
            torch.cuda.set_rng_state(st["cuda_rng"])
            rng.bit_generator.state = st["np_rng"]
            start_ep = int(st["ep"])
            print(f"RESUME from ep{start_ep}", flush=True)
        t0 = time.time()
        for ep in range(start_ep, args.epochs):
            model.train(); expander.train()
            for _ in range(per_epoch // bs):
                # epdb: full-population step -- views drawn for EVERY train
                # game; steps/epoch identical, only the moment sample grows.
                gids = (train_pool_games if all_batch else
                        rng.choice(train_pool_games, bs, replace=False))
                with torch.amp.autocast("cuda"):
                    Zs = [model(*sample_views(gids, W, rng)) for _ in range(NV - 1)]
                    Zs.append(assemble_doc_view(model, gids, W, rng, len(gids)))
                    if epd:
                        Es = [expander(Z.float()) for Z in Zs]
                        loss = sum(vicreg_loss(
                            Es[i], Es[j], invariance_weight=vic_i,
                            variance_weight=vic_v, covariance_weight=vic_c)["loss"]
                            for i, j in pairs) / len(pairs)
                    else:
                        loss = sum(vicreg_centroid_loss(
                            Zs[i].float(), Zs[j].float(), expander,
                            invariance_weight=vic_i, variance_weight=vic_v,
                            covariance_weight=vic_c)["loss"]
                            for i, j in pairs) / len(pairs)
                opt.zero_grad()
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(expander.parameters()), 5.0)
                amp.step(opt)
                amp.update()
            if (ep + 1) % args.ckpt_every == 0:
                sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                xd = {k: v.detach().cpu().clone() for k, v in expander.state_dict().items()}
                # archive the expander per checkpoint (same decree as BYOL aux)
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

    # ---------------- eval machinery ----------------
    VORDER = ["neutral", "noname", "positive", "negative"]

    def zs_metrics(model):
        with torch.no_grad():
            Zg = gallery(model).float().cpu().numpy()
            Za = torch.cat([model(SA[i:i+256], mA[i:i+256])
                            for i in range(0, SA.shape[0], 256)]).float().cpu().numpy()
        gz = Zg / (np.linalg.norm(Zg, axis=1, keepdims=True) + 1e-8)
        az = Za / (np.linalg.norm(Za, axis=1, keepdims=True) + 1e-8)
        out = {}
        for var in ("neutral", "noname"):
            ii = [i for i, g in enumerate(art_games) if g in test_g and variants[i] == var]
            sim = az[ii] @ gz.T
            tgt = A["gidx"][ii]
            rk = (sim > sim[np.arange(len(ii)), tgt][:, None]).sum(1) + 1
            out["nm_" + var] = float((rk == 1).mean())
        sc, rg, al, th, _ = train_anchor_ridge(targs, Zg, y, n2i, tag_split)
        for var in ("neutral", "noname"):
            idx = [i for i in range(len(art_games))
                   if variants[i] == var and art_games[i] in test_g]
            s = rg.predict(sc.transform(np.stack([Za[i] for i in idx]).astype(np.float32)))
            labs = np.stack([y[n2i[art_games[i]]] for i in idx])
            out["tag_" + var] = micro_prf(labs, s, th)["micro_f1"]
        return out

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
        return out

    d_rows = [g2wiki[g] for g in sorted(g2wiki)]
    d_gidx = np.array(sorted(g2wiki), np.int64)

    def project_cache(model, path):
        NQ = len(Qs["gidx"])
        with torch.no_grad():
            SPg = gallery(model).float().cpu().numpy()
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

    # ---------------- tower + checkpoints ----------------
    DONE_FLAG = OUT / f"tower_{name}_ep{args.epochs}.npz"
    if not DONE_FLAG.exists():
        t0 = time.time()
        if tower_kind.startswith("byol"):
            train_byol(seed=0, W=args.view_w)
        elif tower_kind in VIC_W or tower_kind.startswith("epd"):
            train_vicreg(seed=0, W=args.view_w)
        else:
            train_v4doc(seed=0, W=args.view_w)
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
            m2 = SetPoolN(4, bn=tower_kind == "byol2", center=CENTERED).to(dev)
            m2.load_state_dict({k: v.to(dev) for k, v in sd.items()})
            m2.eval()
            zk = zs_metrics(m2)
            zs_traj[f"ep{ek}"] = zk
            print(f"ZS(ep{ek}): {dict((k, round(v, 3)) for k, v in zk.items())}",
                  flush=True)
            json.dump(zs_traj, open(traj_p, "w"), indent=2)
            project_cache(m2, npz)
            del m2
        (OUT / f"resume_{name}.pt").unlink(missing_ok=True)

    # ---------------- heads: per-checkpoint 3 seeds -> vsel pick -> topup ----
    if tower_kind.startswith("byol"):
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
