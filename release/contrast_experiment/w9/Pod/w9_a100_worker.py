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
    # bce family (user): CE with BATCH negatives = in-batch NT-Xent/SimCLR.
    # Bare "ce" above is grandfathered anchored-CE (== ace in the grammar;
    # paper notation aCE). bce uses NO anchor gallery: keys = this step's
    # 4*bs views, positive = the NEXT view (ring) of the same game, self +
    # remaining siblings masked. Same tau .02 (controlled); per-view sum
    # keeps the weight-4 push convention. bce/i2bce skip the full-gallery
    # re-encode entirely; ai2bce pays it only for the I rope rows.
    "wcle_bce_cetf": ("bce", "ce"),          # pure SimCLR baseline
    "wcle_i2bce_icetf": ("i2bce", "ice"),    # + I x2 (views)
    "wcle_ai2bce_icetf": ("ai2bce", "ice"),  # + anchor joins I (CE stays batch)
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
    # ---- view-COMPOSITION grid (user; explicit grammar, see repo-root
    # model_history.md): [d<k>][w<k>][sp<k>]r<n>_i2ce -- every token states
    # its view count. d = TIERED doc slot (wiki -> sp -> review fallback,
    # the protocol slot), w = wiki-ONLY slot, sp = store-page-ONLY slot
    # (full sp coverage, wiki-bearing games NOT excluded), r = review
    # views. Each doc-type slot falls back to a review view where its doc
    # is missing, so every game always gets the full NV. The protocol
    # NV=4 == d1r3 (implicit for every non-grid arm).
    # Per-view-sum convention preserved: CE terms and I pairs grow with NV
    # (4->5/6/7 CE terms; 6->10/15/21 I edges) -- these cells scale the
    # WHOLE objective with the view count, not I alone.
    # (nodoc above = doc value at constant NV; d1r4 = +1 view instead.)
    "wcle_d1r4_i2ce_icetf": ("d1r4_i2ce", "ice"),     # 4R+1D  NV=5
    "wcle_d1r5_i2ce_icetf": ("d1r5_i2ce", "ice"),     # 5R+1D  NV=6
    "wcle_d1r6_i2ce_icetf": ("d1r6_i2ce", "ice"),     # 6R+1D  NV=7
    # w1sp1r3: wiki view AND store-page view COEXIST in the same step
    # (the tiered protocol slot never allows both on one game), NV=5.
    "wcle_w1sp1r3_i2ce_icetf": ("w1sp1r3_i2ce", "ice"),
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
    "wcle_ai2auni25_icetf": ("ai2auni25", "ice"),        # ANCHOR-IN-I + uni@t25 (ex-ai2lse):
    # I pairs span {4 views + OWN ANCHOR} (10 edges) so alignment itself ties
    # anchor_g to its view cluster; repulsion = same negative-only LSE.
    # = CE fully decomposed W&I-style: symmetric constant-weight attraction
    # (incl. the positive edge) + DCL uniformity. (i2lse, the no-anchor-rope
    # variant, was purged pre-run: desert equilibria made it foreseeable.)
    # vs ai2ce isolates dropping CE's adaptive pull (LSE keeps the push).
    "wcle_ai2ce_icetf": ("ai2ce", "ice"),          # BOTH pulls (user, 2x2 corner):
    # full per-view CE (adaptive pull + push, tau .02) AND anchor-in-I
    # (constant rope, 10 edges). vs i2ce = the anchor edge's worth ON TOP
    # of CE; vs ai2auni25 = CE's adaptive pull's worth GIVEN the rope.
    # Completes {CE-pull x anchor-in-I}: i2ce(Y/N) ai2ce(Y/Y) ai2auni25(N/Y),
    # i2lse(N/N) purged.
    "wcle_ai6uni2_icetf": ("ai6uni2", "ice"),        # W&I uniformity swap:
    # same anchor-in-I attraction as ai2auni25, but repulsion = Wang&Isola
    # batch uniformity log mean exp(-t*d^2) (t=2) over the 192 batch views
    # per branch -- NOT the anchor field. On the sphere that kernel is
    # exp(2t*cos) => LSE over cos at tau=1/(2t)=0.25. vs ai6auni2 isolates
    # repulsion SOURCE alone (paper recipe fixed, anchors -> batch).
    "wcle_i6uni2_icetf": ("i6uni2", "ice"),          # PURE Wang&Isola cell:
    # 3*align (= IW 6 x mean(1-cos), alpha=2 identity) + 1*uniformity,
    # official-repo STL-10 flagship weights. NO anchor edge anywhere =
    # first anchor-free NEGATIVE-based arm (BYOL/VICReg are anchor-free
    # but negative-free). No desert trap: batch self-repulsion has no
    # fixed field to hide from, unlike the purged i2lse. Doc VIEW stays
    # (protocol constant NV=4), so anchor-free != doc-free.
    "wcle_ai6auni2_icetf": ("ai6auni2", "ice"),      # W&I BOTH-AT-ANCHOR (user):
    # paper recipe (3*align IW6 + 1*uniform t=2) with BOTH forces aimed at
    # the anchor field: pull = anchor-in-I, push = Gaussian kernel vs the
    # WRONG anchors. Gradient-identical to ai2auni25's LSE at tau=1/(2t)=0.25
    # (log-mean vs log-sum = additive const), so the ladder factorizes:
    # ai2auni25 -> ai6auni2 = kernel softness + pull:push budget (source fixed);
    # ai6auni2 -> ai6uni2 = repulsion source only (recipe fixed).
    "wcle_ai4auni2_icetf": ("ai4auni2", "ice"),  # i4 rung of ai6auni2
    "wcle_ai25auni2_icetf": ("ai25auni2", "ice"),   # deployed high-pull (IW25)
    # gated anchor-W&I 3x3 (user): {pull gate}ai25{push gate}auni2 --
    # align(IW25)@E_g1 + anchor-uniformity(t2)@E_g2, g in {exp,cmp,pj};
    # pj = parallel proj 128->256->128 (no scaling). @512, 2000ep.
    "wcle_expai25expauni2_icetf": ("expai25expauni2", "ice"),
    "wcle_expai25cmpauni2_icetf": ("expai25cmpauni2", "ice"),
    "wcle_expai25pjauni2_icetf": ("expai25pjauni2", "ice"),
    "wcle_cmpai25expauni2_icetf": ("cmpai25expauni2", "ice"),
    "wcle_cmpai25cmpauni2_icetf": ("cmpai25cmpauni2", "ice"),
    "wcle_cmpai25pjauni2_icetf": ("cmpai25pjauni2", "ice"),
    "wcle_pjai25expauni2_icetf": ("pjai25expauni2", "ice"),
    "wcle_pjai25cmpauni2_icetf": ("pjai25cmpauni2", "ice"),
    "wcle_pjai25pjauni2_icetf": ("pjai25pjauni2", "ice"),
    # shared-E (sh{g}): ONE projection E_g serves BOTH align and
    # uniformity (vs the dual grid separate E's) -- like shexpi2ce.
    "wcle_shexpai25auni2_icetf": ("shexpai25auni2", "ice"),
    "wcle_shcmpai25auni2_icetf": ("shcmpai25auni2", "ice"),
    "wcle_shpjai25auni2_icetf": ("shpjai25auni2", "ice"),
    "wcle_i4uni2_icetf": ("i4uni2", "ice"),      # i4 rung of i6uni2
    # digit grammar (user decree v2, ONE dial in i units): the number after
    # i/ai == IW, same scale as i2ce's "2". Invariance == align/2 on the
    # sphere (same operator), so W&I repo-flagship 3*align+1*uniform == i6
    # and its align_w-2 rung == i4.
    # push grammar (user decree v3): uni{t} == Gaussian-kernel/LSE repulsion
    # at tau = 1/(2t); a- prefix = vs the WRONG anchors, bare = batch views.
    # t=25 -> tau .02 == tower tau (the cell formerly named ai2lse); t=2 =
    # W&I repo default. ce keeps its own token: the only push that is FUSED
    # with an adaptive pull inside one softmax.
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
    "wcle_mq3072ce_cetf": ("mq3072ce", "ce"),   # CE-only MoCo queue (no I/C; queue CE baseline)
    # --- save-the-4096-tag trio (user; owned by w9_save_4096_tag.ipynb) ---
    # Diagnosis: tag decay at big caps tracks the ANCHOR SUPPLY, not I
    # (512: i2ce tag > ce; mq queue @2048 keeps I and restores tag .736).
    # Culprit = anchor co-adaptation: CE grad flows through Zg and games
    # the pack embeddings off the content manifold late in training.
    "wcle_i2q2ce_icetf": ("i2q2ce", "ice"),   # 1: soft-band I, lambda*(1-cos)^2
    # (gradient ratio vs plain = 2*(1-cos): equal at cos=.5, 5x softer at
    # .9, 25x at .98 -- a near-aligned dead band preserves per-view
    # individuality; n generalizes as i2q{n}ce if 2 proves out)
    "wcle_i2sgce_icetf": ("i2sgce", "ice"),   # 5a: stop-grad gallery
    # (anchors can't be gamed; no_grad also drops the 1613 x cap backward
    # activations -> cap 8192 fits 80G)
    "wcle_i2esce_icetf": ("i2esce", "ice"),   # 5b: EMA-shadow gallery
    # --- twin-pack fractal (user, w9_packageview.ipynb): i2ce recursed one
    # scale up. Anchor budget cap is SPLIT into two cap/2 packs (contiguous
    # halves of the SAME pack -> at g1024 sentence-identical to i2ce@1024;
    # doc prefix lands in pack A). Pack level runs i2ce: per-pack CE vs the
    # normalized-mean gallery (each 512-pack must identify its game alone;
    # the anchor side gets its OWN discriminative objective instead of being
    # shaped only through view-CE backprop = another anti-gaming lever) +
    # pack-I x2. Views: CE only (vce) or full i2ce (vi2ce, the paired cell
    # that keeps view-I). Eval/deploy gallery = normalize(mean(eA, eB)).
    "wcle_pk2i2cevce_icetf": ("pk2i2cevce", "ice"),
    "wcle_pk2i2cevi2ce_icetf": ("pk2i2cevi2ce", "ice"),
    # ---- SWIN family (user): swin{W}step{S}loop{L}i2ce -- fresh-only
    # sliding window over the anchor catalog (ring order). Per optimizer
    # step, L micro-passes each fresh-encode the NEXT W ring games and run
    # their OWN CE partition [now-batch | window] with immediate backward
    # (window activations freed between passes: compute-for-VRAM); the
    # now-batch anchors are ALWAYS in the field with grad (the noname edge
    # stays open); micro-pass l starts at p + l*S, pointer += L*S per step
    # (S = games slid per micro-pass; W-S overlap between passes). NO cache
    # anywhere -- bkq/bkb proved cached fast-student rows = incoherent field.
    "wcle_swin168step84loop2i2ce_icetf": ("swin168step84loop2i2ce", "ice"),
    "wcle_swin84step42loop2i2ce_icetf": ("swin84step42loop2i2ce", "ice"),
    # coverage-scan low point: (192+126)/1613 ~ 19.7%/step
    "wcle_swin336step168loop2i2ce_icetf": ("swin336step168loop2i2ce", "ice"),
    # coverage-scan high point: (192+504)/1613 ~ 43.2%/step
    "wcle_pk4i2cevi2ce_icetf": ("pk4i2cevi2ce", "ice"),   # QUAD pack (user):
    # 4 packs of cap/4 (@2048 -> 4x512), full i2ce at pack AND view level.
    # Capacity ladder vs pk2@1024 (2x512) / pk2@2048 (2x1024) / pk2@4096.
    "wcle_pk2i2vce_icetf": ("pk2i2vce", "ice"),   # PACK-I ONLY (user): packs
    # tied by I x2 with NO pack-CE -- anchors trained by pack-I + view-CE
    # backprop through the mean gallery; views = per-view CE only. Single-
    # factor pair vs pk2i2cevce (.647): isolates the pack-CE term.
    "wcle_pk2i2cesgvce_icetf": ("pk2i2cesgvce", "ice"),   # sg BOUNDARY (user):
    # capacity/read-out grid (user): slot{N}i2ce{mean|line}, N in {4,8,16};
    # (4,mean) == existing i2ce (kept, not re-run). Loss = plain i2ce.
    "wcle_slot4i2celine_icetf": ("slot4i2celine", "ice"),
    "wcle_slot8i2cemean_icetf": ("slot8i2cemean", "ice"),
    "wcle_slot8i2celine_icetf": ("slot8i2celine", "ice"),
    "wcle_slot16i2cemean_icetf": ("slot16i2cemean", "ice"),
    "wcle_slot16i2celine_icetf": ("slot16i2celine", "ice"),
    # scale-autonomy version -- packs train THEMSELVES (pack-CE + pack-I,
    # grad); views do CE against the DETACHED mean gallery (one-way chase,
    # views can never game the anchors; the user's original mental model
    # of anchor semantics, layered on the fractal design).
    # (stop-grad + target smoothing, tau = 1/(1-MQ_M) = 100 steps ~ 6 ep;
    # fixed points = i2sgce's, transients damped; mq's consistency lesson
    # at full-gallery width)
    "wcle_mq3072i2ce_icetf": ("mq3072i2ce", "ice"),  # queue + I2 (NO C: aligned
    # with the scale grid pair {ce, i2ce}; C shown null-to-harmful, user trim)
}
CENTER_ARMS = {"cegate2c", "i2ccec"}
_CW = {"i2cce": 1.0, "i2ccec": 1.0,
       "bkq192i2cce": 1.0, "bkq48i2cce": 1.0, "bkq12i2cce": 1.0,
       "bkbi2cce": 1.0, "mq3072i2cce": 1.0}           # covariance weight
_IW = {"ice": 1.0, "i2ce": 2.0, "cegate1": 1.0, "cegate2": 2.0, "cegate3": 3.0,
       "cegate4": 4.0, "cegate1w": 1.0, "cegate2w": 2.0, "igate1": 1.0,
       "igate1w": 1.0, "rgate2": 2.0, "nodoc": 2.0, "cegate2c": 2.0,
       "i2cce": 2.0, "i2ccec": 2.0, "ai2auni25": 2.0, "ai2ce": 2.0, "i2bce": 2.0, "ai2bce": 2.0,
       "ai6uni2": 6.0, "i6uni2": 6.0, "ai6auni2": 6.0, "ai4auni2": 4.0,
       "i4uni2": 4.0,   # W&I i-units: digit == IW (2x paper align_w, alpha2)
       "ai25auni2": 25.0,   # deployed high-pull anchor-W&I (user)
       "expai25expauni2": 25.0, "expai25cmpauni2": 25.0, "expai25pjauni2": 25.0, "cmpai25expauni2": 25.0, "cmpai25cmpauni2": 25.0, "cmpai25pjauni2": 25.0, "pjai25expauni2": 25.0, "pjai25cmpauni2": 25.0, "pjai25pjauni2": 25.0,   # gated anchor-W&I 3x3 (all IW25)
       "shexpai25auni2": 25.0, "shcmpai25auni2": 25.0, "shpjai25auni2": 25.0,   # shared-E (IW25)
       "i2expce": 2.0, "i2poolce": 2.0,   # (were bug-commented -> IW0)
       "ceexpi2": 2.0, "expi2expce": 2.0, "poolceexpi2": 2.0, "expi2poolexpce": 2.0,
       "shexpi2ce": 2.0, "shexpi2poolce": 2.0, "i2cmpce": 2.0, "shcmpi2ce": 2.0,
       "i2poolexpce": 2.0, "i2poolcmpce": 2.0, "expi2cmpce": 2.0,
       "cmpi2expce": 2.0, "cmpi2cmpce": 2.0, "expi2poolcmpce": 2.0,
       "shexpi2poolexpce": 2.0, "shcmpi2poolcmpce": 2.0,
       "cecmpi2": 2.0, "poolcecmpi2": 2.0, "shcmpi2poolce": 2.0,
       "cmpi2poolexpce": 2.0, "cmpi2poolcmpce": 2.0,
       "bkq192i2cce": 2.0, "bkq48i2cce": 2.0, "bkq12i2cce": 2.0,
       "bkbi2cce": 2.0, "mq3072i2cce": 2.0, "mq3072i2ce": 2.0,
       "i2q2ce": 2.0, "i2sgce": 2.0, "i2esce": 2.0,
       "pk2i2cevi2ce": 2.0, "pk4i2cevi2ce": 2.0, "swin168step84loop2i2ce": 2.0, "swin84step42loop2i2ce": 2.0, "swin336step168loop2i2ce": 2.0,   # vce variant has NO view-I (pack-I is hardcoded 2)
       "slot4i2celine": 2.0, "slot8i2cemean": 2.0, "slot8i2celine": 2.0, "slot16i2cemean": 2.0, "slot16i2celine": 2.0,
       "d1r4_i2ce": 2.0, "d1r5_i2ce": 2.0, "d1r6_i2ce": 2.0,
       "w1sp1r3_i2ce": 2.0}
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
    ap.add_argument("--head", action="store_true",   # legacy FT head phase (ZS-only is the default protocol)
                    default=False)
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
    ap.add_argument("--measure-vram", default="",
                    help="VRAM calibration: run 3 real training steps at this "
                         "cap (incl. the 4x-view backward loss matrices), write "
                         "peak max_memory_allocated bytes to this file, exit(0). "
                         "Used by the scheduler warmup.")
    return ap.parse_args()


class SetPoolN(nn.Module):
    def __init__(s, N, bn=False, center=False, pool="mean"):
        super().__init__()
        s.q0 = nn.Parameter(torch.randn(1, N, DM) * 0.02)
        s.attn = nn.MultiheadAttention(DM, HEADS, kdim=1024, vdim=1024,
                                       batch_first=True)
        s.pool = pool
        if pool == "line":
            # learned linear pool over the N slots (slot-concat -> DM).
            # Initialized to the block [I/N ... I/N] so it EQUALS mean-pool
            # at init -> identical start, any learned deviation = pure signal.
            s.lp = nn.Linear(N * DM, DM)
            with torch.no_grad():
                w = torch.zeros(DM, N * DM)
                for n in range(N):
                    w[:, n * DM:(n + 1) * DM] = torch.eye(DM) / N
                s.lp.weight.copy_(w)
                s.lp.bias.zero_()
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
        pooled = (s.lp(a.reshape(a.shape[0], -1)) if s.pool == "line"
                  else a.mean(1))
        h = s.head(pooled)
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
    # USER LAW (2026-07-17, 16/16 structure cells): pooled CE is a SHORTCUT
    # -- only the mean direction is constrained, so the encoder learns an
    # input-conditional policy (signal into easy views, cancelling noise in
    # the rest) and per-view eval collapses (pool cells mean neu .572/non
    # .261 vs per-view .877/.575; pool->E->CE worst, zvsel to -223). CE must
    # be PER-VIEW in every future arm; the pool cells stay as evidence.
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
    BCE = tower_kind in ("bce", "i2bce", "ai2bce")   # in-batch negatives
    PK = tower_kind in ("pk2i2cevce", "pk2i2cevi2ce", "pk2i2cesgvce",
                        "pk2i2vce", "pk4i2cevi2ce")   # N-pack fractal
    PKN = int(re.match(r"pk(\d+)", tower_kind).group(1)) if PK else 0
    swin_m = re.match(r"swin(\d+)step(\d+)loop(\d+)i2ce$", tower_kind)
    SWIN = bool(swin_m)                # fresh-only sliding-window CE field
    SWIN_W = int(swin_m.group(1)) if swin_m else 0
    SWIN_S = int(swin_m.group(2)) if swin_m else 0
    SWIN_L = int(swin_m.group(3)) if swin_m else 0
    slot_m = re.match(r"slot(\d+)i2ce(mean|line)$", tower_kind)   # capacity grid
    SLOTS = int(slot_m.group(1)) if slot_m else 4
    POOL_MODE = slot_m.group(2) if slot_m else "mean"   # slot pooling (NOT the sentence POOL tensor!)
    PCE = tower_kind in _CE_POOL       # CE pools the views first
    IE = tower_kind in _I_E            # I pairs live in expander space
    XPD = XCE or IE                    # arm carries the expander module
    DUAL = tower_kind in _DUAL         # separate E_I (xpd2) and E_CE (xpd)
    POOL_AFTER = tower_kind in _POOL_AFTER
    EDIR = _EDIR.get(tower_kind, (None, None))
    # gated anchor-W&I (user 3x3): {g1}ai25{g2}auni2 -- align(pull) in
    # E_g1, anchor-uniformity(push) in E_g2, dual projections g in
    # {exp,cmp,pj}. Reuses the DUAL wiring: xpd=push E_g2, xpd2=pull E_g1.
    _gm = re.match(r"^(exp|cmp|pj)ai25(exp|cmp|pj)auni2$", tower_kind)
    _gs = re.match(r"^sh(exp|cmp|pj)ai25auni2$", tower_kind)   # ONE shared E
    GAUNI = (_gm is not None) or (_gs is not None)
    if _gm:
        EDIR = (_gm.group(2), _gm.group(1))   # (push g2 -> xpd, pull g1 -> xpd2)
        DUAL = XPD = True
    elif _gs:
        EDIR = (_gs.group(1), None)   # single shared E_g -> xpd (both losses)
        XPD = True
    # naming grammar (user decree): i2exp* = I in DEPLOYED space (original
    # n4expce design); expi2* = I after the expander (new design)
    CW = _CW.get(tower_kind, 0.0)
    bank_m = re.match(r"bk(?:q(\d+)|b)i2cce$", tower_kind)
    BANK_POLICY = (("q" if bank_m.group(1) else "b") if bank_m else None)
    BANK_K = int(bank_m.group(1)) if bank_m and bank_m.group(1) else 0
    # mq{N}i2cce = MoCo queue + I2CCE (I2+C+CE); mq{N}ce = CE-ONLY queue
    mq_m = re.match(r"mq(\d+)(?:i2cce|i2ce|ce)$", tower_kind)
    MQ_LEN = int(mq_m.group(1)) if mq_m else 0
    MQ_M = 0.99                        # shadow-tower weight-EMA momentum
    USE_SHADOW = bool(MQ_LEN) or tower_kind == "i2esce"   # EMA twin needed
    # view-composition grid, explicit grammar (see model_history.md):
    # [d<k>][w<k>][sp<k>]r<n>_i2ce -- d = tiered doc slots (wiki -> sp ->
    # review fallback, the protocol slot), w = wiki-only slots, sp =
    # store-page-only slots (full sp coverage), r = review views. Every
    # non-grid arm is implicitly d1r3 (protocol NV=4) -- NV_ARM == NV.
    vp_m = re.match(r"(?:d(\d+))?(?:w(\d+))?(?:sp(\d+))?r(\d+)_i2ce$",
                    tower_kind)
    N_DOC = int(vp_m.group(1) or 0) if vp_m else 1     # tiered doc slots
    N_WIKI = int(vp_m.group(2) or 0) if vp_m else 0    # wiki-only slots
    N_SP = int(vp_m.group(3) or 0) if vp_m else 0      # sp-only slots
    N_REV = int(vp_m.group(4)) if vp_m else NV - 1     # review views
    NV_ARM = N_REV + N_DOC + N_WIKI + N_SP
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
    assert not (N_SP and args.no_sp_view), \
        "sp<k> arms need the sp view; --no-sp-view contradicts the arm"
    _vplan = (f"{N_REV}R" + (f"+{N_DOC}D" if N_DOC else "")
              + (f"+{N_WIKI}W" if N_WIKI else "")
              + (f"+{N_SP}SP" if N_SP else ""))
    print(f"[{name}] tower={tower_kind} ft={FT} IW={IW} anchor_cap={args.anchor_cap}"
          f" view_w={args.view_w} nv={NV_ARM}({_vplan})", flush=True)

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
    # sp<k> slots (e.g. w1sp1r3_i2ce): full sp coverage INCLUDING
    # wiki-bearing games (the tiered g2store above deliberately excludes
    # them; the coexisting-doc arms want both docs in the same step).
    g2store_all = {int(ST["gidx"][i]): i for i in range(len(ST["gidx"]))
                   if str(ST["names"][i]) not in excl}
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
    if PK:
        # SEEDED RANDOM PACKS (user): each of the PKN packs is an
        # INDEPENDENT random h-sample of the game's anchor sentences --
        # packs may OVERLAP (the user rejected disjoint contiguous
        # slices). Per-game fixed Generator seed => packs are bit-
        # identical across restarts/resume/eval. Review indices come from
        # a TILED randperm over [gal_doc, gal_len): without replacement
        # while the pool lasts, balanced wrap when the pool is short (so
        # under-filled games need no refill and no dead packs). The doc
        # prefix stays pinned at the head of pack 0 and never enters the
        # sample (copies elsewhere would escape the pack-0 nodoc mask).
        # Degenerate doc-only rows (no reviews) keep pack 0 only; their
        # later packs stay dead: position 0 unmasked (finite attention)
        # and aliveW weights them out of every mean/CE/pack-I term.
        assert args.anchor_cap % PKN == 0 and args.anchor_cap // PKN >= 512, \
            f"pk{PKN} arms need cap % {PKN} == 0 and packs >= 512"
        # NOTATION: the user's pk@N codename numbers are PER-PACK size;
        # --anchor-cap here is the TOTAL budget = PKN * per-pack.
        PK_SEED = 20260718
        _h = SGal.shape[1] // PKN
        idxP = torch.zeros(PKN, NG, _h, dtype=torch.long)
        _mP = torch.ones(PKN, NG, _h, dtype=torch.bool)
        aliveW = torch.ones(NG, PKN)
        for _g in range(NG):
            _l, _d = int(gal_len[_g]), int(gal_doc[_g])
            _R = _l - _d
            _gen = torch.Generator().manual_seed(PK_SEED + _g)
            for _k in range(PKN):
                _dk = min(_d, _h) if _k == 0 else 0
                _need = _h - _dk
                if _need == 0:               # pack 0 entirely doc prefix
                    idxP[_k, _g] = torch.arange(_h)
                    _mP[_k, _g] = False
                elif _R > 0:
                    _pm = torch.cat([torch.randperm(_R, generator=_gen)
                                     for _ in range(-(-_need // _R))])[:_need]
                    if _dk:
                        idxP[_k, _g] = torch.cat(
                            [torch.arange(_dk), _d + _pm])
                    else:
                        idxP[_k, _g] = _d + _pm
                    _mP[_k, _g] = False
                elif _k == 0:                # doc-only row, short prefix
                    idxP[_k, _g, :_dk] = torch.arange(_dk)
                    _mP[_k, _g, :_dk] = False
                else:                        # dead pack: NaN guard only
                    _mP[_k, _g, 0] = False
                    aliveW[_g, _k] = 0.0
        idxP = idxP.to(dev)
        aliveW = aliveW.to(dev)
        mGalP = [_mP[k].to(dev) for k in range(PKN)]
        mGalP_nd = [mGalP[0] | (torch.arange(_h, device=dev)[None, :] <
                                torch.clamp(gal_doc, max=_h)[:, None])] \
            + mGalP[1:]

        def pk_sg(k, rows):
            # gather pack k's sentences for the given game rows on the fly
            # (rows: 1-D long tensor) -> (len(rows), _h, D); no cap-sized
            # copies are ever materialized.
            return SGal[rows[:, None], idxP[k][rows]]

        def pk_mean(embs, w):
            # alive-weighted normalized mean of per-pack embeddings.
            # embs: PKN x (B, D); w: (B, PKN). Pack 0 is always alive.
            e = torch.stack(embs, 1)
            return F.normalize((e * w[:, :, None]).sum(1)
                               / w.sum(1, keepdim=True), dim=-1)
    if args.full_pool and POOL is not None:
        # POOL (2020x2048 sentence pool, ~8.5 GB on GPU) is only needed to
        # BUILD anchors at startup under the fp protocol -- the view sampler
        # reads FULLV (host mmap), never POOL. Freeing it returns ~8.5 GB
        # per worker; the 3-tower packageview co-residency was riding the
        # 80G edge with it resident (~22G/tower -> ~13.5G).
        POOL = None
        torch.cuda.empty_cache()
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
                if PK:
                    _rw = torch.arange(i, min(i + chunk, NG), device=dev)
                    _es = [model(pk_sg(k, _rw), mGalP[k][i:i + chunk])
                           for k in range(PKN)]
                    outs.append(pk_mean(_es, aliveW[i:i + chunk]))
                else:
                    outs.append(model(SGal[i:i + chunk], mGal[i:i + chunk]))
        return torch.cat(outs)

    def gallery_nodoc(model, chunk=128):
        outs = []
        with torch.no_grad():
            for i in range(0, NG, chunk):
                if PK:
                    _rw = torch.arange(i, min(i + chunk, NG), device=dev)
                    _es = [model(pk_sg(k, _rw), mGalP_nd[k][i:i + chunk])
                           for k in range(PKN)]
                    outs.append(pk_mean(_es, aliveW[i:i + chunk]))
                else:
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

    def assemble_doc_view(model, gids, W, rng, bs, tiers_=None):
        Zlast = torch.empty(bs, DM, device=dev, dtype=torch.float16)
        assigned = np.zeros(bs, bool)
        for g2x, Sx, mx in (tiers if tiers_ is None else tiers_):
            msk = np.array([(not a) and (g in g2x) for a, g in zip(assigned, gids)])
            if msk.any():
                rows = [g2x[g] for g in gids[msk]]
                Zlast[torch.tensor(msk).to(dev)] = model(Sx[rows], mx[rows]).half()
                assigned |= msk
        rest = ~assigned
        if rest.any():
            Zlast[torch.tensor(rest).to(dev)] = model(*sample_views(gids[rest], W, rng)).half()
        return Zlast

    pairs = list(combinations(range(NV_ARM), 2))
    inv_t = 1.0 / 0.02

    def train_v4doc(seed=0, W=16, bs=192, per_epoch=3072):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = SetPoolN(SLOTS, center=CENTERED, pool=POOL_MODE).to(dev)
        def _mkE(direction):
            # disposable loss space, discarded at eval -- deploy = pre-E.
            # exp UP (VICReg), cmp DOWN (SimCLR bottleneck), pj FLAT
            # (parallel projection 128->128->128, EQUAL width -- faithful to
            # Wang&Isola's fc7 4096->4096, no scaling between layers; user).
            d = {"exp": (256, 512), "cmp": (128, 64),
                 "pj": (128, 128)}.get(direction, (128, 64))
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
        swin_ptr = 0
        shadow, mqueue, mq_gid, mq_ptr = None, None, None, 0
        if USE_SHADOW:
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
            if SWIN:
                swin_ptr = int(st.get("swin_ptr", 0))
            if MQ_LEN and "mqueue" in st:
                shadow.load_state_dict({k: v.to(dev) for k, v in st["shadow"].items()})
                mqueue = st["mqueue"].to(dev)
                mq_gid = st["mq_gid"].to(dev)
                mq_ptr = int(st["mq_ptr"])
            elif USE_SHADOW and "shadow" in st:
                shadow.load_state_dict({k: v.to(dev) for k, v in st["shadow"].items()})
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
                if USE_SHADOW and isinstance(st, dict) and "shadow" in st:
                    shadow.load_state_dict({k: v.to(dev)
                                            for k, v in st["shadow"].items()})
                if SWIN and isinstance(st, dict) and "swin_ptr" in st:
                    swin_ptr = int(st["swin_ptr"])
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
        mv_steps = 0
        if args.measure_vram and dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for ep in range(start_ep, args.epochs):
            model.train()
            for _ in range(per_epoch // bs):
                gids = rng.choice(train_pool_games, bs, replace=False)
                tgt = pos_of_g_t[gids].to(dev)
                with torch.amp.autocast("cuda"):
                    if MQ_LEN:
                        Zg = None                       # queue replaces gallery
                    elif SWIN:
                        # fresh-only sliding-window field: NO gallery, NO
                        # cache. CE lives entirely in the L window micro-
                        # passes grafted at the backward stage; here only
                        # the NOW anchors are encoded (grad, always in the
                        # field). Generic CE chain gets an empty ride; the
                        # I chain still runs on the views.
                        rows_now = pos_of_g[gids]
                        rows_now_t = torch.as_tensor(rows_now, device=dev)
                        eNow = gallery_rows(model, rows_now).float()
                        Zg = None
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
                    elif PK:
                        # N packs WITH grad (train pool); CE gallery = the
                        # alive-weighted normalized mean; per-pack CE +
                        # pack-I added after the view-CE chain.
                        _ls = [[] for _ in range(PKN)]
                        for _i in range(0, len(train_pool_games), 128):
                            _r = torch.as_tensor(
                                train_pool_games[_i:_i + 128], device=dev)
                            for _k in range(PKN):
                                _ls[_k].append(
                                    model(pk_sg(_k, _r), mGalP[_k][_r]))
                        eP = [torch.cat(x) for x in _ls]
                        aliveT = aliveW[torch.as_tensor(
                            train_pool_games, device=dev)]
                        Zg_pk = pk_mean(eP, aliveT)
                        # sg arm: views chase a DETACHED gallery; pack terms
                        # keep the grad path (they train the anchors).
                        Zg = (Zg_pk.detach() if tower_kind == "pk2i2cesgvce"
                              else Zg_pk)
                    elif tower_kind == "i2sgce":
                        # 5a stop-grad gallery: no grad through the anchors,
                        # AND no retained forward activations (torch.no_grad,
                        # NOT .detach() -- detach builds then discards the
                        # graph, so the forward peak wouldn't drop).
                        with torch.no_grad():
                            Zg = gallery_train(model)
                    elif tower_kind == "i2esce":
                        # 5b EMA-shadow gallery: stop-grad + slow target
                        # (lag 1/(1-MQ_M) = 100 steps ~ 6 ep).
                        with torch.no_grad():
                            Zg = gallery_train(shadow)
                    elif tower_kind in ("bce", "i2bce"):
                        Zg = None       # anchor-free: no gallery re-encode
                    else:
                        Zg = gallery_train(model)
                    # view plan (explicit grammar): N_REV review views, then
                    # the doc-type slots -- d = tiered protocol slot, w =
                    # wiki-only, sp = store-page-only; every doc slot falls
                    # back to a review view where its doc is missing.
                    Zs = [model(*sample_views(gids, W, rng)) for _ in range(N_REV)]
                    for _ in range(N_DOC):
                        Zs.append(assemble_doc_view(model, gids, W, rng, bs))
                    for _ in range(N_WIKI):
                        Zs.append(assemble_doc_view(model, gids, W, rng, bs,
                                                    tiers_=[(g2wiki, SW, mW)]))
                    for _ in range(N_SP):
                        Zs.append(assemble_doc_view(model, gids, W, rng, bs,
                                                    tiers_=[(g2store_all, SS, mS)]))
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
                    elif SWIN:
                        loss = torch.zeros((), device=dev)
                    elif BCE:
                        # in-batch NT-Xent: logits vs ALL 4*bs views of this
                        # step; target = ring sibling; self + other siblings
                        # masked (false negatives).
                        keys = torch.cat([Z.float() for Z in Zs])
                        ar = torch.arange(bs, device=dev)
                        loss = 0.0
                        for v in range(NV):
                            lg = Zs[v].float() @ keys.T * inv_t
                            for u in range(NV):
                                if u != (v + 1) % NV:
                                    lg[ar, u * bs + ar] = -1e4
                            loss = loss + F.cross_entropy(
                                lg, ((v + 1) % NV) * bs + ar)
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
                    elif tower_kind == "ai2auni25":
                        # uniformity-only: repel from every WRONG anchor; no
                        # positive term (attraction is entirely the I block)
                        loss = 0.0
                        for Z in Zs:
                            lg = (Z.float() @ Zg.T.float() * inv_t).scatter(
                                1, tgt[:, None], -1e4)      # k^- only
                            loss = loss + torch.logsumexp(lg, dim=1).mean()
                    elif tower_kind in ("ai6auni2", "ai4auni2", "ai25auni2"):
                        # W&I uniformity vs the WRONG anchors (kernel zeroed
                        # at the own column) -- soft ai2auni25, tau_eff 0.25.
                        loss = 0.0
                        for Z in Zs:
                            # exp(-t*||z-a||^2) == exp(2t*(cos-1)) on the
                            # sphere; computed via cos because cdist's
                            # BACKWARD is numerically wrong (~20x) here.
                            sim = Z.float() @ Zg.T.float()
                            ker = ((sim - 1.0) * (2 * UNI_T)).exp()
                            ker = ker.scatter(1, tgt[:, None], 0.0)
                            loss = loss + ker.sum(1).div(ker.shape[1] - 1)                                .log().mean() / len(Zs)
                    elif tower_kind in ("i6uni2", "i4uni2", "ai6uni2"):
                        # W&I uniformity, official-repo form per view branch:
                        # log mean exp(-t*pdist^2) over the batch. Repulsion
                        # source = BATCH samples (self-organizing), never the
                        # anchor field; on the sphere == LSE(cos) at tau 0.25.
                        loss = 0.0
                        for Z in Zs:
                            d2 = torch.pdist(Z.float()).pow(2)
                            loss = loss + d2.mul(-UNI_T).exp().mean().log() / len(Zs)
                    elif GAUNI:
                        # push: anchor-uniformity in E_g2 (xpd). Kernel
                        # exp(2t(cos-1)) of E(views) vs E(WRONG anchors),
                        # own column zeroed; t = UNI_T (=2, "auni2").
                        Eg = F.normalize(xpd(Zg.float()), dim=-1)
                        loss = 0.0
                        for Z in Zs:
                            Ez = F.normalize(xpd(Z.float()), dim=-1)
                            ker = ((Ez @ Eg.T - 1.0) * (2 * UNI_T)).exp(
                                ).scatter(1, tgt[:, None], 0.0)
                            loss = loss + ker.sum(1).div(
                                ker.shape[1] - 1).log().mean() / len(Zs)
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
                    if PK:
                        # pack-level i2ce, alive-weighted: per-pack CE vs the
                        # mean gallery (empty packs sit out) + pack-I x2 over
                        # every alive pack pair. Uses Zg_pk (the GRAD
                        # gallery) so the sg arm's boundary only cuts the
                        # view->anchor edge, never the pack level. For pk2
                        # with all packs alive this reduces EXACTLY to the
                        # old eA/eB code (mean over the single pair).
                        if tower_kind != "pk2i2vce":   # pack-I-only arm skips pack-CE
                            _tga = torch.arange(Zg_pk.shape[0], device=dev)
                            for _k in range(PKN):
                                _al = aliveT[:, _k] > 0
                                if _al.any():
                                    loss = loss + F.cross_entropy(
                                        eP[_k][_al].float()
                                        @ Zg_pk.T.float() * inv_t, _tga[_al])
                        _is, _ic = 0.0, 0.0
                        for _a in range(PKN):
                            for _b in range(_a + 1, PKN):
                                _w = aliveT[:, _a] * aliveT[:, _b]
                                _is = _is + ((1 - (eP[_a].float()
                                                  * eP[_b].float()).sum(-1))
                                             * _w).sum()
                                _ic = _ic + _w.sum()
                        loss = loss + 2.0 * _is / _ic.clamp(min=1.0)
                    if IW > 0 and GAUNI:
                        # pull: anchor joins align, in E_g1 (xpd2). 4
                        # views + own anchor, 10 edges, weight IW (=25).
                        _ep = xpd2 if DUAL else xpd   # shared xpd for sh*
                        E1 = [F.normalize(_ep(Z.float()), dim=-1) for Z in Zs] \
                            + [F.normalize(_ep(Zg[tgt].float()), dim=-1)]
                        loss = loss + IW * sum(
                            (1 - (E1[i] * E1[j]).sum(-1)).mean()
                            for i in range(5) for j in range(i + 1, 5)) / 10.0
                    elif IW > 0 and tower_kind == "i2q2ce":
                        # soft-band I (user): lambda*(1-cos)^2 per view pair.
                        loss = loss + IW * sum(
                            ((1 - (Zs[i].float() * Zs[j].float()).sum(-1)) ** 2
                             ).mean() for i, j in pairs) / len(pairs)
                    elif IW > 0 and tower_kind in ("ai2ce", "ai2bce", "ai2auni25", "ai6uni2", "ai6auni2", "ai4auni2", "ai25auni2"):
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
                if SWIN:
                    # L window micro-passes (compute-for-VRAM): each pass
                    # fresh-encodes the next W ring games, builds its OWN
                    # CE partition [now | window] for every view, and
                    # backwards IMMEDIATELY -- the window's activations are
                    # freed before the next pass. retain_graph keeps only
                    # the shared view/eNow graphs alive; the final generic
                    # backward (I terms) releases them. Window columns that
                    # duplicate a now-batch game are masked (their positive
                    # already sits in the now block). SEMANTICS (user): pass
                    # l's window starts at p + l*S (S = games slid per
                    # micro-pass; consecutive passes overlap W-S); after the
                    # step p += L*S. S=W/2, L=2 => half-overlap chain, every
                    # ring position covered by exactly W/S=2 passes/sweep.
                    _arb = torch.arange(bs, device=dev)
                    for _l in range(SWIN_L):
                        _w = (swin_ptr + SWIN_S * _l
                              + np.arange(SWIN_W)) % n_train
                        _wt = torch.as_tensor(_w, device=dev)
                        _dup = torch.isin(_wt, rows_now_t)
                        with torch.amp.autocast("cuda"):
                            eWin = gallery_rows(model, _w).float()
                            lw = 0.0
                            for Z in Zs:
                                lg = torch.cat(
                                    [Z.float() @ eNow.T * inv_t,
                                     (Z.float() @ eWin.T * inv_t
                                      ).masked_fill(_dup[None, :], -1e4)], 1)
                                lw = lw + F.cross_entropy(lg, _arb)
                        amp.scale(lw).backward(retain_graph=True)
                    swin_ptr = int((swin_ptr + SWIN_L * SWIN_S) % n_train)
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
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
                if USE_SHADOW:
                    with torch.no_grad():
                        for pk, pq in zip(shadow.parameters(), model.parameters()):
                            pk.data.mul_(MQ_M).add_(pq.data, alpha=1 - MQ_M)
            if (ep + 1) % args.ckpt_every == 0:
                sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if USE_SHADOW:
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
                if USE_SHADOW:
                    bundle["shadow"] = {k: v.detach().cpu().clone()
                                        for k, v in shadow.state_dict().items()}
                if MQ_LEN:
                    bundle["mqueue"] = mqueue.detach().cpu()
                    bundle["mq_gid"] = mq_gid.detach().cpu()
                    bundle["mq_ptr"] = mq_ptr
                if SWIN:
                    bundle["swin_ptr"] = swin_ptr
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

    def zs_from_arrays(Zg, Za, Zq):
        # ZS-primary protocol (user 2026-07-16): all four TEST variants
        # (h1+h5), VAL-side neutral/noname, and zvsel = the head-phase vsel
        # piecewise ported to zero-shot:
        #   zvsel = max(S(v_non@1;.45), S(v_non5@5;.65)) + S(v_neu@1;.85)
        gz = Zg / (np.linalg.norm(Zg, axis=1, keepdims=True) + 1e-8)
        az = Za / (np.linalg.norm(Za, axis=1, keepdims=True) + 1e-8)
        out = {}

        def _rk(idx):
            sim = az[idx] @ gz.T
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
        out["zvsel"] = (max(S_fn(out["v_non"], 0.45), S_fn(out["v_non5"], 0.65))
                        + S_fn(out["v_neu"], 0.85))
        sc, rg, al, th, _ = train_anchor_ridge(targs, Zg, y, n2i, tag_split)
        for var in ("neutral", "noname"):
            idx = [i for i in range(len(art_games))
                   if variants[i] == var and art_games[i] in test_g]
            s = rg.predict(sc.transform(np.stack([Za[i] for i in idx]).astype(np.float32)))
            labs = np.stack([y[n2i[art_games[i]]] for i in idx])
            out["tag_" + var] = micro_prf(labs, s, th)["micro_f1"]
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
            sim = qz[qi] @ gz.T
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
            Zg = gallery(model).float().cpu().numpy()
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
            m2 = SetPoolN(SLOTS, bn=tower_kind == "byol2", center=CENTERED, pool=POOL_MODE).to(dev)
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

    # ---------------- ZS-primary: refresh traj + val-select (zsbest) ----
    # Old traj entries (test-only keys) are refreshed from the projection
    # npz caches -- no GPU re-embedding. Selection = max zvsel, earlier
    # epoch on ties (matches the head post-hoc convention).
    traj_p = OUT / f"zs_traj_{name}.json"
    zs_traj = json.loads(traj_p.read_text()) if traj_p.exists() else {}
    for npzp in sorted(OUT.glob(f"tower_{name}_ep*.npz"),
                       key=lambda q: int(q.stem.split("_ep")[-1])):
        ek = npzp.stem.split("_ep")[-1]
        if "rvsel" in zs_traj.get(f"ep{ek}", {}):
            continue
        T0 = np.load(npzp)
        zs_traj[f"ep{ek}"] = zs_from_arrays(T0["SPg"], T0["SPa"], T0["SPq"])
        print(f"ZS-refresh(ep{ek}) zvsel={zs_traj[f'ep{ek}']['rvsel']:.3f}",
              flush=True)
        json.dump(zs_traj, open(traj_p, "w"), indent=2)
    cand = {k: v for k, v in zs_traj.items() if "rvsel" in v}
    if cand:
        bk = max(cand, key=lambda k: (cand[k]["rvsel"], -int(k[2:])))
        json.dump(dict(best_ep=int(bk[2:]), **cand[bk]),
                  open(OUT / f"zsbest_{name}.json", "w"), indent=2)
        print(f"ZSBEST {name}: ep{bk[2:]} rvsel={cand[bk]['rvsel']:.3f} "
              + " ".join(f"{v[:3]}:{cand[bk]['nm_' + v]:.3f}" for v in VORDER),
              flush=True)

    # ---------------- heads (LEGACY, --head only): 3 seeds -> vsel -> topup --
    if args.head:
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
