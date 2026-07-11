# -*- coding: utf-8 -*-
"""BackHead-NAME: the two-phase linear retrieval head on tower projections.

Operates in projection space (tower features precomputed per checkpoint):
  phase-1  pseudo-query warm start — anchor-shaped queries of train games;
  phase-2  the family's document features of the wiki-train games with the
           arm-matched loss:
             ice — CE vs train gallery + in-batch negs + own-anchor pull
             ce  — CE vs train gallery + in-batch negs (no pull)
             by  — own-anchor alignment only, target detached (no negatives)
             arc — additive angular margin CE, fixed scale
Training negatives = train-game gallery columns only (fully inductive);
selection = the vsel composite on val wiki_neutral/wiki_noname hits.

Doc-gating (GAME-level 2:1): each game carries its own phase offset — its
anchor is doc-bearing for 2 of every 3 gallery computations and reviews-only
for the 3rd, so every batch sees a MIXED gallery. Selection/eval always use
full (doc-bearing) anchors.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .protocol import vsel_score

ARC_S, ARC_M = 30.0, 0.2       # head ArcFace scale / additive angular margin


def _arc_logits(logits, tgt):
    cos_t = logits.gather(1, tgt[:, None]).clamp(-1 + 1e-7, 1 - 1e-7)
    phi = torch.cos(torch.acos(cos_t) + ARC_M)
    phi = torch.where(cos_t > math.cos(math.pi - ARC_M),
                      phi, cos_t - ARC_M * math.sin(ARC_M))
    return logits.scatter(1, tgt[:, None], phi)


def train_backhead_name(B, Xg, Xg_nd, Xa, Xq, Xd, d_pos, seed, ep2=600,
                        p1="ice", p2="ice", ls=0.0, iw=2.0, out_dim=128):
    """Returns (gallery_head_feats, query_head_feats, vsel_stats)."""
    dev = B.dev
    torch.manual_seed(seed)
    np.random.seed(seed)
    o_g = torch.tensor(np.random.randint(0, 3, Xg.shape[0]), device=dev)
    gstep = 0

    def gated_gallery():
        nonlocal gstep
        use_doc = ((gstep + o_g) % 3) < 2          # per-game bool
        gstep += 1
        return torch.where(use_doc[:, None], Xg, Xg_nd)

    head = nn.Linear(Xg.shape[1], out_dim).to(dev)
    fwd = lambda x: F.normalize(head(x), dim=-1)
    logt = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32,
                                     device=dev))
    opt = torch.optim.AdamW(list(head.parameters()) + [logt], lr=1e-3,
                            weight_decay=1e-4)
    qpos = B.pos_of_g_t[B.q_gidx].to(dev)      # -1 for held-out (never sampled)
    gA_t = B.gA

    def vsel():
        with torch.no_grad():
            Zg = fwd(Xg)
            sim = fwd(Xa[B.va_neu]) @ Zg.T
            rk = (sim > sim.gather(1, gA_t[B.va_neu][:, None])).sum(1) + 1
            h_neu = float((rk == 1).float().mean())
            sim = fwd(Xa[B.va_non]) @ Zg.T
            rk = (sim > sim.gather(1, gA_t[B.va_non][:, None])).sum(1) + 1
            h_non = float((rk == 1).float().mean())
            h_non5 = float((rk <= 5).float().mean())
        return vsel_score(h_neu, h_non, h_non5), h_neu, h_non, h_non5

    # ---------------- phase 1: pseudo-query warm start ----------------
    best, pat = -float("inf"), 0
    bs_ = {k: v.detach().clone() for k, v in head.state_dict().items()}
    for ep in range(80):
        head.train()
        order = np.random.choice(B.q_train, 6144, replace=True)
        for k in range(0, 6144, 256):
            b = torch.tensor(order[k:k + 256]).to(dev)
            Zg = fwd(gated_gallery()[B.tp_t])
            if p1 == "by":
                loss = (1 - (fwd(Xq[b]) * Zg[qpos[b]].detach()).sum(-1)).mean()
            elif p1 == "ice":
                zq = fwd(Xq[b])
                loss = F.cross_entropy(zq @ Zg.T * logt.exp(), qpos[b],
                                       label_smoothing=ls)
                loss = loss + iw * (1 - (zq * Zg[qpos[b]]).sum(-1)).mean()
            else:
                loss = F.cross_entropy(fwd(Xq[b]) @ Zg.T * logt.exp(), qpos[b],
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
                bs_ = {k: v.detach().clone() for k, v in head.state_dict().items()}
            else:
                pat += 1
            if pat >= 8:
                break
    head.load_state_dict(bs_)

    # ---------------- phase 2: family document features ----------------
    opt = torch.optim.AdamW(list(head.parameters()) + [logt], lr=1e-4,
                            weight_decay=1e-4)
    bsel = vsel()
    best, pat = bsel[0], 0
    bs_ = {k: v.detach().clone() for k, v in head.state_dict().items()}
    nd = Xd.shape[0]
    for ep in range(ep2):
        head.train()
        order = np.random.permutation(nd)
        for k in range(0, nd, 128):
            b = torch.tensor(order[k:k + 128]).to(dev)
            z = fwd(Xd[b])
            Zg = fwd(gated_gallery()[B.tp_t])
            tgt = d_pos[b]
            if p2 == "by":
                loss = (1 - (z * Zg[tgt].detach()).sum(-1)).mean()
            elif p2 == "arc":
                la = z @ Zg.T
                lt = z @ z.T - torch.eye(z.shape[0], device=dev) * 1e4
                loss = F.cross_entropy(
                    _arc_logits(torch.cat([la, lt], 1), tgt) * ARC_S, tgt)
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
                bs_ = {k: v.detach().clone() for k, v in head.state_dict().items()}
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
