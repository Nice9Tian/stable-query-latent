# -*- coding: utf-8 -*-
"""Evaluation: four-variant retrieval + tag metrics, bare-tower zero-shot,
and the frozen-embedder baseline (paper row E)."""
import numpy as np
import torch

from .anchors import gallery
from .backhead_tag import micro_prf, train_anchor_ridge

VORDER = ["neutral", "noname", "positive", "negative"]


def metrics4(B, gal, art):
    """Retrieval (hit@1/hit@5/median-rank) + tag micro-F1 on the TEST games,
    per query variant. gal [NG, D] anchors, art [NQ, D] query features."""
    gz = gal / (np.linalg.norm(gal, axis=1, keepdims=True) + 1e-8)
    az = art / (np.linalg.norm(art, axis=1, keepdims=True) + 1e-8)
    out = {}
    sc, rg, al, th, _ = train_anchor_ridge(B.targs, gal, B.y, B.n2i, B.tag_split)
    for var in VORDER:
        ii = [i for i, g in enumerate(B.art_games)
              if g in B.test_g and B.variants[i] == var]
        sim = az[ii] @ gz.T
        tgt = B.gidxA[ii]
        rk = (sim > sim[np.arange(len(ii)), tgt][:, None]).sum(1) + 1
        s = rg.predict(sc.transform(
            np.stack([art[i] for i in ii]).astype(np.float32)))
        labs = np.stack([B.y[B.n2i[B.art_games[i]]] for i in ii])
        out[var] = dict(h1=float((rk == 1).mean()), h5=float((rk <= 5).mean()),
                        med=float(np.median(rk)),
                        tag=micro_prf(labs, s, th)["micro_f1"])
    return out


def project_queries(model, B, chunk=256):
    with torch.no_grad():
        return torch.cat([model(B.SA[i:i + chunk], B.mA[i:i + chunk])
                          for i in range(0, B.SA.shape[0], chunk)])


def zs_metrics(model, B):
    """Bare-tower zero-shot: name hit@1 + anchor-ridge tag on neutral/noname."""
    with torch.no_grad():
        Zg = gallery(model, B).float().cpu().numpy()
        Za = project_queries(model, B).float().cpu().numpy()
    gz = Zg / (np.linalg.norm(Zg, axis=1, keepdims=True) + 1e-8)
    az = Za / (np.linalg.norm(Za, axis=1, keepdims=True) + 1e-8)
    out = {}
    for var in ("neutral", "noname"):
        ii = [i for i, g in enumerate(B.art_games)
              if g in B.test_g and B.variants[i] == var]
        sim = az[ii] @ gz.T
        tgt = B.gidxA[ii]
        rk = (sim > sim[np.arange(len(ii)), tgt][:, None]).sum(1) + 1
        out["nm_" + var] = float((rk == 1).mean())
    sc, rg, al, th, _ = train_anchor_ridge(B.targs, Zg, B.y, B.n2i, B.tag_split)
    for var in ("neutral", "noname"):
        idx = [i for i in range(len(B.art_games))
               if B.variants[i] == var and B.art_games[i] in B.test_g]
        s = rg.predict(sc.transform(
            np.stack([Za[i] for i in idx]).astype(np.float32)))
        labs = np.stack([B.y[B.n2i[B.art_games[i]]] for i in idx])
        out["tag_" + var] = micro_prf(labs, s, th)["micro_f1"]
    return out


def frozen_baseline(B):
    """Paper row E: game vector = masked mean of its anchor sentences
    (input_dim-d, no tower); query vector = mean of its variant sentences."""
    with torch.no_grad():
        w = (~B.mGal).float().unsqueeze(-1)
        Zg = (B.SGal.float() * w).sum(1) / w.sum(1).clamp(min=1)
        wA = (~B.mA).float().unsqueeze(-1)
        Za = (B.SA.float() * wA).sum(1) / wA.sum(1).clamp(min=1)
    return metrics4(B, Zg.cpu().numpy(), Za.cpu().numpy())
