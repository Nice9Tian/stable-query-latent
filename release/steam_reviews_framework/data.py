# -*- coding: utf-8 -*-
"""Asset loading: one call builds the full task bundle (games, anchors,
review pool, eval queries, pseudo-queries, tag labels, split/exclusions).

GPU layout mirrors the local 3080 protocol: anchors + eval queries resident
on device (fp16), the review pool and the ~9.5 GB pseudo-query bank stay in
host RAM as mmaps.
"""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from dataset_builder.paths import ASSETS, TEXT_H5
from .protocol import load_split
from .sampler import build_review_table
from .backhead_tag import make_or_load_split


def rown(x, eps=1e-6):
    """Legacy per-row standardization (mean 0 / std 1 along the last axis)."""
    m = x.mean(-1, keepdims=True)
    s = x.std(-1, keepdims=True)
    return (x - m) / (s + eps)


def load_views(fname, device):
    """A views npz (S, S_len, gidx, names) -> (raw dict, tensor, mask)."""
    d = np.load(ASSETS / fname, allow_pickle=True)
    Sx = rown(torch.tensor(d["S"]).to(device).float()).half()
    mx = torch.arange(Sx.shape[1], device=device)[None, :] >= \
        torch.tensor(d["S_len"]).to(device)[:, None]
    return d, Sx, mx


def load_bundle(device, cv_fold=None):
    B = SimpleNamespace()
    B.dev = device
    G = np.load(ASSETS / "games.npz", allow_pickle=True)
    B.names = [str(x) for x in G["names"]]
    B.NG = len(B.names)
    B.n2i = {n: i for i, n in enumerate(B.names)}
    appid2name = {n.split("_")[0]: n for n in B.names}

    # ---- anchors (sp_clean doc prefix + whole reviews @512) ----
    GAL = np.load(ASSETS / "wscan_gal_rev.npz")
    B.SGal = torch.tensor(GAL["gal"]).to(device)
    gl = torch.tensor(GAL["gal_len"]).to(device)
    ar = torch.arange(B.SGal.shape[1], device=device)[None, :]
    B.mGal = ar >= gl[:, None]
    # reviews-only anchor mask (doc prefix ALSO masked) for head doc-gating
    B.mGal_nd = B.mGal | (ar < torch.tensor(GAL["gal_doc_len"]).to(device)[:, None])

    # ---- review pool (host mmap) + rejection table ----
    B.pool = np.load(ASSETS / "wscan_pool_rev.npy", mmap_mode="r")
    rid = np.load(ASSETS / "wscan_pool_rev_rid.npy")
    B.rev_tab = build_review_table(rid)

    # ---- wiki eval queries (4 variants x eval universe, COMPLETE text) ----
    A = np.load(ASSETS / "wiki_eval.npz", allow_pickle=True)
    B.SA = rown(torch.tensor(A["S"]).to(device).float()).half()
    B.mA = torch.arange(B.SA.shape[1], device=device)[None, :] >= \
        torch.tensor(A["S_len"]).to(device)[:, None]
    B.gA = torch.tensor(A["gidx"]).to(device)
    B.gidxA = A["gidx"]
    B.variants = [str(x) for x in A["variants"]]
    B.art_games = [str(x) for x in A["names"]]

    # ---- split + fully-inductive exclusion ----
    B.test_g, B.val_g, B.train_doc_g = load_split(appid2name, cv_fold)
    B.excl = B.test_g | B.val_g
    assert cv_fold is not None or len(B.excl) == 407
    B.train_pool_games = np.array(
        [i for i in range(B.NG) if B.names[i] not in B.excl])
    B.pos_of_g = np.full(B.NG, -1, dtype=np.int64)
    B.pos_of_g[B.train_pool_games] = np.arange(len(B.train_pool_games))
    B.pos_of_g_t = torch.tensor(B.pos_of_g)
    B.tp_t = torch.tensor(B.train_pool_games).to(device)

    # ---- 23-tag labels + tag probe split ----
    with h5py.File(TEXT_H5, "r") as h:
        tgn = [g.decode() if isinstance(g, bytes) else str(g)
               for g in h["game_names"][:]]
        tl = h["tag_labels"][:].astype(np.int8)
    g2 = {n: i for i, n in enumerate(tgn)}
    B.y = np.zeros((B.NG, tl.shape[1]), np.int8)
    for i, n in enumerate(B.names):
        if n in g2:
            B.y[i] = tl[g2[n]]
    B.targs = SimpleNamespace(tag_text_train_frac=0.7, tag_text_val_frac=0.15,
                              tag_text_split_seed=42, seed=42,
                              tag_text_threshold_steps=33)
    B.tag_split = make_or_load_split(ASSETS / "_tag_split.json", B.names, B.targs)

    # ---- pseudo-queries (anchor-shaped, flat-stored, host mmap) ----
    Qs = np.load(ASSETS / "ss_queries_rev.npz")
    B.q_off, B.q_gidx = Qs["off"], Qs["gidx"]
    B.QS_S = np.load(ASSETS / "ss_queries_rev_S.npy", mmap_mode="r")
    hgi = {B.n2i[g] for g in B.excl}      # inductive: held-out queries banned
    B.q_train = np.where(~np.isin(B.q_gidx, list(hgi)))[0]

    # ---- val/train eval-query row indices used by heads + selection ----
    B.va_neu = [i for i, g in enumerate(B.art_games)
                if g in B.val_g and B.variants[i] == "neutral"]
    B.va_non = [i for i, g in enumerate(B.art_games)
                if g in B.val_g and B.variants[i] == "noname"]
    B.tr_neu = [i for i, g in enumerate(B.art_games)
                if g in B.train_doc_g and B.variants[i] == "neutral"]
    return B
