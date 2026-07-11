# -*- coding: utf-8 -*-
"""Review-level rejection sampling + padding utilities.

Rejection design: whole reviews only (never truncated); acceptance
probability a(L) = 0.2 + 0.7*(L-Lmin)/(Lmax-Lmin) per game — the game's
shortest review is accepted 20% of the time, its longest 90%, linear in
sentence count. Long "gold-mine" reviews therefore stay reachable while
short spam does not dominate.
"""
import numpy as np
import torch


def accept_prob(lens, lmin, lmax):
    if lmax <= lmin:
        return np.full_like(np.asarray(lens, np.float64), 0.9)
    return 0.2 + 0.7 * (np.asarray(lens, np.float64) - lmin) / (lmax - lmin)


def build_review_table(rid):
    """Per-game (starts, lens, accept_prob) from the pool's review-id rows."""
    tab = []
    for g in range(rid.shape[0]):
        r = rid[g]
        n = int(r.max()) + 1
        lens = np.bincount(r[r >= 0], minlength=n).astype(np.int64)
        starts = np.zeros(n, np.int64)
        starts[1:] = np.cumsum(lens)[:-1]
        tab.append((starts, lens, accept_prob(lens, lens.min(), lens.max())))
    return tab


def sample_views(pool, rev_tab, gids, W, rng, device):
    """One review-level view per game: rejection-draw WHOLE reviews until
    >= W sentences (the last review is never truncated). Returns a padded
    fp16 tensor [B, S, D] + padding mask [B, S]."""
    blocks = []
    for g in gids:
        st, ln, a = rev_tab[int(g)]
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
        blocks.append(np.asarray(pool[int(g)][seg]))
    LM = max(len(b) for b in blocks)
    out = np.zeros((len(blocks), LM, blocks[0].shape[-1]), np.float16)
    lens = np.zeros(len(blocks), np.int64)
    for k, b in enumerate(blocks):
        out[k, :len(b)] = b
        lens[k] = len(b)
    S = torch.from_numpy(out).to(device, non_blocking=True)
    m = torch.arange(LM, device=device)[None, :] >= \
        torch.as_tensor(lens, device=device)[:, None]
    return S, m


def pad_flat(Sflat, off, idx, device):
    """Materialize flat-stored (concat + offsets) queries into a padded
    batch + mask — flat storage because one query may hold a 1000-sentence
    gold-mine review and global padding would explode."""
    segs = [Sflat[off[j]:off[j + 1]] for j in idx]
    LM = max(len(s) for s in segs)
    X = np.zeros((len(segs), LM, segs[0].shape[-1]), np.float16)
    L = np.zeros(len(segs), np.int64)
    for k, s in enumerate(segs):
        X[k, :len(s)] = s
        L[k] = len(s)
    Xt = torch.from_numpy(X).to(device)
    m = torch.arange(LM, device=device)[None, :] >= \
        torch.as_tensor(L, device=device)[:, None]
    return Xt, m
