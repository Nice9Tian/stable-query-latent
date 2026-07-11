# -*- coding: utf-8 -*-
"""Gallery (anchor-set) encodings through a tower.

The anchor of a game = sp_clean store text FIRST, then whole reviews, to a
fixed 512-sentence budget. Three gallery flavours:
  gallery        full 2020 games, doc-bearing anchors (eval / projection)
  gallery_nodoc  full 2020 games, doc prefix masked (reviews-only anchors,
                 used by the head's doc-gating augmentation; no grad)
  gallery_train  ONLY the train games, gradients on — the training-time
                 CE negative set under the fully-inductive protocol
"""
import torch


def gallery(model, B, chunk=256, grad=False):
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(0, B.NG, chunk):
            outs.append(model(B.SGal[i:i + chunk], B.mGal[i:i + chunk]))
    return torch.cat(outs)


def gallery_nodoc(model, B, chunk=256):
    outs = []
    with torch.no_grad():
        for i in range(0, B.NG, chunk):
            outs.append(model(B.SGal[i:i + chunk], B.mGal_nd[i:i + chunk]))
    return torch.cat(outs)


def gallery_train(model, B, chunk=256, grad=True):
    """CE targets must use B.pos_of_g[game_idx] positions in this gallery."""
    rows = B.train_pool_games
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(0, len(rows), chunk):
            r = torch.as_tensor(rows[i:i + chunk], device=B.dev)
            outs.append(model(B.SGal[r], B.mGal[r]))
    return torch.cat(outs)
