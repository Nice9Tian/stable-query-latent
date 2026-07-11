# -*- coding: utf-8 -*-
"""larice — Latent Represent I-CE: the champion tower + champion loss
(architecture lineage: SetPoolN — N latent queries cross-attending a set).

Tensor protocol (see README): the leading two axes of every input are
[data, view] —
    x:    [B, V, S, D_in]   float; S = set elements (e.g. sentences),
                            D_in = upstream embedding dim
    mask: [B, V, S]         bool, True = PADDING (torch key_padding_mask)
    out:  [B, V, out_dim]   L2-normalised
Single-view tasks use V = 1. A rank-3 x [B, S, D_in] is accepted as a
convenience and treated as V = 1 with the view axis squeezed away.

Loss axis semantics: the invariance term I is reduced along the *view*
axis; the CE term is reduced along the *data* axis (against an anchor
gallery). The CE gate is a per-item bool mask — CE fires only on gated
items, I always fires. That factorisation is the champion recipe.
"""
import math
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LariceConfig


class LariceTower(nn.Module):
    """N learnable latent queries cross-attend over an embedding set."""

    def __init__(self, cfg: LariceConfig | None = None, **kw):
        super().__init__()
        self.cfg = cfg or LariceConfig(**kw)
        c = self.cfg
        self.q0 = nn.Parameter(torch.randn(1, c.num_queries, c.dim_model) * 0.02)
        self.attn = nn.MultiheadAttention(c.dim_model, c.num_heads,
                                          kdim=c.input_dim, vdim=c.input_dim,
                                          batch_first=True)
        self.head = nn.Sequential(nn.Linear(c.dim_model, c.hidden), nn.GELU(),
                                  nn.Linear(c.hidden, c.dim_model))
        if c.tau_mode == "learnable":
            self.log_inv_tau = nn.Parameter(
                torch.tensor(math.log(1.0 / c.tau), dtype=torch.float32))

    @property
    def inv_tau(self) -> torch.Tensor:
        if self.cfg.tau_mode == "learnable":
            return self.log_inv_tau.exp()
        return torch.tensor(1.0 / self.cfg.tau)

    def encode_set(self, x, mask=None):
        """x [B, S, D_in], mask [B, S] -> [B, out_dim] L2-normalised."""
        a, _ = self.attn(self.q0.expand(x.shape[0], -1, -1),
                         x.float(), x.float(),
                         key_padding_mask=mask, need_weights=False)
        if self.cfg.readout == "pool":                 # champion / name recall
            return F.normalize(self.head(a.mean(1)), dim=-1)
        z = self.head(a)                               # per-slot readout
        return F.normalize(z.flatten(1), dim=-1)       # concat N slots

    def forward(self, x, mask=None):
        if x.dim() == 3:                               # [B, S, D] -> V = 1
            return self.encode_set(x, mask)
        B, V, S, D = x.shape                           # [B, V, S, D]
        z = self.encode_set(x.reshape(B * V, S, D),
                            None if mask is None else mask.reshape(B * V, S))
        return z.reshape(B, V, -1)


# --------------------------- champion loss ---------------------------------

def invariance_loss(z):
    """I term: mean cosine misalignment over all view pairs.

    z: [B, V, D] (or a list of V tensors [B, D]). Reduced along the view
    axis; returns a scalar. V = 1 yields 0.
    """
    zs = list(z.unbind(1)) if torch.is_tensor(z) else list(z)
    pairs = list(combinations(range(len(zs)), 2))
    if not pairs:
        return torch.zeros((), device=zs[0].device)
    return sum((1 - (zs[i].float() * zs[j].float()).sum(-1)).mean()
               for i, j in pairs) / len(pairs)


def gated_ce_loss(z, gallery, targets, inv_tau, gate=None):
    """CE term against an anchor gallery, per view, gate-masked on data axis.

    z: [B, V, D] or list of [B, D]; gallery: [G, D] (rows L2-normalised);
    targets: [B] gallery column of each item; gate: [B] bool or None
    (None = CE on every item). Returns a scalar (sum over views).
    """
    zs = list(z.unbind(1)) if torch.is_tensor(z) else list(z)
    if gate is not None:
        hd = gate.nonzero(as_tuple=True)[0]
        if len(hd) == 0:
            return torch.zeros((), device=zs[0].device)
        return sum(F.cross_entropy(zv.float()[hd] @ gallery.T.float() * inv_tau,
                                   targets[hd]) for zv in zs)
    return sum(F.cross_entropy(zv.float() @ gallery.T.float() * inv_tau,
                               targets) for zv in zs)


def champion_loss(z, gallery, targets, cfg: LariceConfig, gate=None,
                  inv_tau=None):
    """Full champion objective: gated CE (data axis) + I (view axis)."""
    it = inv_tau if inv_tau is not None else 1.0 / cfg.tau
    loss = gated_ce_loss(z, gallery, targets, it, gate)
    if cfg.inv_weight > 0:
        loss = loss + cfg.inv_weight * invariance_loss(z)
    return loss
