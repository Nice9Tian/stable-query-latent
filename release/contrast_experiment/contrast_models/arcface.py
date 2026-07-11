# -*- coding: utf-8 -*-
"""Contrast tower: pure ArcFace (margin philosophy, no invariance term).

Tower-level scale s = 50 matches the CE arms' frozen 1/tau = 50, so
ARC-vs-CE differs ONLY by the additive angular margin m.
"""
import math

import torch
import torch.nn.functional as F

ARC_S_T, ARC_M_T = 50.0, 0.2


def arcface_ce(logits, tgt):
    """Additive angular-margin CE on cosine logits (target column gets
    cos(theta+m); numerically-stable fallback past pi-m), fixed scale.
    Forced float32: autocast downcasts the matmul while trig stays fp32."""
    logits = logits.float()
    cos_t = logits.gather(1, tgt[:, None]).clamp(-1 + 1e-7, 1 - 1e-7)
    phi = torch.cos(torch.acos(cos_t) + ARC_M_T)
    phi = torch.where(cos_t > math.cos(math.pi - ARC_M_T),
                      phi, cos_t - ARC_M_T * math.sin(ARC_M_T))
    return F.cross_entropy(logits.scatter(1, tgt[:, None], phi) * ARC_S_T, tgt)


def arc_loss_hook(Zs, Zg, tgt, gate_idx=None):
    """Drop-in loss_hook for steam_reviews_framework.train.train_tower."""
    return sum(arcface_ce(Z.float() @ Zg.T.float(), tgt) for Z in Zs)
