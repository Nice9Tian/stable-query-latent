# -*- coding: utf-8 -*-
"""Contrast head grids — the two-phase BackHead-NAME mechanics live in
steam_reviews_framework.backhead_name; each contrast arm only declares WHICH loss
philosophy drives the two phases.

Row format: (suffix, phase1, phase2, label_smoothing)
  phase1/phase2 in {"ice", "ce", "by", "arc"}.
"""

HEAD_ICE = (("", "ice", "ice", 0.0),)             # champion + gated arms
HEAD_ICE_P1CE = (("", "ice", "ice", 0.0),         # + v4art-faithful pure-CE
                 ("_p1ce", "ce", "ice", 0.0))     #   warm-start ablation
HEAD_CE = (("", "ce", "ce", 0.0),)
HEAD_ARC = (("", "ce", "arc", 0.0),)
HEAD_BYOL = (("", "by", "by", 0.0),               # pure BYOL philosophy
             ("_ce", "ce", "ce", 0.0),            # raw CE head
             ("_cetf", "ce", "ce", 0.1))          # smoothed CE (blast guard)
