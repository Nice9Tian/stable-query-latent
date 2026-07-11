# -*- coding: utf-8 -*-
"""The full contrast roster (CURRENT design only — R60 wiki protocol).

Arm = ArmSpec (tower recipe + gate + head grid). The champion tower lives in
model/ + steam_reviews_framework; every OTHER philosophy and dose lives here.

Factorised axes:
  loss family     I-CE (ice/i2ce) | pure CE | ArcFace | BYOL
  CE gate         cegate1/2/3/4 (I dose), scope doc vs wiki (…w)
  I gate          igate1 / igate1w (CE always on, I gated)
  gate control    rgate2 (random coverage-matched CE gate)
  doc ablations   nodoc (no doc views), *_wllm (wiki_llm doc source —
                  the pretraining-leak ablation, applied to CE/ARC/BYOL
                  single-constraint towers and the champion)
"""
from steam_reviews_framework.train import ArmSpec

from ..contrast_heads.configs import (HEAD_ARC, HEAD_BYOL, HEAD_CE, HEAD_ICE,
                                      HEAD_ICE_P1CE)


def _arm(name, **kw):
    return ArmSpec(name=name, **kw)


ARMS = {
    # ---- loss families (ungated) ----
    "ice": _arm("ice", tower="plain", inv_weight=1.0, head_cfgs=HEAD_ICE_P1CE),
    "i2ce": _arm("i2ce", tower="plain", inv_weight=2.0, head_cfgs=HEAD_ICE_P1CE),
    "ce": _arm("ce", tower="plain", inv_weight=0.0, head_cfgs=HEAD_CE),
    "arc": _arm("arc", tower="arc", inv_weight=0.0, head_cfgs=HEAD_ARC),
    "byol": _arm("byol", tower="byol", inv_weight=0.0, head_cfgs=HEAD_BYOL),
    # ---- CE-gated I-dose ladder (champion = cegate2, run via path 1) ----
    "cegate1": _arm("cegate1", tower="cegate", inv_weight=1.0),
    "cegate3": _arm("cegate3", tower="cegate", inv_weight=3.0),
    "cegate4": _arm("cegate4", tower="cegate", inv_weight=4.0),
    "cegate1w": _arm("cegate1w", tower="cegate", inv_weight=1.0,
                     gate_scope="wiki"),
    "cegate2w": _arm("cegate2w", tower="cegate", inv_weight=2.0,
                     gate_scope="wiki"),
    # ---- I-gated (mirror hypothesis) ----
    "igate1": _arm("igate1", tower="igate", inv_weight=1.0),
    "igate1w": _arm("igate1w", tower="igate", inv_weight=1.0,
                    gate_scope="wiki"),
    # ---- controls ----
    "rgate2": _arm("rgate2", tower="rgate", inv_weight=2.0),
    "nodoc": _arm("nodoc", tower="plain", inv_weight=2.0, use_doc_view=False),
    # ---- pretraining-leak ablation (wiki_llm paraphrase docs) ----
    "cegate2_wllm": _arm("cegate2_wllm", tower="cegate", inv_weight=2.0,
                         wiki_src="llm"),
    "ce_wllm": _arm("ce_wllm", tower="plain", inv_weight=0.0, wiki_src="llm",
                    head_cfgs=HEAD_CE),
    "arc_wllm": _arm("arc_wllm", tower="arc", inv_weight=0.0, wiki_src="llm",
                     head_cfgs=HEAD_ARC),
    "byol_wllm": _arm("byol_wllm", tower="byol", inv_weight=0.0,
                      wiki_src="llm", head_cfgs=HEAD_BYOL),
}

# 5-fold CV recipes (fixed-split winners re-adjudicated out-of-fold)
CV_RECIPES = ["champion_cegate2", "ice", "i2ce", "ce", "arc", "byol"]
