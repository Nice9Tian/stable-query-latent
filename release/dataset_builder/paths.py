# -*- coding: utf-8 -*-
"""The DATA LAYER owns every location: heavy artefacts under one data root,
outside the code tree. Every location is overridable by environment
variable so an existing local layout can be linked in without copying.

  LARICE_DATA_ROOT   data root            (default: <release>/data)
  LARICE_ASSETS      built training/eval tensors (npz/npy)
  LARICE_RESULTS     checkpoints, projections, per-arm result jsons
  LARICE_CORPORA     text corpora (wiki_*, sp_*)
  LARICE_EMBED_H5    73M-sentence review-embedding h5
  LARICE_TEXT_H5     sentence-index + 23-tag-label h5
"""
import os
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]


def _p(env: str, default: Path) -> Path:
    return Path(os.environ[env]) if os.environ.get(env) else default


DATA = _p("LARICE_DATA_ROOT", RELEASE / "data")
ASSETS = _p("LARICE_ASSETS", DATA / "assets")
RESULTS = _p("LARICE_RESULTS", DATA / "results")
CORPORA = _p("LARICE_CORPORA", DATA / "corpora")
EMBED_H5 = _p("LARICE_EMBED_H5", DATA / "reviews" / "embedding_h5.h5")
TEXT_H5 = _p("LARICE_TEXT_H5", DATA / "reviews" / "text_h5.h5")

# The authoritative fixed split (seed 20260711, 204/203/407 over the 814
# clean wiki universe) ships WITH the data layer — it pins the protocol.
SPLIT_JSON = Path(__file__).resolve().parent / "wiki_eval_split.json"
