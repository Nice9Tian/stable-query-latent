# -*- coding: utf-8 -*-
"""One-click data rebuild: checks every layer, tells you exactly what is
missing and which command produces it, then builds what it can.

    python dataset_builder/rebuild_data.py           # report + build assets
    python dataset_builder/rebuild_data.py --check   # report only

Layers (top depends on bottom):
  assets   <- build_assets.py            (needs corpora + reviews h5)
  corpora  <- corpora/*.py               (needs games.json + LLM API)
  reviews  <- reviews/*.py               (needs the Kaggle dump + GPU)
Already-built artefacts are used as-is — link an existing layout via the
LARICE_* environment variables (see dataset_builder/paths.py) to skip rebuilds.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_builder.paths import ASSETS, CORPORA, EMBED_H5, TEXT_H5

DP = Path(__file__).resolve().parent

ASSET_FILES = ["games.npz", "wiki_clean_views.npz", "wiki_llm_views.npz",
               "sp_raw_views.npz", "wscan_pool_rev.npy",
               "wscan_pool_rev_rid.npy", "wscan_gal_rev.npz",
               "ss_queries_rev.npz", "ss_queries_rev_S.npy", "wiki_eval.npz"]
CORPORA_DIRS = ["wiki_clean", "wiki_variants", "wiki_llm", "sp_raw"]


def check():
    ok = True
    print("== reviews layer ==")
    for p, hint in ((EMBED_H5, "reviews/: prepare_kaggle -> build -> "
                     "Build_new.py (embed all sentences -> h5)"),
                    (TEXT_H5, "reviews/h5_corpus.py (sentence index + tags)")):
        exist = p.exists()
        ok &= exist
        print(f"  [{'OK' if exist else 'MISSING'}] {p}" +
              ("" if exist else f"\n      -> {hint}"))
    print("== corpora layer ==")
    for d in CORPORA_DIRS:
        n = len(list((CORPORA / d).glob("**/*.txt"))) if (CORPORA / d).exists() else 0
        good = n > 0
        ok &= good
        print(f"  [{'OK' if good else 'MISSING'}] {CORPORA / d} ({n} txt)" +
              ("" if good else "  -> corpora/: collect_wiki_descriptions -> "
               "build_wiki_clean -> wiki_variants_rewrite / wiki_llm_rewrite; "
               "build_sp_corpus (needs llmAPI.txt)"))
    print("== assets layer ==")
    missing = [f for f in ASSET_FILES if not (ASSETS / f).exists()]
    for f in ASSET_FILES:
        print(f"  [{'OK' if (ASSETS / f).exists() else 'MISSING'}] {ASSETS / f}")
    return ok, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only")
    a = ap.parse_args()
    lower_ok, missing = check()
    if a.check:
        return
    if not missing:
        print("\nall assets present — nothing to build")
        return
    if not lower_ok:
        print("\nlower layers incomplete — build them first (commands above), "
              "or point LARICE_* env vars at an existing layout")
        sys.exit(1)
    print(f"\nbuilding {len(missing)} missing assets ...")
    subprocess.check_call([sys.executable, str(DP / "build_assets.py")])


if __name__ == "__main__":
    main()
