# -*- coding: utf-8 -*-
"""ONE-CLICK champion reproduction.

    python steam_reviews_framework/run.py [--epochs 1000] [--device cuda]

What it does, in order:
  1. corpora   — unpack the BUNDLED wikipage.zip / storepage.zip into the
                 data root. The bundles are always preferred: nothing is
                 re-scraped from Wikipedia and nothing is re-generated
                 through an LLM, so results cannot drift with live wiki
                 edits or non-deterministic rewrites.
  2. reviews   — the heavy review-embedding files are fetched on demand
                 (LARICE_EMBED_H5_URL / LARICE_TEXT_H5_URL), or you build
                 them once from the Kaggle dump via data_pipeline/reviews/.
  3. assets    — training/eval tensors built by data_pipeline/build_assets.
  4. train     — the champion tower (cegate2) + BackHeads + vsel selection.

Every step is resume-safe; rerunning skips whatever already exists.
contrast_experiment/run.py (the full contrast suite) reuses steps 1-3.
"""
import argparse
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

FRAMEWORK = Path(__file__).resolve().parent
RELEASE = FRAMEWORK.parent
sys.path.insert(0, str(RELEASE))

from data_pipeline.paths import ASSETS, CORPORA, EMBED_H5, TEXT_H5

BUNDLES = {                       # bundle -> corpora dirs it provides
    "wikipage.zip": ("wiki_clean", "wiki_variants", "wiki_llm"),
    "storepage.zip": ("sp_raw", "sp_neutral", "sp_llm", "sp_positive",
                      "sp_negative", "sp_noname"),
}
ASSET_FILES = ("games.npz", "wiki_clean_views.npz", "wiki_llm_views.npz",
               "sp_raw_views.npz", "wscan_pool_rev.npy",
               "wscan_pool_rev_rid.npy", "wscan_gal_rev.npz",
               "ss_queries_rev.npz", "ss_queries_rev_S.npy", "wiki_eval.npz")


def ensure_corpora():
    """Bundled text corpora take absolute priority — never re-collect."""
    for zname, dirs in BUNDLES.items():
        missing = [d for d in dirs
                   if not (CORPORA / d).exists() or not any((CORPORA / d).iterdir())]
        if not missing:
            print(f"corpora [{zname}]: present — skip", flush=True)
            continue
        zp = FRAMEWORK / "corpora_bundles" / zname
        assert zp.exists(), f"bundle missing from the repo: {zp}"
        CORPORA.mkdir(parents=True, exist_ok=True)
        print(f"corpora [{zname}]: unpacking ({', '.join(missing)}) ...",
              flush=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(CORPORA)
    print("corpora ready (bundled texts; wiki NOT re-scraped, LLM NOT re-run)",
          flush=True)


def _download(url, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    print(f"downloading {url}\n  -> {dst}", flush=True)
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(64 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"  {done / 1e9:.1f} / {total / 1e9:.1f} GB",
                          flush=True)
        tmp.replace(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def ensure_reviews():
    """The heavy layer: 73M-sentence embedding h5 + tag/text h5."""
    ok = True
    for p, env in ((EMBED_H5, "LARICE_EMBED_H5_URL"),
                   (TEXT_H5, "LARICE_TEXT_H5_URL")):
        if p.exists():
            print(f"reviews: {p.name} present — skip", flush=True)
            continue
        url = os.environ.get(env, "")
        if url:
            _download(url, p)
        else:
            ok = False
            print(f"reviews: MISSING {p}\n"
                  f"  option A: set {env} to a download source and rerun\n"
                  f"  option B: build once from the Kaggle dump —\n"
                  f"    data_pipeline/reviews/: prepare_kaggle_steam_reviews "
                  f"-> build.py -> Build_new.py (embed) -> h5_corpus.py",
                  flush=True)
    if not ok:
        sys.exit(1)


def ensure_assets():
    missing = [f for f in ASSET_FILES if not (ASSETS / f).exists()]
    if not missing:
        print("assets: all present — skip", flush=True)
        return
    print(f"assets: building {len(missing)} missing ...", flush=True)
    subprocess.check_call([sys.executable,
                           str(RELEASE / "data_pipeline" / "build_assets.py")])


def ensure_data():
    """Steps 1-3 (shared with contrast_experiment/run.py)."""
    ensure_corpora()
    ensure_reviews()
    ensure_assets()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-only", action="store_true",
                    help="prepare corpora/reviews/assets, skip training")
    args = ap.parse_args()
    ensure_data()
    if args.data_only:
        print("data ready — stopping before training (--data-only)")
        return
    subprocess.check_call([sys.executable,
                           str(FRAMEWORK / "train_champion.py"),
                           "--epochs", str(args.epochs),
                           "--ckpt-every", str(args.ckpt_every),
                           "--device", args.device])


if __name__ == "__main__":
    main()
