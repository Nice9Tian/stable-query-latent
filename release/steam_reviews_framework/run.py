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
                 them once from the Kaggle dump via dataset_builder/reviews/.
  3. assets    — training/eval tensors built by dataset_builder/build_assets.
  4. train     — the champion tower (cegate2) + BackHeads + vsel selection.

Every step is resume-safe; rerunning skips whatever already exists.
contrast_experiment/run.py (the full contrast suite) reuses steps 1-3.
"""

# ═════════════════════ API SETTINGS — FILL IN HERE ═════════════════════════
# Two ways to provide each value: type it right here, or put it in the
# credential files under dataset_builder/ (embeddingAPI.txt / llmAPI.txt,
# copied from the *.template.txt next to them). Either source alone works;
# if BOTH are set and disagree, a NOTE is printed and the in-code value
# wins (one side being empty is NOT a mismatch).
API_EMBEDDING_BASEURL = ""   # https://<endpoint>.huggingface.cloud (cloud embedding; blank = local GPU)
API_EMBEDDING_TOKEN = ""     # hf_xxx...
API_LLM_BASEURL = ""         # https://<gateway>/v1  (ONLY for regenerating corpora — normal runs use the bundled zips)
API_LLM_TOKEN = ""           # sk-...
API_LLM_MODEL = ""           # e.g. gpt-5.4-mini
API_EMBED_H5_URL = ""        # download source for reviews/embedding_h5.h5 (~164 GB)
API_TEXT_H5_URL = ""         # download source for reviews/text_h5.h5
# ════════════════════════════════════════════════════════════════════════════

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

from dataset_builder.paths import ASSETS, CORPORA, EMBED_H5, TEXT_H5


def _read_kv(path: Path) -> dict:
    kv = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                kv[k.strip().lower()] = v.strip()
    return kv


def apply_api_settings():
    """Reconcile the in-code API block with the credential files and export
    the effective values as env vars (inherited by every subprocess).
    Mismatch rule: warn only when BOTH sources are non-empty and differ —
    the in-code value wins; one side missing is never a mismatch."""
    emb_file = RELEASE / "dataset_builder" / "embeddingAPI.txt"
    llm_file = RELEASE / "dataset_builder" / "llmAPI.txt"
    emb, llm = _read_kv(emb_file), _read_kv(llm_file)
    plan = [  # (in-code value, file value, file path, label, env var)
        (API_EMBEDDING_BASEURL, emb.get("url", ""), emb_file,
         "embedding base URL", "EMBEDDING_API_URL"),
        (API_EMBEDDING_TOKEN, emb.get("token", ""), emb_file,
         "embedding token", "EMBEDDING_API_TOKEN"),
        (API_LLM_BASEURL, llm.get("url", ""), llm_file,
         "LLM base URL", "LLM_API_URL"),
        (API_LLM_TOKEN, llm.get("token", ""), llm_file,
         "LLM token", "LLM_API_TOKEN"),
        (API_LLM_MODEL, llm.get("model", ""), llm_file,
         "LLM model", "LLM_API_MODEL"),
        (API_EMBED_H5_URL, os.environ.get("LARICE_EMBED_H5_URL", ""), None,
         "embedding_h5 URL", "LARICE_EMBED_H5_URL"),
        (API_TEXT_H5_URL, os.environ.get("LARICE_TEXT_H5_URL", ""), None,
         "text_h5 URL", "LARICE_TEXT_H5_URL"),
    ]
    for code_v, file_v, src, label, env in plan:
        code_v = code_v.strip()
        if code_v and file_v and code_v != file_v:
            where = src if src is not None else f"env {env}"
            print(f"NOTE: {label} set in run.py differs from {where} — "
                  f"using the in-code value.", flush=True)
        if code_v:
            os.environ[env] = code_v

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
                  f"    dataset_builder/reviews/: prepare_kaggle_steam_reviews "
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
                           str(RELEASE / "dataset_builder" / "build_assets.py")])


def ensure_data():
    """Steps 1-3 (shared with contrast_experiment/run.py)."""
    apply_api_settings()
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
