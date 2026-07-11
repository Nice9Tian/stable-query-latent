# -*- coding: utf-8 -*-
"""SP corpus step 1 (no API): materialize sp_raw/ and seed sp_neutral/.

User spec (2026-07-11): over the 1,811 games whose store description is
>=300 chars, build three parallel corpora:
  sp_raw/<appid>.txt      — store original text (HTML-stripped)   [this script]
  sp_neutral/<appid>.txt  — LLM neutral rewrite of the store page [seeded here,
                            gaps filled by sp_neutral_gen.py]
  sp_llm/<appid>.txt      — sentence-wise faithful paraphrase     [sp_llm_rewrite.py]

sp_neutral seeds (same SYSTEM_PROMPTS['neutral'] + same user-message format):
  text_variants_generated/<appid>/neutral.txt  (the 258-game eval set)
  store_neutral/<appid>.txt                    (the 858 coverage rewrites)
Writes sp_manifest.json: per-appid chars_raw + neutral seed source.
"""
import json
import re
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR = SCRIPT_DIR
import os as _os
# data/code separation: corpora TEXT lives under the data root, not the repo
SCRIPT_DIR = (Path(_os.environ["LARICE_CORPORA"]) if _os.environ.get("LARICE_CORPORA")
              else Path(__file__).resolve().parents[2] / "data" / "corpora")
ROOT = SCRIPT_DIR.parent
SC = (Path(_os.environ["LARICE_ASSETS"]) if _os.environ.get("LARICE_ASSETS")
      else Path(__file__).resolve().parents[2] / "data" / "assets")
MIN_IN = 300

RAW = SCRIPT_DIR / "sp_raw"
NEU = SCRIPT_DIR / "sp_neutral"
RAW.mkdir(exist_ok=True)
NEU.mkdir(exist_ok=True)


def strip_html(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t or "")).strip()


def main() -> None:
    import numpy as np
    G = np.load(SC / "games.npz", allow_pickle=True)
    names = [str(x) for x in G["names"]]
    games = json.loads((SCRIPT_DIR.parent / "reviews" / "games.json").read_text(encoding="utf-8"))

    manifest = {}
    n_seed258 = n_seed858 = 0
    for n in names:
        appid = n.split("_")[0]
        meta = games.get(appid) or {}
        desc = strip_html(meta.get("detailed_description") or meta.get("about_the_game")
                          or meta.get("short_description") or "")
        if len(desc) < MIN_IN:
            continue
        (RAW / f"{appid}.txt").write_text(desc, encoding="utf-8")
        rec = {"chars_raw": len(desc), "neutral_src": None}
        # ---- seed sp_neutral (never overwrite an existing file) ----
        dst = NEU / f"{appid}.txt"
        if not dst.exists():
            s258 = SCRIPT_DIR / "text_variants_generated" / appid / "neutral.txt"
            s858 = SCRIPT_DIR / "store_neutral" / f"{appid}.txt"
            if s258.exists():
                shutil.copyfile(s258, dst)
                rec["neutral_src"] = "seed-258"
                n_seed258 += 1
            elif s858.exists():
                shutil.copyfile(s858, dst)
                rec["neutral_src"] = "seed-858"
                n_seed858 += 1
        else:
            rec["neutral_src"] = "existing"
        manifest[appid] = rec

    json.dump(manifest, open(SCRIPT_DIR / "sp_manifest.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    missing = [a for a, r in manifest.items() if r["neutral_src"] is None]
    print(f"sp_raw: {len(manifest)} files | sp_neutral seeded: "
          f"{n_seed258} from 258-set + {n_seed858} from store_neutral | "
          f"neutral gap for sp_neutral_gen.py: {len(missing)}", flush=True)


if __name__ == "__main__":
    main()
