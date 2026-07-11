# -*- coding: utf-8 -*-
"""Full-coverage document corpus (mission round 7): generate NEUTRAL LLM
rewrites from Steam store descriptions (games.json) for every game that has
neither an eval article (the 258) nor a clean wiki page (the 844) — so that
(almost) every non-test/val game gets a document view.

Output: store_neutral/<appid>.txt. Elaboration-retry as per user spec.
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from generate_text_variants import chat, SYSTEM_PROMPTS  # noqa: E402

import numpy as np  # noqa: E402

DST = SCRIPT_DIR / "store_neutral"
DST.mkdir(exist_ok=True)
MIN_IN, MIN_OUT, MAX_CHARS, WORKERS = 300, 300, 12000, 8

SC = Path(r"C:/Users/admin/AppData/Local/Temp/claude/C--Users-admin-Documents-studable-query-latent"
          r"/f61c4010-5a2e-42bd-a560-64dc160587f6/scratchpad")


def strip_html(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t or "")).strip()


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def one(appid: str, name: str, desc: str) -> str:
    out = DST / f"{appid}.txt"
    if out.exists():
        return "skip"
    user_msg = f"Game: {name}\n\nFull store page text:\n{desc[:MAX_CHARS]}"
    best = ""
    for _ in range(3):
        text = chat(SYSTEM_PROMPTS["neutral"], user_msg) or ""
        if len(text) > len(best):
            best = text
        if len(best) >= MIN_OUT:
            break
        user_msg += ("\n\nYour previous answer was too short. Please write a "
                     "considerably MORE DETAILED descriptive article (at least "
                     "300 words) about the game's content, story, world and "
                     "mechanics from the page text above.")
    if len(best) < MIN_OUT:
        return "short-output"
    write_atomic(out, best)
    return "ok"


def main() -> None:
    G = np.load(SC / "fusion_cache" / "games.npz", allow_pickle=True)
    A = np.load(SC / "fusion_cache" / "articles.npz", allow_pickle=True)
    names = [str(x) for x in G["names"]]
    art_appids = {str(x).split("_")[0] for x in A["names"]}          # the 258 (incl. test/val)
    wiki_appids = {d.name for d in (SCRIPT_DIR / "wiki_variants").iterdir() if d.is_dir()}
    games = json.loads((ROOT / "game_review_data" / "games.json").read_text(encoding="utf-8"))

    todo = []
    for n in names:
        appid = n.split("_")[0]
        if appid in art_appids or appid in wiki_appids:
            continue                                                  # already covered
        meta = games.get(appid) or {}
        desc = strip_html(meta.get("detailed_description") or meta.get("about_the_game")
                          or meta.get("short_description") or "")
        if len(desc) < MIN_IN:
            continue
        todo.append((appid, meta.get("name") or appid, desc))
    print(f"games without any doc: targeting {len(todo)} store rewrites", flush=True)

    t0 = time.time()
    ok = err = done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(one, a, nm, d): a for a, nm, d in todo}
        for fut in as_completed(futs):
            a = futs[fut]
            done += 1
            try:
                r = fut.result()
                ok += (r == "ok")
                if r == "short-output":
                    err += 1
            except Exception as e:
                err += 1
                print(f"  [err] {a}: {type(e).__name__}: {e}", flush=True)
            if done % 50 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} (ok {ok}, err {err}) [{time.time()-t0:.0f}s]",
                      flush=True)
    print(f"all done: ok {ok}, err {err}", flush=True)


if __name__ == "__main__":
    main()
