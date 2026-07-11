# -*- coding: utf-8 -*-
"""SP corpus step 2: fill sp_neutral/ gaps — LLM NEUTRAL rewrite of every
sp_raw store page that has no rewrite yet (same prompt + elaboration-retry
as gen_store_neutral.py). Resume by file existence; atomic writes."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR = SCRIPT_DIR
import os as _os
# data/code separation: corpora TEXT lives under the data root, not the repo
SCRIPT_DIR = (Path(_os.environ["LARICE_CORPORA"]) if _os.environ.get("LARICE_CORPORA")
              else Path(__file__).resolve().parents[2] / "data" / "corpora")
sys.path.insert(0, str(_CODE_DIR))
from generate_text_variants import chat, SYSTEM_PROMPTS  # noqa: E402

import json  # noqa: E402

RAW = SCRIPT_DIR / "sp_raw"
DST = SCRIPT_DIR / "sp_neutral"
MIN_OUT, MAX_CHARS, WORKERS = 300, 12000, 8


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def one(appid: str, name: str) -> str:
    out = DST / f"{appid}.txt"
    if out.exists():
        return "skip"
    desc = (RAW / f"{appid}.txt").read_text(encoding="utf-8", errors="ignore")
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
    games = json.loads((SCRIPT_DIR.parent / "game_review_data" / "games.json")
                       .read_text(encoding="utf-8"))
    todo = []
    for f in sorted(RAW.glob("*.txt")):
        appid = f.stem
        if not (DST / f"{appid}.txt").exists():
            todo.append((appid, (games.get(appid) or {}).get("name") or appid))
    print(f"sp_neutral gap: {len(todo)} games", flush=True)

    t0 = time.time()
    ok = err = done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(one, a, nm): a for a, nm in todo}
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
    n_total = len(list(DST.glob("*.txt")))
    json.dump({"generated_ok": ok, "err": err, "total_files": n_total},
              open(SCRIPT_DIR / "sp_neutral_manifest.json", "w"), indent=1)
    print(f"sp_neutral done: generated {ok}, err {err}, total on disk {n_total}",
          flush=True)


if __name__ == "__main__":
    main()
