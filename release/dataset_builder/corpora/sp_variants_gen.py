# -*- coding: utf-8 -*-
"""SP corpus completion: sp_positive / sp_negative / sp_noname for ALL 1,811
store-page games (user decree: full symmetry with the wiki side).

Seeds: text_variants_generated/<appid>/{positive,negative,noname}.txt (the 258
old-eval games, same SYSTEM_PROMPTS + same user-message format) — copied by
seed_all() before generation. Gap (~1,553 per variant) generated via API with
elaboration-retry and a language guard (English source must yield English
output; CJK-majority sources may answer in kind).
Resume by file existence; atomic writes; manifest on exit."""
import json
import re
import shutil
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

RAW = SCRIPT_DIR / "sp_raw"
SEED = SCRIPT_DIR / "text_variants_generated"
SEED_DEP = SCRIPT_DIR / "_deprecated" / "text_variants_generated"
VARIANTS = ("positive", "negative", "noname")
MIN_OUT, MAX_CHARS, WORKERS = 300, 12000, 8
CJK = re.compile(r"[一-鿿]")

for v in VARIANTS:
    (SCRIPT_DIR / f"sp_{v}").mkdir(exist_ok=True)


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def seed_all() -> int:
    src = SEED if SEED.exists() else SEED_DEP
    n = 0
    if not src.exists():
        print("seed dir not found — skipping seed phase", flush=True)
        return 0
    for d in src.iterdir():
        if not d.is_dir():
            continue
        for v in VARIANTS:
            s, t = d / f"{v}.txt", SCRIPT_DIR / f"sp_{v}" / f"{d.name}.txt"
            if s.exists() and not t.exists():
                shutil.copyfile(s, t)
                n += 1
    return n


def one(appid: str, name: str, variant: str) -> str:
    out = SCRIPT_DIR / f"sp_{variant}" / f"{appid}.txt"
    if out.exists():
        return "skip"
    desc = (RAW / f"{appid}.txt").read_text(encoding="utf-8", errors="ignore")
    src_cjk = len(CJK.findall(desc)) > 0.05 * len(desc)
    user_msg = f"Game: {name}\n\nFull store page text:\n{desc[:MAX_CHARS]}"
    best = ""
    for _ in range(4):
        text = chat(SYSTEM_PROMPTS[variant], user_msg) or ""
        if not src_cjk and CJK.search(text):
            continue                      # language drift on an English source
        if len(text) > len(best):
            best = text
        if len(best) >= MIN_OUT:
            break
        user_msg += ("\n\nYour previous answer was too short. Please write a "
                     "considerably MORE DETAILED article (at least 300 words) "
                     "in the same style as instructed.")
    if len(best) < MIN_OUT:
        return "short-output"
    write_atomic(out, best)
    return "ok"


def main() -> None:
    n_seed = seed_all()
    print(f"seeded {n_seed} files from the 258-set", flush=True)
    games = json.loads((SCRIPT_DIR.parent / "game_review_data" / "games.json")
                       .read_text(encoding="utf-8"))
    todo = []
    for f in sorted(RAW.glob("*.txt")):
        appid = f.stem
        nm = (games.get(appid) or {}).get("name") or appid
        for v in VARIANTS:
            if not (SCRIPT_DIR / f"sp_{v}" / f"{appid}.txt").exists():
                todo.append((appid, nm, v))
    print(f"gap: {len(todo)} rewrites across {len(VARIANTS)} variants", flush=True)

    t0 = time.time()
    ok = err = done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(one, a, nm, v): (a, v) for a, nm, v in todo}
        for fut in as_completed(futs):
            a, v = futs[fut]
            done += 1
            try:
                r = fut.result()
                ok += (r == "ok")
                if r == "short-output":
                    err += 1
            except Exception as e:
                err += 1
                print(f"  [err] {a}/{v}: {type(e).__name__}: {e}", flush=True)
            if done % 100 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} (ok {ok}, err {err}) [{time.time()-t0:.0f}s]",
                      flush=True)
    counts = {v: len(list((SCRIPT_DIR / f"sp_{v}").glob("*.txt"))) for v in VARIANTS}
    json.dump({"generated_ok": ok, "err": err, "totals": counts},
              open(SCRIPT_DIR / "sp_variants_manifest.json", "w"), indent=1)
    print(f"done: generated {ok}, err {err}, totals {counts}", flush=True)


if __name__ == "__main__":
    main()
