# -*- coding: utf-8 -*-
"""Generate ALL FOUR LLM variants (neutral / positive / negative / noname) for
the filtered+validated wiki corpus (wiki_clean), so the expanded set supports
the full 4-rewrite metric protocol (user spec).

Layout mirrors text_variants_generated:  wiki_variants/<appid>/<variant>.txt
Reuses generate_text_variants' chat client + SYSTEM_PROMPTS. Per-variant
elaboration-retry (output < 300 chars -> append a "more detail" instruction and
resend, up to 3 attempts). Resumable per (game, variant). Existing neutral
rewrites from wiki_descriptions_neutral are migrated in (no re-calls).
GAME-major order: a game's missing variants fire concurrently, so interruption
leaves whole games rather than one variant everywhere.
"""
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

SRC = SCRIPT_DIR / "wiki_clean"
OLD_NEU = SCRIPT_DIR / "wiki_descriptions_neutral"
DST = SCRIPT_DIR / "wiki_variants"
DST.mkdir(exist_ok=True)
MIN_CHARS, MAX_CHARS, WORKERS = 300, 12000, 8
VARIANTS = list(SYSTEM_PROMPTS)          # neutral / positive / negative / noname


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def migrate_neutral() -> int:
    moved = 0
    for f in OLD_NEU.glob("*.txt"):
        appid = f.name.split("_")[0]
        out = DST / appid / "neutral.txt"
        if out.exists():
            continue
        out.parent.mkdir(exist_ok=True)
        text = f.read_text(encoding="utf-8", errors="ignore")
        if len(text) >= MIN_CHARS:
            write_atomic(out, text)
            moved += 1
    return moved


def gen_variant(appid: str, name: str, raw: str, variant: str) -> str:
    out = DST / appid / f"{variant}.txt"
    if out.exists():
        return "skip"
    user_msg = f"Game: {name}\n\nFull page text:\n{raw[:MAX_CHARS]}"
    best = ""
    for _ in range(3):
        text = chat(SYSTEM_PROMPTS[variant], user_msg) or ""
        if len(text) > len(best):
            best = text
        if len(best) >= MIN_CHARS:
            break
        user_msg += ("\n\nYour previous answer was too short. Please write a "
                     "considerably MORE DETAILED article (at least 300 words) "
                     "in the requested style, covering the game's content, "
                     "story, world and mechanics from the page text above.")
    if len(best) < MIN_CHARS:
        return "short-output"
    write_atomic(out, best)
    return "ok"


def main() -> None:
    moved = migrate_neutral()
    print(f"migrated {moved} existing neutral rewrites", flush=True)
    files = sorted(SRC.glob("*.txt"))
    jobs = []                                # (appid, name, raw, [variants])
    for f in files:
        appid = f.name.split("_")[0]
        name = f.stem.split("_", 1)[1] if "_" in f.stem else f.stem
        missing = [v for v in VARIANTS if not (DST / appid / f"{v}.txt").exists()]
        if not missing:
            continue
        raw = f.read_text(encoding="utf-8", errors="ignore").strip()
        if len(raw) < MIN_CHARS:
            continue
        (DST / appid).mkdir(exist_ok=True)
        jobs.append((appid, name, raw, missing))
    total = sum(len(j[3]) for j in jobs)
    print(f"{len(files)} games; {len(jobs)} with missing variants; "
          f"{total} calls to make", flush=True)
    t0 = time.time()
    ok = err = done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for appid, name, raw, missing in jobs:      # GAME-major
            futs = {pool.submit(gen_variant, appid, name, raw, v): v
                    for v in missing}
            for fut in as_completed(futs):
                v = futs[fut]
                done += 1
                try:
                    r = fut.result()
                    ok += (r == "ok")
                    if r == "short-output":
                        err += 1
                        print(f"  [short] {appid}/{v}", flush=True)
                except Exception as e:
                    err += 1
                    print(f"  [err] {appid}/{v}: {type(e).__name__}: {e}", flush=True)
                if done % 50 == 0 or done == total:
                    print(f"  {done}/{total} (ok {ok}, err {err}) "
                          f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"all done: ok {ok}, err {err}", flush=True)


if __name__ == "__main__":
    main()
