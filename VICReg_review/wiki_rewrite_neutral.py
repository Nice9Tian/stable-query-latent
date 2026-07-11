# -*- coding: utf-8 -*-
"""Rewrite the raw wiki descriptions into the neutral LLM register (the missing
pipeline step): wiki_descriptions/<appid>_<name>.txt -> wiki_descriptions_neutral/
same filename. Reuses generate_text_variants' chat client, retries and neutral
system prompt (API key stays in that one file). Resumable: existing outputs are
skipped. Only texts > 300 chars are rewritten (923 of 924).
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from generate_text_variants import chat, SYSTEM_PROMPTS  # noqa: E402

SRC = SCRIPT_DIR / "wiki_clean"          # filtered + match-validated corpus
DST = SCRIPT_DIR / "wiki_descriptions_neutral"
DST.mkdir(exist_ok=True)
MIN_CHARS, MAX_CHARS, WORKERS = 300, 12000, 8


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def one(f: Path) -> str:
    out = DST / f.name
    if out.exists():
        return "skip"
    raw = f.read_text(encoding="utf-8", errors="ignore").strip()
    if len(raw) < MIN_CHARS:
        return "thin"
    name = f.stem.split("_", 1)[1] if "_" in f.stem else f.stem
    user_msg = f"Game: {name}\n\nFull page text:\n{raw[:MAX_CHARS]}"
    # elaboration-retry loop (user spec): if the output is too short, append a
    # "be more detailed" instruction to the SAME prompt and resend, up to 3x.
    best = ""
    for attempt in range(3):
        text = chat(SYSTEM_PROMPTS["neutral"], user_msg) or ""
        if len(text) > len(best):
            best = text
        if len(best) >= MIN_CHARS:
            break
        user_msg += ("\n\nYour previous answer was too short. Please write a "
                     "considerably MORE DETAILED descriptive article (at least "
                     "300 words), covering the game's content, story, world and "
                     "mechanics from the page text above.")
    if len(best) < MIN_CHARS:
        return "short-output"
    write_atomic(out, best)
    return "ok"


def main() -> None:
    files = sorted(SRC.glob("*.txt"))
    todo = [f for f in files if not (DST / f.name).exists()]
    print(f"{len(files)} wiki texts; {len(files) - len(todo)} done; {len(todo)} to rewrite",
          flush=True)
    t0 = time.time()
    ok = err = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(one, f): f for f in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            f = futs[fut]
            try:
                r = fut.result()
                ok += (r == "ok")
            except Exception as e:
                err += 1
                print(f"  [err] {f.name}: {type(e).__name__}: {e}", flush=True)
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} (ok {ok}, err {err}) [{time.time()-t0:.0f}s]",
                      flush=True)
    print(f"all done: ok {ok}, err {err}", flush=True)


if __name__ == "__main__":
    main()
