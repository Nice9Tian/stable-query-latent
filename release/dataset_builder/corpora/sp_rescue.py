# -*- coding: utf-8 -*-
"""Rescue pass for the SP variant corpus:
1. finish missing sp_positive / sp_negative files (relaxed length floor);
2. regenerate sp_noname files that leak the game's own name (reject-on-name
   retry x4; keeps the old file if no name-free candidate emerges)."""
import json
import re
import sys
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
MAX_CHARS = 12000
gm = json.loads((SCRIPT_DIR.parent / "game_review_data" / "games.json")
                .read_text(encoding="utf-8"))


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# ---- 1. missing files ----
for v in ("positive", "negative"):
    d = SCRIPT_DIR / f"sp_{v}"
    missing = [f.stem for f in RAW.glob("*.txt") if not (d / f"{f.stem}.txt").exists()]
    for a in missing:
        nm = (gm.get(a) or {}).get("name") or a
        desc = (RAW / f"{a}.txt").read_text(encoding="utf-8", errors="ignore")
        user_msg = f"Game: {nm}\n\nFull store page text:\n{desc[:MAX_CHARS]}"
        best = ""
        for _ in range(5):
            t = chat(SYSTEM_PROMPTS[v], user_msg) or ""
            if len(t) > len(best):
                best = t
            if len(best) >= 200:
                break
        if best:
            write_atomic(d / f"{a}.txt", best)
        print(f"[missing] sp_{v}/{a}: {len(best)} chars "
              f"({'ok' if best else 'FAIL'})", flush=True)

# ---- 2. name-leaking noname files ----
d = SCRIPT_DIR / "sp_noname"
leaks = []
for f in d.glob("*.txt"):
    nm = (gm.get(f.stem) or {}).get("name")
    if not nm or len(norm(nm)) < 4:
        continue
    if norm(nm) in norm(f.read_text(encoding="utf-8", errors="ignore")):
        leaks.append((f.stem, nm))
print(f"noname leaks to fix: {len(leaks)}", flush=True)
fixed = 0
for a, nm in leaks:
    desc = (RAW / f"{a}.txt").read_text(encoding="utf-8", errors="ignore")
    user_msg = (f"Game: {nm}\n\nFull store page text:\n{desc[:MAX_CHARS]}\n\n"
                f"IMPORTANT: your previous rewrite mentioned the game's title. The "
                f"title '{nm}' (or any recognizable part of it) must NOT appear "
                f"anywhere in your output. Refer to it only as 'this game' or with "
                f"an invented placeholder name.")
    got = ""
    for _ in range(4):
        t = chat(SYSTEM_PROMPTS["noname"], user_msg) or ""
        if len(t) >= 200 and norm(nm) not in norm(t):
            got = t
            break
    if got:
        write_atomic(d / f"{a}.txt", got)
        fixed += 1
        print(f"  [fixed] {a} ({nm[:30]})", flush=True)
    else:
        print(f"  [STILL-LEAKING] {a} ({nm[:30]}) — old file kept", flush=True)
print(f"rescue done: {fixed}/{len(leaks)} leaks fixed", flush=True)
