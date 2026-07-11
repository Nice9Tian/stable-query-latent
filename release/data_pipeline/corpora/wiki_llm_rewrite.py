# -*- coding: utf-8 -*-
"""Faithful PARAPHRASE rewrite of the validated wiki corpus (user spec):
training-side text must come from a DIFFERENT distribution than the eval-side
neutral summaries, so generalization is real rather than style-fitting.

Pipeline: wiki_clean/<file> -> drop section-header lines -> SaT sentence split
(CPU) -> chunks of 10 sentences -> LLM per chunk with the content-preserving
prompt ("保留完整内容, 优化语言表达") -> re-concatenate -> wiki_para/<appid>.txt.
Doc-level resume; empty-output retry per chunk.
"""
import re
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
from generate_text_variants import chat  # noqa: E402

SRC = SCRIPT_DIR / "wiki_clean"
DST = SCRIPT_DIR / "wiki_llm"          # codename wiki_llm (formerly wiki_para)
DST.mkdir(exist_ok=True)
CHUNK, WORKERS, MIN_OUT_RATIO = 10, 8, 0.5

PARA_SYSTEM = (
    "You are a text rewriting assistant. Rewrite the passage the user gives you, "
    "PRESERVING THE COMPLETE CONTENT — every fact, name, and detail must remain — "
    "while improving the wording and flow (优化语言表达). Do NOT summarize, do NOT "
    "omit information, do NOT add new information, do NOT change the meaning. "
    "Output only the rewritten passage."
)

HDR = re.compile(r"^=+\s*.+?\s*=+\s*$")


def make_splitter():
    try:
        from wtpsplit import SaT
        sat = SaT("sat-3l-sm")          # CPU on purpose (GPU busy with training)
        print("splitter: SaT sat-3l-sm (cpu)", flush=True)
        return lambda text: sat.split(text)
    except Exception as e:
        print(f"splitter: regex fallback ({type(e).__name__})", flush=True)
        rx = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
        return lambda text: rx.split(text)


SPLIT = None


def para_chunk(text: str) -> str:
    best = ""
    for _ in range(3):
        out = chat(PARA_SYSTEM, text) or ""
        if len(out) > len(best):
            best = out
        if len(best) >= MIN_OUT_RATIO * len(text):
            break
    return best


def one(f: Path) -> str:
    appid = f.name.split("_")[0]
    out = DST / f"{appid}.txt"
    if out.exists():
        return "skip"
    body = "\n".join(ln for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines()
                     if not HDR.match(ln))
    sents = [s.strip() for s in SPLIT(body) if len(s.strip()) >= 10]
    if len(sents) < 3:
        return "thin"
    parts = []
    for k in range(0, len(sents), CHUNK):
        chunk = " ".join(sents[k:k + CHUNK])
        r = para_chunk(chunk)
        if len(r) < MIN_OUT_RATIO * len(chunk):
            return "chunk-fail"        # no partial file; doc retried on resume
        parts.append(r)
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n\n".join(parts), encoding="utf-8")
    tmp.replace(out)
    return "ok"


def main() -> None:
    global SPLIT
    SPLIT = make_splitter()
    files = sorted(SRC.glob("*.txt"))
    todo = [f for f in files if not (DST / f"{f.name.split('_')[0]}.txt").exists()]
    print(f"{len(files)} docs; {len(todo)} to paraphrase", flush=True)
    t0 = time.time()
    ok = err = done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(one, f): f for f in todo}
        for fut in as_completed(futs):
            f = futs[fut]
            done += 1
            try:
                r = fut.result()
                ok += (r == "ok")
                if r in ("chunk-fail",):
                    err += 1
            except Exception as e:
                err += 1
                print(f"  [err] {f.name}: {type(e).__name__}: {e}", flush=True)
            if done % 20 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} (ok {ok}, err {err}) [{time.time()-t0:.0f}s]",
                      flush=True)
    print(f"all done: ok {ok}, err {err}", flush=True)


if __name__ == "__main__":
    main()
