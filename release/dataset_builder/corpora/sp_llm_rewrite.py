# -*- coding: utf-8 -*-
"""SP corpus step 3: sp_llm/ — sentence-wise FAITHFUL PARAPHRASE of every
sp_raw store page (all 1,811), same recipe as wiki_para_rewrite.py:
SaT sentence split -> chunks of 10 sentences -> LLM with the
content-preserving prompt (保留完整内容, 优化语言表达) -> re-concatenate.

Docs are pre-split in the main thread (one batched SaT pass); worker threads
only make API calls. Doc-level resume; no partial files (chunk failure skips
the doc, retried on next run)."""
import json
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
from wiki_llm_rewrite import PARA_SYSTEM  # noqa: E402

RAW = SCRIPT_DIR / "sp_raw"
DST = SCRIPT_DIR / "sp_llm"
DST.mkdir(exist_ok=True)
CHUNK, WORKERS, MIN_OUT_RATIO = 10, 8, 0.5


def split_all(texts):
    try:
        from wtpsplit import SaT
        sat = SaT("sat-3l-sm")
        try:
            sat.half().to("cuda")
            print("splitter: SaT sat-3l-sm (cuda)", flush=True)
        except Exception:
            print("splitter: SaT sat-3l-sm (cpu)", flush=True)
        return list(sat.split(texts))
    except Exception as e:
        import re
        print(f"splitter: regex fallback ({type(e).__name__})", flush=True)
        rx = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
        return [rx.split(t) for t in texts]


def para_chunk(text: str) -> str:
    best = ""
    for _ in range(3):
        out = chat(PARA_SYSTEM, text) or ""
        if len(out) > len(best):
            best = out
        if len(best) >= MIN_OUT_RATIO * len(text):
            break
    return best


def one(appid: str, sents) -> str:
    out = DST / f"{appid}.txt"
    if out.exists():
        return "skip"
    if len(sents) < 3:
        return "thin"
    parts = []
    for k in range(0, len(sents), CHUNK):
        chunk = " ".join(sents[k:k + CHUNK])
        r = para_chunk(chunk)
        if len(r) < MIN_OUT_RATIO * len(chunk):
            return "chunk-fail"
        parts.append(r)
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n\n".join(parts), encoding="utf-8")
    tmp.replace(out)
    return "ok"


def main() -> None:
    files = sorted(RAW.glob("*.txt"))
    todo = [f for f in files if not (DST / f.name).exists()]
    print(f"{len(files)} store pages; {len(todo)} to paraphrase", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return
    texts = [f.read_text(encoding="utf-8", errors="ignore") for f in todo]
    sent_lists = split_all(texts)
    sent_lists = [[s.strip() for s in sl if len(s.strip()) >= 2] for sl in sent_lists]
    print(f"pre-split done: {sum(map(len, sent_lists))} sentences", flush=True)

    t0 = time.time()
    ok = err = done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(one, f.stem, sl): f for f, sl in zip(todo, sent_lists)}
        for fut in as_completed(futs):
            f = futs[fut]
            done += 1
            try:
                r = fut.result()
                ok += (r == "ok")
                if r in ("chunk-fail", "thin"):
                    err += 1
            except Exception as e:
                err += 1
                print(f"  [err] {f.name}: {type(e).__name__}: {e}", flush=True)
            if done % 50 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} (ok {ok}, err {err}) [{time.time()-t0:.0f}s]",
                      flush=True)
    n_total = len(list(DST.glob("*.txt")))
    json.dump({"generated_ok": ok, "err": err, "total_files": n_total},
              open(SCRIPT_DIR / "sp_llm_manifest.json", "w"), indent=1)
    print(f"sp_llm done: generated {ok}, err {err}, total on disk {n_total}", flush=True)


if __name__ == "__main__":
    main()
