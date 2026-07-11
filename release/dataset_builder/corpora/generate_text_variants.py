"""Generate pseudo real-text variants for the tag-regression eval via an LLM API.

The battery's real_text_tag eval discovers `text_variant_dir/<appid>/<variant>.txt`
automatically -- coverage today is 2 hand-collected games. This script expands it
to the whole tag-eval TEST split (303 games): each game's FULL Steam store
page text goes to a chat-completions API, which summarizes it into a
descriptive article in four styles -- neutral (中立), praising (赞扬肯定),
critical (消极批评), and noname (neutral tone but every game/character/item
name swapped for invented words) -- all preserving the factual gameplay /
mechanics content. Loop order is GAME-major: each game's missing variants fire
concurrently and the game is fully completed before the next one starts, so an
interrupted run leaves whole games, never one variant everywhere.

Run LOCALLY (games.json lives in game_review_data/):
    python VICReg_review/generate_text_variants.py --dry-run   # coverage check
    python VICReg_review/generate_text_variants.py             # generate

Then upload to the volume and re-embed on a pod (see --help epilog).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---- API credentials: NEVER hardcoded. Loaded from llmAPI.txt next to the
# repo root (gitignored; lines "url=", "token=", optional "model=") or from
# the LLM_API_URL / LLM_API_TOKEN / LLM_API_MODEL environment variables. ----
import os


def _load_llm_credentials():
    """Two sources, either alone is fine: the in-code API block at the top
    of steam_reviews_framework/run.py (exported as LLM_API_* env vars —
    they WIN) and dataset_builder/llmAPI.txt. A mismatch is only reported
    when BOTH sources are non-empty and differ."""
    env = {"url": os.environ.get("LLM_API_URL", ""),
           "token": os.environ.get("LLM_API_TOKEN", ""),
           "model": os.environ.get("LLM_API_MODEL", "")}
    cred = {"url": "", "token": "", "model": ""}
    src = None
    for cand in (Path(__file__).resolve().parents[1] / "llmAPI.txt",
                 Path(__file__).resolve().parents[2] / "llmAPI.txt",
                 Path(__file__).resolve().parent / "llmAPI.txt"):
        if cand.exists():
            src = cand
            for line in cand.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() in cred and v.strip():
                    cred[k.strip()] = v.strip()
            break
    for k in cred:
        # a mismatch needs BOTH sources present; one side missing is fine
        if env[k] and cred[k] and env[k] != cred[k]:
            print(f"NOTE: LLM-API {k} set in code differs from {src} — "
                  f"using the in-code value.", flush=True)
        if env[k]:
            cred[k] = env[k]
    cred["model"] = cred["model"] or "gpt-5.4-mini"
    if not cred["url"] or not cred["token"]:
        raise SystemExit(
            "LLM API credentials missing: create llmAPI.txt (url=..., "
            "token=..., model=...) in dataset_builder/ (see "
            "llmAPI.template.txt), or set "
            "LLM_API_URL / LLM_API_TOKEN.")
    return cred


_CRED = _load_llm_credentials()
BASE_URL = _CRED["url"]
API_TOKEN = _CRED["token"]
MODEL = _CRED["model"]

TEMPERATURE = 0.7
MAX_RETRIES = 5
MIN_DESC_CHARS = 500                # skip games whose description is too thin
LIMIT = 0                         # cap games per run; 0 = all (CLI --limit overrides)

# The user message carries the game's FULL store-page text; the model
# summarizes it into a descriptive article in the requested tone.
_SUMMARIZE = (
    "阅读用户提供的完整游戏页面文本，从中总结出一篇游戏描述性质的文章。"
    "完整保留游戏内容/游戏玩法/游戏机制的真实信息，不要虚构页面中不存在的内容。"
    "语言风格要求：{tone}。用英文撰写，直接输出文章正文。"
)
SYSTEM_PROMPTS = {
    "neutral": _SUMMARIZE.format(tone="中立"),
    "positive": _SUMMARIZE.format(tone="赞扬肯定"),
    "negative": _SUMMARIZE.format(tone="消极批评"),
    "noname": (
        "阅读用户提供的完整游戏页面文本，从中总结出一篇游戏描述性质的文章，"
        "语言风格中立。完整保留游戏内容/游戏玩法/游戏机制的真实信息，"
        "不要虚构页面中不存在的内容。但是不要暴露出游戏的名字/角色的名字/"
        "道具的名字等信息，全部替换成假想的虚构单词。"
        "用英文撰写，直接输出文章正文。"
    ),
}

SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR = SCRIPT_DIR
import os as _os
# data/code separation: corpora TEXT lives under the data root, not the repo
SCRIPT_DIR = (Path(_os.environ["LARICE_CORPORA"]) if _os.environ.get("LARICE_CORPORA")
              else Path(__file__).resolve().parents[2] / "data" / "corpora")
REPO = SCRIPT_DIR.parent


def strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def chat_once(system: str, user: str) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": TEMPERATURE,
    }).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_TOKEN}"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"].strip()


def chat(system: str, user: str) -> str:
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            return chat_once(system, user)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            transient = e.code in (429, 500, 502, 503, 504)
            if not transient or attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"HTTP {e.code}: {body}") from e
            print(f"    retry {attempt + 1}/{MAX_RETRIES - 1}: HTTP {e.code} {body}",
                  flush=True)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"    retry {attempt + 1}/{MAX_RETRIES - 1}: {type(e).__name__}: {e}",
                  flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


def preflight() -> None:
    """One tiny call before the real run: a misconfigured MODEL / BASE_URL /
    token otherwise burns ~30 s of silent retry backoff PER GAME and looks
    like a hang."""
    try:
        chat_once("You are a helpful assistant.", "Reply with the single word: ok")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise SystemExit(
            f"preflight FAILED: HTTP {e.code} from {BASE_URL} (model={MODEL})\n"
            f"  response: {body}\n"
            "  -> check MODEL (is it available on this gateway?), BASE_URL and API_TOKEN."
        )
    except Exception as e:
        raise SystemExit(f"preflight FAILED: {type(e).__name__}: {e}\n"
                         "  -> check BASE_URL / network / API_TOKEN.")
    print(f"preflight ok: {MODEL} @ {BASE_URL}", flush=True)


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def gen_variant(appid: str, name: str, desc: str, variant: str, out_root: Path) -> str:
    """One API call -> one <appid>/<variant>.txt."""
    user_msg = f"Game: {name}\n\nFull store page text:\n{desc}"
    text = chat(SYSTEM_PROMPTS[variant], user_msg)
    write_atomic(out_root / appid / f"{variant}.txt", text)
    return variant


def one_game(appid: str, name: str, desc: str, out_root: Path) -> tuple[list, list]:
    """Complete ONE game before moving on: its missing variants (all four are
    LLM summaries of the full page text now, incl. neutral) fire CONCURRENTLY
    and we wait for all of them. An interrupted run therefore leaves N
    fully-finished games, not one variant scattered across every game.
    Returns (made, failed_variants)."""
    made, failed = [], []
    variants = [v for v in SYSTEM_PROMPTS if not (out_root / appid / f"{v}.txt").exists()]
    if variants:
        with cf.ThreadPoolExecutor(max_workers=len(variants)) as pool:
            futs = {pool.submit(gen_variant, appid, name, desc, v, out_root): v
                    for v in variants}
            for fut in cf.as_completed(futs):
                v = futs[fut]
                try:
                    made.append(fut.result())
                except Exception as e:
                    failed.append(f"{v} ({type(e).__name__}: {e})")
    return made, failed


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "after generation, upload + re-embed:\n"
            "  aws s3 cp <out-dir> s3://<bucket>/larice/VICReg_review/"
            "text_variants_generated/ --recursive ...\n"
            "  (on a pod) point --text-variant-dir at the uploaded dir and delete the\n"
            "  text_variant_embedding_cache.npz so the eval re-embeds the new files."
        ))
    ap.add_argument("--split-json", default=str(
        Path(r"C:/runpod_data/larice/VICReg_review/heads"
             r"/cloud_full_sweep_a100/tag_text_eval_split.json")))
    ap.add_argument("--games-json", default=str(SCRIPT_DIR.parent / "reviews" / "games.json"))
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR / "text_variants_generated"))
    ap.add_argument("--split", default="test", choices=("test", "val", "train"))
    ap.add_argument("--limit", type=int, default=LIMIT,
                    help=f"cap games (0 = all; default from LIMIT constant = {LIMIT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls: report coverage and what would be generated")
    args = ap.parse_args()

    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    names = split[args.split]
    games = json.loads(Path(args.games_json).read_text(encoding="utf-8"))

    todo, missing, thin = [], [], []
    for full_name in names:
        appid = full_name.split("_", 1)[0]
        meta = games.get(appid)
        if not meta:
            missing.append(appid)
            continue
        # FULL page text (detailed_description is the whole store page); the
        # model does the condensing itself
        desc = strip_html(meta.get("detailed_description") or meta.get("about_the_game")
                          or meta.get("short_description") or "")
        if len(desc) < MIN_DESC_CHARS:
            thin.append(appid)
            continue
        todo.append((appid, meta.get("name") or full_name, desc))
    if args.limit:
        todo = todo[: args.limit]

    out_root = Path(args.out_dir)
    # game-major: only games with at least one missing rewrite are visited,
    # and each is fully completed (3 concurrent calls) before the next starts
    pending = [(a, n, d) for a, n, d in todo
               if any(not (out_root / a / f"{v}.txt").exists() for v in SYSTEM_PROMPTS)]
    print(f"{args.split} split: {len(names)} games | with usable description: {len(todo)} "
          f"| missing meta: {len(missing)} | too-thin desc: {len(thin)}")
    print(f"games to process: {len(pending)} (complete: {len(todo) - len(pending)}; "
          f"each = up to {len(SYSTEM_PROMPTS)} concurrent API calls)")
    if args.dry_run:
        for appid, name, _desc in pending[:12]:
            miss = [v for v in SYSTEM_PROMPTS if not (out_root / appid / f"{v}.txt").exists()]
            print(f"  would generate {appid} {name[:40]!r}: {', '.join(miss)}")
        return
    if not API_TOKEN:
        raise SystemExit("API_TOKEN is empty -- paste your key at the top of this script.")
    preflight()

    ok = err = 0
    t0 = time.time()
    for i, (appid, name, desc) in enumerate(pending, 1):
        made, failed = one_game(appid, name, desc, out_root)
        if failed:
            err += 1
            print(f"[{i}/{len(pending)}] {appid} {name[:36]!r} "
                  f"PARTIAL: ok={made} failed={failed}", flush=True)
        else:
            ok += 1
            print(f"[{i}/{len(pending)}] {appid} {name[:36]!r} ok ({', '.join(made)})",
                  flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min: {ok} game(s) complete, "
          f"{err} with failures -> {out_root}")
    if err:
        print("rerun the same command to retry failures (existing files are skipped).")


if __name__ == "__main__":
    main()
