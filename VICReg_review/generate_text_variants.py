"""Generate pseudo real-text variants for the tag-regression eval via an LLM API.

The battery's real_text_tag eval discovers `text_variant_dir/<appid>/<variant>.txt`
automatically -- coverage today is 2 hand-collected games. This script expands it
to the whole tag-eval TEST split (303 games): for each game's Steam
`about_the_game` description it asks a chat-completions API for three rewrites
-- critical (negative), praising (positive), and de-identified (noname: every
game/character/item name swapped for invented words) -- all preserving the
factual content / gameplay / mechanics, and stores the original description as
the neutral variant for free. A game's rewrites run CONCURRENTLY (one task per
missing file), so no variant systematically finishes before another.

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

# ---- API credentials (paste locally; NEVER commit a real token) -------------
BASE_URL = "https://yunwu.ai/v1"
API_TOKEN = ""                      # <-- paste your key here before running
MODEL = "gemma-7b-it"

TEMPERATURE = 0.7
CONCURRENCY = 8
MAX_RETRIES = 5
MIN_DESC_CHARS = 200                # skip games whose description is too thin

SYSTEM_PROMPTS = {
    "negative": (
        "完整保留游戏内容/游戏玩法/游戏机制的真实内容，"
        "但是使用批评态度来完成一篇关于游戏的描述文章。"
        "不要虚构游戏中不存在的内容。用英文撰写，直接输出文章正文。"
    ),
    "positive": (
        "完整保留游戏内容/游戏玩法/游戏机制的真实内容，"
        "但是使用赞扬态度来完成一篇关于游戏的描述文章。"
        "不要虚构游戏中不存在的内容。用英文撰写，直接输出文章正文。"
    ),
    "noname": (
        "保留完整的文章内容，但是不要暴露出游戏的名字/角色的名字/道具的名字"
        "等信息，全部替换成假想的虚构单词，其余内容保持不变。"
        "不要虚构游戏中不存在的内容。用英文撰写，直接输出文章正文。"
    ),
}

SCRIPT_DIR = Path(__file__).resolve().parent
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
            transient = e.code in (429, 500, 502, 503, 504)
            if not transient or attempt == MAX_RETRIES - 1:
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
            if attempt == MAX_RETRIES - 1:
                raise
        time.sleep(delay)
        delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def gen_variant(appid: str, name: str, desc: str, variant: str, out_root: Path) -> str:
    """One API call -> one <appid>/<variant>.txt. Tasks are submitted per
    (game, variant) so a game's positive/negative/noname run CONCURRENTLY --
    no 'all negatives first' ordering, and resume is per-file."""
    user_msg = f"Game: {name}\n\nOfficial description:\n{desc}"
    text = chat(SYSTEM_PROMPTS[variant], user_msg)
    write_atomic(out_root / appid / f"{variant}.txt", text)
    return f"{appid}/{variant}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "after generation, upload + re-embed:\n"
            "  aws s3 cp <out-dir> s3://<bucket>/stable-query-latent/VICReg_review/"
            "text_variants_generated/ --recursive ...\n"
            "  (on a pod) point --text-variant-dir at the uploaded dir and delete the\n"
            "  text_variant_embedding_cache.npz so the eval re-embeds the new files."
        ))
    ap.add_argument("--split-json", default=str(
        Path(r"C:/runpod_data/stable-query-latent/VICReg_review/heads"
             r"/cloud_full_sweep_a100/tag_text_eval_split.json")))
    ap.add_argument("--games-json", default=str(REPO / "game_review_data" / "games.json"))
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR / "text_variants_generated"))
    ap.add_argument("--split", default="test", choices=("test", "val", "train"))
    ap.add_argument("--limit", type=int, default=0, help="cap games (0 = all)")
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
        desc = strip_html(meta.get("about_the_game") or meta.get("detailed_description")
                          or meta.get("short_description") or "")
        if len(desc) < MIN_DESC_CHARS:
            thin.append(appid)
            continue
        todo.append((appid, meta.get("name") or full_name, desc))
    if args.limit:
        todo = todo[: args.limit]

    out_root = Path(args.out_dir)
    # one task per MISSING (game, variant) file -> a game's three rewrites run
    # concurrently and a rerun resumes at file granularity
    tasks = [(a, n, d, v) for a, n, d in todo for v in SYSTEM_PROMPTS
             if not (out_root / a / f"{v}.txt").exists()]
    print(f"{args.split} split: {len(names)} games | with usable description: {len(todo)} "
          f"| missing meta: {len(missing)} | too-thin desc: {len(thin)}")
    print(f"variant files to generate: {len(tasks)} "
          f"(= missing files among {len(todo)} games x {list(SYSTEM_PROMPTS)})")
    if args.dry_run:
        for appid, name, _desc, variant in tasks[:12]:
            print(f"  would generate {appid}/{variant}.txt   {name[:40]!r}")
        return
    if not API_TOKEN:
        raise SystemExit("API_TOKEN is empty -- paste your key at the top of this script.")

    # neutral = the original description, written up front (no API cost)
    for appid, _name, desc in todo:
        neutral = out_root / appid / "neutral.txt"
        if not neutral.exists():
            write_atomic(neutral, desc)

    ok = err = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = {pool.submit(gen_variant, a, n, d, v, out_root): (a, v)
                for a, n, d, v in tasks}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            appid, variant = futs[fut]
            try:
                fut.result()
                ok += 1
                print(f"[{i}/{len(tasks)}] {appid}/{variant} ok", flush=True)
            except Exception as e:                      # keep going; rerun resumes
                err += 1
                print(f"[{i}/{len(tasks)}] {appid}/{variant} FAILED: "
                      f"{type(e).__name__}: {e}", flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min: {ok} file(s) ok, {err} failed "
          f"-> {out_root}")
    if err:
        print("rerun the same command to retry failures (existing files are skipped).")


if __name__ == "__main__":
    main()
