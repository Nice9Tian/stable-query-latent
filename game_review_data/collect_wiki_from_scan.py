# -*- coding: utf-8 -*-
"""Collect wiki full-text for the games the coverage scan already flagged as
having a page (wiki_coverage_23k.csv). Reuses the scan's matched_title so we
fetch extracts directly by title (no re-probe), reject same-name films/
albums via is_likely_game_page + score_page, and write one wiki_descriptions
txt per surviving game in the exact layout build_wiki_clean expects. Fault-
tolerant (per-batch skip on 503) + throttled, like the scan.

Only the DIRECT-hit rows are fetched (status == 'direct'). Restrict to the
gap games with --gap-appids so we collect only the NEW games (the 814
existing wiki games are already built).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "PXIbench_test" / "build"))
from collect_game_descriptions import (          # noqa: E402
    WIKI_API, score_page, is_likely_game_page, title_key)


def batch_extracts(session, titles, max_retries=8):
    """prop=extracts|info for a batch of titles, own retry, None on give-up."""
    params = dict(action="query", prop="extracts|info", explaintext=1,
                  exlimit="max",   # WITHOUT this, extracts default exlimit=1 ->
                  #                  only the first title gets an extract, the
                  #                  rest come back empty and get rejected.
                  redirects=1, inprop="url", titles="|".join(titles),
                  format="json", formatversion="2")
    for attempt in range(max_retries):
        try:
            r = session.get(WIKI_API, params=params, timeout=60)
            if r.status_code in {429, 500, 502, 503, 504}:
                ra = r.headers.get("Retry-After")
                time.sleep(float(ra) if ra else min(2 ** attempt, 60))
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(min(2 ** attempt, 60))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-csv", default="C:/runpod_data/wiki_coverage_23k.csv")
    ap.add_argument("--games-json", default=str(SCRIPT_DIR / "games.json"))
    ap.add_argument("--gap-appids", default="C:/runpod_data/fullcorpus_gap_appids.txt",
                    help="restrict to these appids (the NEW games)")
    ap.add_argument("--output-dir", default="C:/runpod_data/wiki_descriptions_new")
    ap.add_argument("--chunk-size", type=int, default=12)
    ap.add_argument("--throttle", type=float, default=0.25)
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    gap = None
    if args.gap_appids:
        gap = {l.strip() for l in Path(args.gap_appids).read_text().split()
               if l.strip()}
    gj = json.loads(Path(args.games_json).read_text(encoding="utf-8"))
    rows = [r for r in csv.DictReader(open(args.scan_csv, encoding="utf-8"))
            if r["status"] == "direct"
            and (gap is None or r["appid"] in gap)]
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # skip games already written (resume)
    done = {p.stem.split("_")[0] for p in out.glob("*.txt")}
    rows = [r for r in rows if r["appid"] not in done]
    print(f"collecting wiki full-text for {len(rows):,} direct-hit new games "
          f"(resume: {len(done):,} already done)", flush=True)

    session = requests.Session()
    session.headers["User-Agent"] = "game-repr-research/1.0 (wiki collect)"

    # Per-game single-title fetch: prop=extracts only reliably returns ONE
    # full extract per request regardless of exlimit (content-size cap), so
    # batching silently drops all but one. One request per game is slower but
    # correct.
    kept = rejected = failed = 0
    t0 = time.time()
    for idx, r in enumerate(rows):
        data = batch_extracts(session, [r["matched_title"]])
        if data is None:
            failed += 1
            continue
        pages = data.get("query", {}).get("pages", [])
        page = next((p for p in pages if not p.get("missing")
                     and p.get("extract", "").strip()), None)
        name = (gj.get(r["appid"], {}).get("name") or r["name"]).strip()
        if (page is None or not is_likely_game_page(page)
                or score_page(name, page) < 3.0
                or len(page.get("extract", "").strip()) < args.min_chars):
            rejected += 1
        else:
            safe = "".join(c if c.isalnum() or c in " -_" else "_"
                           for c in name)[:80].strip() or r["appid"]
            body = "\n".join([
                f"Game: {name}", f"appid: {r['appid']}",
                f"Source: {page.get('title','')}",
                f"URL: {page.get('fullurl','')}",
                f"Match score: {score_page(name, page):.2f}", "",
                page["extract"].strip()]) + "\n"
            (out / f"{r['appid']}_{safe}.txt").write_text(body, encoding="utf-8")
            kept += 1
        if idx % 200 == 0:
            print(f"  {idx}/{len(rows)}  kept {kept} rejected {rejected} "
                  f"failed {failed}  [{(time.time()-t0)/60:.1f}m]", flush=True)
        time.sleep(args.throttle)

    print(f"\nDONE: kept {kept:,} / rejected {rejected:,} / failed {failed:,} "
          f"-> {out}")
    print(f"kept fraction of {len(rows):,} attempted: {kept/max(len(rows),1):.1%} "
          f"(these are the games that pass the first game-page filter; "
          f"build_wiki_clean's embedding-rank check at 23k will trim further)")


if __name__ == "__main__":
    main()
