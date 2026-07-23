# -*- coding: utf-8 -*-
"""Scan how many full-corpus games have an English Wikipedia page BEFORE
committing to full collection. Reuses the existing MediaWiki client
(PXIbench_test/build/collect_game_descriptions.py): for each game, batch-
resolve its candidate titles (name, "<name> (video game)", ... + aliases)
through prop=info + redirect/normalize, mark a DIRECT hit if any candidate
resolves to a real page; optionally run the full-text SEARCH fallback for
the misses. Reports coverage (an UPPER BOUND -- a resolved page may still
be a same-name film/album that build_wiki_clean's top-3 embedding check
would later reject). No full-text download, no GPU.

Usage:
  python game_review_data/scan_wiki_coverage.py --games-json <games.json> \
      [--appid-list <catalog_appids.txt>] [--search-fallback] \
      [--limit N] --out <coverage.csv>
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "PXIbench_test" / "build"))
from collect_game_descriptions import (          # noqa: E402
    candidate_titles, search_pages, title_key, similarity, WIKI_API)


def robust_info(session, titles, max_retries=8):
    """One prop=info batch with OWN retry; returns json or None (give up on
    this batch only -- never crash the whole scan on a transient 503)."""
    params = dict(action="query", prop="info", redirects=1, inprop="url",
                  titles="|".join(titles), format="json", formatversion="2")
    for attempt in range(max_retries):
        try:
            r = session.get(WIKI_API, params=params, timeout=40)
            if r.status_code in {429, 500, 502, 503, 504}:
                ra = r.headers.get("Retry-After")
                time.sleep(float(ra) if ra else min(2 ** attempt, 60))
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(min(2 ** attempt, 60))
    return None


def direct_scan(session, games, chunk=20, throttle=0.2):
    """Fault-tolerant batched direct-title resolution. Mirrors
    fetch_direct_page_cache's redirect/normalize mapping but skips (not
    crashes on) any batch that keeps 503-ing."""
    all_titles, seen = [], set()
    for g in games:
        for t in candidate_titles(g.name):
            k = title_key(t)
            if k not in seen:
                seen.add(k)
                all_titles.append(t)
    cache, failed_titles = {}, 0
    nb = (len(all_titles) + chunk - 1) // chunk
    for bi, start in enumerate(range(0, len(all_titles), chunk)):
        ch = all_titles[start:start + chunk]
        data = robust_info(session, ch)
        if data is None:
            failed_titles += len(ch)
            if bi % 100 == 0:
                print(f"  batch {bi}/{nb} FAILED (skipped)", flush=True)
            continue
        q = data.get("query", {})
        pages_by = {title_key(p.get("title", "")): p
                    for p in q.get("pages", [])
                    if not p.get("missing") and not p.get("invalid")
                    and "pageid" in p}
        norm = {title_key(i.get("from", "")): title_key(i.get("to", ""))
                for i in q.get("normalized", [])}
        redir = {title_key(i.get("from", "")): title_key(i.get("to", ""))
                 for i in q.get("redirects", [])}
        for title in ch:
            keys = [title_key(title)]
            if keys[-1] in norm:
                keys.append(norm[keys[-1]])
            if keys[-1] in redir:
                keys.append(redir[keys[-1]])
            page = next((pages_by[k] for k in reversed(keys)
                         if k in pages_by), None)
            if page:
                cache.setdefault(title_key(title), []).append(page)
        if bi % 100 == 0:
            print(f"  direct batch {bi}/{nb} ({start+chunk}/{len(all_titles)} "
                  f"titles, {len(cache)} resolved)", flush=True)
        time.sleep(throttle)
    return cache, failed_titles


def load_games(games_json, appid_filter):
    gj = json.loads(Path(games_json).read_text(encoding="utf-8"))
    out = []
    for appid, rec in gj.items():
        if appid_filter is not None and appid not in appid_filter:
            continue
        name = (rec.get("name") or "").strip()
        if name:
            out.append(SimpleNamespace(appid=appid, name=name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-json",
                    default=str(SCRIPT_DIR / "games.json"))
    ap.add_argument("--appid-list", default=None,
                    help="optional file with one catalog appid per line; "
                         "restricts the scan to the full-corpus review games")
    ap.add_argument("--search-fallback", action="store_true",
                    help="run the full-text search fallback for direct misses "
                         "(higher recall, slower, lower precision)")
    ap.add_argument("--limit", type=int, default=0,
                    help="scan only the first N games (smoke test)")
    ap.add_argument("--out", default=str(SCRIPT_DIR / "wiki_coverage.csv"))
    ap.add_argument("--chunk-size", type=int, default=45)
    args = ap.parse_args()

    appid_filter = None
    if args.appid_list:
        appid_filter = {l.strip() for l in Path(args.appid_list).read_text().split()
                        if l.strip()}
        print(f"catalog filter: {len(appid_filter):,} appids")

    games = load_games(args.games_json, appid_filter)
    if args.limit:
        games = games[:args.limit]
    print(f"scanning {len(games):,} games for en.wikipedia pages ...", flush=True)

    session = requests.Session()
    session.headers["User-Agent"] = "game-repr-research/1.0 (wiki coverage scan)"

    # DIRECT: batch-resolve all candidate titles (fault-tolerant + throttled)
    t0 = time.time()
    cache, failed = direct_scan(session, games, chunk=args.chunk_size)
    print(f"direct scan done in {(time.time()-t0)/60:.1f} min "
          f"({failed} titles in skipped batches)", flush=True)

    rows = []
    direct_hit = 0
    miss = []
    for g in games:
        matched = None
        for title in candidate_titles(g.name):
            pages = cache.get(title_key(title))
            if pages:
                matched = pages[0].get("title", title)
                break
        if matched:
            direct_hit += 1
            rows.append((g.appid, g.name, "direct", matched))
        else:
            miss.append(g)
            rows.append((g.appid, g.name, "none", ""))

    # SEARCH fallback (optional) for the direct misses
    search_hit = 0
    if args.search_fallback and miss:
        print(f"search-fallback over {len(miss):,} misses ...", flush=True)
        row_by_appid = {r[0]: i for i, r in enumerate(rows)}
        for j, g in enumerate(miss):
            try:
                results = search_pages(session, g.name, limit=5)
            except Exception:
                results = []
            best = None
            for it in results:
                t = it.get("title", "")
                if similarity(g.name, t) >= 0.6:
                    best = t
                    break
            if best:
                search_hit += 1
                rows[row_by_appid[g.appid]] = (g.appid, g.name, "search", best)
            if j % 200 == 0:
                print(f"  search {j}/{len(miss)}", flush=True)
            time.sleep(0.1)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["appid", "name", "status", "matched_title"])
        w.writerows(rows)

    n = len(games)
    cov = direct_hit + search_hit
    print("\n=== WIKI COVERAGE (upper bound; before build_wiki_clean validation) ===")
    print(f"games scanned:        {n:,}")
    print(f"direct-title hit:     {direct_hit:,} ({direct_hit/n:.1%})")
    if args.search_fallback:
        print(f"+ search-fallback hit: {search_hit:,} ({search_hit/n:.1%})")
    print(f"TOTAL with a page:    {cov:,} ({cov/n:.1%})")
    print(f"no page found:        {n-cov:,} ({(n-cov)/n:.1%})")
    print(f"per-game detail -> {args.out}")
    print("NOTE: this is an UPPER BOUND. build_wiki_clean's name-in-body + "
          "top-3 embedding-rank check will reject same-name films/albums and "
          "franchise confusions, so the trainable universe is smaller.")


if __name__ == "__main__":
    main()
