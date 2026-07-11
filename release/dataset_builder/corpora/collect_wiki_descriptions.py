"""Collect full Wikipedia description texts for the current review corpus.

Reads the corpus game list (appids from text_h5.h5, human names from
games.json), reuses the Wikipedia lookup machinery from
PXIbench_test/build/collect_game_descriptions.py (direct-title candidates,
game-page scoring, retry-safe MediaWiki client), but keeps the FULL plain-text
extract per game instead of the PXI few-sentence summary, so games can be
ranked by how much independent description text exists.

Outputs one txt per matched game plus metadata.csv, and a top-N selection
(top100.csv by default) ranked by extract character count. Resumes by default:
existing txt files are skipped but still parsed so ranking stays complete.

    python VICReg_review/collect_wiki_descriptions.py            # full run
    python VICReg_review/collect_wiki_descriptions.py --limit 20 # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR = SCRIPT_DIR
import os as _os
# data/code separation: corpora TEXT lives under the data root, not the repo
SCRIPT_DIR = (Path(_os.environ["LARICE_CORPORA"]) if _os.environ.get("LARICE_CORPORA")
              else Path(__file__).resolve().parents[2] / "data" / "corpora")
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "PXIbench_test" / "build"))

import collect_game_descriptions as wikilib  # noqa: E402

DEFAULT_H5 = ROOT / "game_review_data" / "text_h5.h5"
DEFAULT_GAMES_JSON = ROOT / "game_review_data" / "games.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "wiki_descriptions"
HEADER_SEPARATOR = "-" * 8


@dataclass
class CorpusGame:
    appid: str
    name: str


def clean_name(name: str) -> str:
    for mark in ("®", "™", "©"):
        name = name.replace(mark, "")
    return " ".join(name.split())


def read_corpus_games(h5_path: Path, games_json_path: Path) -> list[CorpusGame]:
    with h5py.File(h5_path, "r") as h5:
        appids = [a.decode() if isinstance(a, bytes) else str(a) for a in h5["appids"][:]]
    meta = json.loads(games_json_path.read_text(encoding="utf-8"))
    games = []
    for appid in appids:
        rec = meta.get(appid)
        name = clean_name((rec or {}).get("name", "").strip())
        if name and not name.isdigit():
            games.append(CorpusGame(appid=appid, name=name))
    return games


def out_filename(game: CorpusGame) -> str:
    safe = wikilib.safe_filename(game.name)
    return f"{game.appid}_{safe}"


def build_text(game: CorpusGame, page: wikilib.WikiPage, retrieved_on: str) -> str:
    header = [
        f"Game: {game.name}",
        f"appid: {game.appid}",
        f"Source: {page.title}",
        f"URL: {page.url}",
        f"Match score: {page.score:.2f}",
        f"Retrieved: {retrieved_on}",
        HEADER_SEPARATOR,
    ]
    return "\n".join(header) + "\n" + page.extract.strip() + "\n"


def read_existing_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    _, sep, body = text.partition("\n" + HEADER_SEPARATOR + "\n")
    return body if sep else text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--games-json", type=Path, default=DEFAULT_GAMES_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--limit", type=int, help="Only process the first N games.")
    parser.add_argument("--select-top", type=int, default=100,
                        help="Write topN.csv with the N games holding the most extract text.")
    parser.add_argument("--search-fallback", action="store_true",
                        help="Use Wikipedia full-text search when direct title lookup fails.")
    args = parser.parse_args()

    games = read_corpus_games(args.h5, args.games_json)
    if args.limit:
        games = games[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"corpus games with names: {len(games)}", flush=True)

    session = requests.Session()
    session.headers.update({"User-Agent": wikilib.USER_AGENT})
    retrieved_on = time.strftime("%Y-%m-%d")

    pending = [g for g in games
               if args.overwrite or not (args.output_dir / out_filename(g)).exists()]
    print(f"pending lookups: {len(pending)} (existing files reused: {len(games) - len(pending)})", flush=True)
    direct_cache = (
        wikilib.fetch_direct_page_cache(
            session, [wikilib.Game(game_id=g.appid, name=g.name, genres="") for g in pending]
        )
        if pending else {}
    )

    rows: list[dict[str, object]] = []
    for index, game in enumerate(games, start=1):
        path = args.output_dir / out_filename(game)
        if path.exists() and not args.overwrite:
            body = read_existing_body(path)
            rows.append({"appid": game.appid, "game_name": game.name, "filename": path.name,
                         "status": "existing", "extract_chars": len(body.strip()),
                         "extract_words": len(body.split())})
            continue

        try:
            page = wikilib.find_best_page_from_cache_with_extract(session, game.name, direct_cache)
            if page is None and args.search_fallback:
                page = wikilib.find_best_page(session, game.name, search_fallback=True)
        except Exception as exc:  # one bad title must not kill a multi-hour run
            rows.append({"appid": game.appid, "game_name": game.name, "filename": "",
                         "status": f"error: {type(exc).__name__}", "extract_chars": 0,
                         "extract_words": 0})
            print(f"[{index}/{len(games)}] error: {game.name} ({exc})", flush=True)
            continue
        if page is None:
            rows.append({"appid": game.appid, "game_name": game.name, "filename": "",
                         "status": "no_match", "extract_chars": 0, "extract_words": 0})
            print(f"[{index}/{len(games)}] no_match: {game.name}", flush=True)
        else:
            wikilib.atomic_write_text(path, build_text(game, page, retrieved_on))
            rows.append({"appid": game.appid, "game_name": game.name, "filename": path.name,
                         "status": "ok", "source_title": page.title, "source_url": page.url,
                         "match_score": f"{page.score:.2f}",
                         "extract_chars": len(page.extract.strip()),
                         "extract_words": len(page.extract.split())})
            print(f"[{index}/{len(games)}] ok: {game.name} -> {page.title} "
                  f"({len(page.extract)} chars)", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    fieldnames = ["appid", "game_name", "filename", "status", "source_title",
                  "source_url", "match_score", "extract_chars", "extract_words"]
    meta_tmp = args.output_dir / "metadata.csv.tmp"
    with meta_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    meta_tmp.replace(args.output_dir / "metadata.csv")

    ranked = sorted((r for r in rows if r["extract_chars"]),
                    key=lambda r: -int(r["extract_chars"]))
    top = ranked[: args.select_top]
    top_tmp = args.output_dir / f"top{args.select_top}.csv.tmp"
    with top_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top)
    top_tmp.replace(args.output_dir / f"top{args.select_top}.csv")

    matched = sum(1 for r in rows if r["extract_chars"])
    print(f"done: {matched}/{len(games)} games matched; "
          f"top{args.select_top}.csv floor = {top[-1]['extract_chars'] if top else 0} chars",
          flush=True)


if __name__ == "__main__":
    main()
