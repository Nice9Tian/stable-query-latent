"""Stage 1 of the game-review embedding pipeline: build the per-game review arrays
that are ready for sentence splitting / embedding.

This consolidates the interactive clean_reviews.py chain (length filter -> count
filter -> merge reviews + prepend game metadata, i.e. the `cleaned_3` form) into a
single non-interactive pass:

    for each <game_id>_<collect_id>.csv in the reviews dir:
        keep reviews whose text length >= --min-length
        drop the whole game if fewer than --min-count reviews remain
        write <game_id>_<collect_id>.json =
            [detailed_description, about_the_game, short_description, review_1, review_2, ...]

The three metadata strings (looked up in games.json by game_id) are prepended only
when --with-meta is set (default). Output is one JSON array per game.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# Script-relative defaults keep the pipeline portable. Override them if your raw
# review CSVs or games.json live elsewhere.
DEFAULT_REVIEWS_DIR = SCRIPT_DIR / "reviews"
DEFAULT_GAMES_JSON = SCRIPT_DIR / "games.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "game_review_metadata"
META_FIELDS = ("detailed_description", "about_the_game", "short_description")


def load_games_meta(games_json):
    with open(games_json, "r", encoding="utf-8") as file:
        meta = json.load(file)
    if not isinstance(meta, dict):
        raise ValueError(f"{games_json} is not a JSON object keyed by game_id.")
    return meta


def read_reviews(csv_path):
    """Return the review-text column as a list of strings (NaN -> '')."""
    df = pd.read_csv(csv_path)
    if "review" not in df.columns:
        raise KeyError("missing 'review' column")
    return df["review"].fillna("").astype(str).tolist()


def build_metadata(
    reviews_dir,
    games_json,
    output_dir,
    min_length=300,
    min_count=500,
    with_meta=True,
    overwrite=False,
):
    reviews_dir = Path(reviews_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    games = load_games_meta(games_json) if with_meta else {}
    csv_files = sorted(reviews_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {reviews_dir}")

    print(
        f"build_metadata: {len(csv_files)} games | min_length={min_length} "
        f"min_count={min_count} with_meta={with_meta} -> {output_dir}"
    )

    kept = skipped = 0
    for csv_path in csv_files:
        stem = csv_path.stem
        output_path = output_dir / f"{stem}.json"
        if output_path.exists() and not overwrite:
            kept += 1
            continue

        try:
            reviews = read_reviews(csv_path)
        except Exception as exc:
            print(f"  [skip] {csv_path.name}: {exc}")
            skipped += 1
            continue

        reviews = [review for review in reviews if len(review) >= min_length]
        if len(reviews) < min_count:
            skipped += 1
            continue

        if with_meta:
            game_id = stem.split("_")[0]
            meta = games.get(game_id, {})
            if not any(meta.get(field) for field in META_FIELDS):
                print(f"  [warn] {csv_path.name}: no metadata for game_id={game_id}")
            data = [meta.get(field, "") or "" for field in META_FIELDS] + reviews
        else:
            data = reviews

        tmp_path = output_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)
        tmp_path.replace(output_path)

        kept += 1
        print(f"  {csv_path.name}: {len(reviews)} reviews -> {output_path.name}")

    print(f"Done. kept {kept}, skipped {skipped} games -> {output_dir}")
    return kept, skipped


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviews-dir",
        default=DEFAULT_REVIEWS_DIR,
        help="Raw review CSV directory (default: script-relative reviews/).",
    )
    parser.add_argument(
        "--games-json",
        default=DEFAULT_GAMES_JSON,
        help="Game metadata JSON file (default: script-relative games.json).",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-length", default=300, type=int)
    parser.add_argument("--min-count", default=500, type=int)
    parser.add_argument(
        "--no-meta",
        dest="with_meta",
        action="store_false",
        help="Do not prepend game metadata (produces the `cleaned_2` form).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    build_metadata(
        args.reviews_dir,
        args.games_json,
        args.output_dir,
        min_length=args.min_length,
        min_count=args.min_count,
        with_meta=args.with_meta,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
