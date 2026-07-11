"""Stage 1 of the game-review embedding pipeline: build the per-game review arrays
that are ready for sentence splitting / embedding.

This consolidates the interactive clean_reviews.py chain (length filter -> count
filter -> merge reviews + prepend game metadata, i.e. the `cleaned_3` form) into a
single non-interactive pass:

    for each <game_id>_<collect_id>.csv in the reviews dir(s):
        keep reviews whose text length >= --min-length
        drop the whole game if fewer than --min-count reviews remain
        write <game_id>_<collect_id>.json =
            [detailed_description, about_the_game, short_description, review_1, review_2, ...]

Multiple review directories can be supplied; they are processed in priority order
and deduplicated by appid (first directory wins). The three metadata strings
(looked up in games.json by game_id) are prepended only when --with-meta is set
(default). Output is one JSON array per game.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# Script-relative defaults keep the pipeline portable. Override them if your raw
# review CSVs or games.json live elsewhere.
DEFAULT_REVIEWS_DIR = SCRIPT_DIR / "reviews"
DEFAULT_GAMES_JSON = SCRIPT_DIR / "games.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "game_review_metadata"
META_FIELDS = ("detailed_description", "about_the_game", "short_description")
DEFAULT_WORKERS = 0


def replace_with_retry(tmp_path: Path, output_path: Path, attempts: int = 8) -> None:
    for attempt in range(attempts):
        try:
            tmp_path.replace(output_path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25 * (attempt + 1))


def load_games_meta(games_json):
    with open(games_json, "r", encoding="utf-8-sig") as file:
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


def _process_metadata_task(task):
    (
        csv_path,
        output_path,
        min_length,
        min_count,
        meta_values,
        warn_missing_meta,
    ) = task
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    try:
        reviews = read_reviews(csv_path)
    except Exception as exc:
        return {
            "status": "skipped_error",
            "csv_name": csv_path.name,
            "message": str(exc),
        }

    reviews = [review for review in reviews if len(review) >= min_length]
    if len(reviews) < min_count:
        return {
            "status": "skipped_count",
            "csv_name": csv_path.name,
        }

    data = list(meta_values) + reviews if meta_values is not None else reviews
    tmp_path = output_path.with_suffix(".json.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)
        replace_with_retry(tmp_path, output_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    return {
        "status": "kept",
        "csv_name": csv_path.name,
        "output_name": output_path.name,
        "review_count": len(reviews),
        "warn_missing_meta": warn_missing_meta,
    }


def resolve_workers(workers: int | None) -> int:
    if workers is None or workers <= 0:
        return min(os.cpu_count() or 1, 16)
    return max(1, int(workers))


def build_metadata(
    reviews_dirs=None,
    games_json=DEFAULT_GAMES_JSON,
    output_dir=DEFAULT_OUTPUT_DIR,
    min_length=300,
    min_count=500,
    with_meta=True,
    overwrite=False,
    workers=DEFAULT_WORKERS,
    reviews_dir=None,  # deprecated alias for a single directory
):
    # Normalise to a list of Path objects.
    if reviews_dirs is None:
        reviews_dirs = reviews_dir if reviews_dir is not None else DEFAULT_REVIEWS_DIR
    if isinstance(reviews_dirs, (str, Path)):
        reviews_dirs = [Path(reviews_dirs)]
    else:
        reviews_dirs = [Path(d) for d in reviews_dirs]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect CSVs from all source dirs and dedup by appid (first dir wins).
    appid_to_csv: dict[str, Path] = {}
    source_counts: list[int] = []
    for reviews_dir in reviews_dirs:
        reviews_dir = Path(reviews_dir)
        before = len(appid_to_csv)
        for csv_path in sorted(reviews_dir.glob("*.csv")):
            appid = csv_path.stem.split("_")[0]
            if appid not in appid_to_csv:
                appid_to_csv[appid] = csv_path
        source_counts.append(len(appid_to_csv) - before)

    csv_files = sorted(appid_to_csv.values(), key=lambda p: p.stem)
    if not csv_files:
        raise ValueError(f"No CSV files found in {reviews_dirs}")

    src_summary = ", ".join(f"src{i+1}={n}" for i, n in enumerate(source_counts))
    print(
        f"build_metadata: {len(csv_files)} unique games ({src_summary}) | "
        f"min_length={min_length} min_count={min_count} with_meta={with_meta} -> {output_dir}"
    )

    games = load_games_meta(games_json) if with_meta else {}
    existing_stems = set()
    if not overwrite:
        existing_stems = {path.stem for path in output_dir.glob("*.json")}

    kept = skipped = existing = 0
    tasks = []
    for index, csv_path in enumerate(csv_files, start=1):
        stem = csv_path.stem
        output_path = output_dir / f"{stem}.json"
        if stem in existing_stems:
            existing += 1
            kept += 1
            if existing == 1 or existing % 1000 == 0:
                print(
                    f"  resume: {existing} existing metadata files skipped "
                    f"({index}/{len(csv_files)} scanned)",
                    flush=True,
                )
            continue

        meta_values = None
        warn_missing_meta = False
        if with_meta:
            game_id = stem.split("_")[0]
            meta = games.get(game_id, {})
            warn_missing_meta = not any(meta.get(field) for field in META_FIELDS)
            meta_values = tuple(meta.get(field, "") or "" for field in META_FIELDS)

        tasks.append(
            (
                str(csv_path),
                str(output_path),
                min_length,
                min_count,
                meta_values,
                warn_missing_meta,
            )
        )

    worker_count = resolve_workers(workers)
    if tasks:
        print(
            f"build_metadata: processing {len(tasks)} missing/overwrite files "
            f"with {worker_count} worker(s)",
            flush=True,
        )

    if tasks and worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_process_metadata_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                status = result["status"]
                if status == "kept":
                    kept += 1
                    if result.get("warn_missing_meta"):
                        print(f"  [warn] {result['csv_name']}: no metadata", flush=True)
                    print(
                        f"  {result['csv_name']}: {result['review_count']} reviews "
                        f"-> {result['output_name']}",
                        flush=True,
                    )
                elif status == "skipped_error":
                    skipped += 1
                    print(
                        f"  [skip] {result['csv_name']}: {result['message']}",
                        flush=True,
                    )
                else:
                    skipped += 1
    else:
        for task in tasks:
            result = _process_metadata_task(task)
            status = result["status"]
            if status == "kept":
                kept += 1
                if result.get("warn_missing_meta"):
                    print(f"  [warn] {result['csv_name']}: no metadata", flush=True)
                print(
                    f"  {result['csv_name']}: {result['review_count']} reviews "
                    f"-> {result['output_name']}",
                    flush=True,
                )
            elif status == "skipped_error":
                skipped += 1
                print(f"  [skip] {result['csv_name']}: {result['message']}", flush=True)
            else:
                skipped += 1

    print(
        f"Done. kept {kept}, skipped {skipped} games "
        f"(existing {existing}) -> {output_dir}"
    )
    return kept, skipped


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviews-dirs",
        nargs="+",
        default=None,
        metavar="DIR",
        help="Raw review CSV directories, in priority order (first dir wins per appid).",
    )
    parser.add_argument(
        "--reviews-dir",
        default=None,
        help="Single raw review CSV directory. Alias for --reviews-dirs with one value.",
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
        "--workers",
        default=DEFAULT_WORKERS,
        type=int,
        help="Process-pool workers for CSV filtering/writing (0 -> up to 16 CPU cores).",
    )
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
    reviews_dirs = args.reviews_dirs or ([args.reviews_dir] if args.reviews_dir else None)
    build_metadata(
        reviews_dirs=reviews_dirs,
        games_json=args.games_json,
        output_dir=args.output_dir,
        min_length=args.min_length,
        min_count=args.min_count,
        with_meta=args.with_meta,
        overwrite=args.overwrite,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
