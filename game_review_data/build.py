"""Unified build: merge source1 + source2 → clean → split → embed → H5.

note.txt cleaning spec (applied to both sources jointly):
  cleaned   : keep reviews with length > --min-length (default 300)
  cleaned_1 : keep games with >= --min-count reviews remaining (default 500)
  cleaned_2 : all reviews for a game concatenated into one list
  cleaned_3 : game descriptions (detailed / about / short) prepended as first
              entries so the encoder always sees metadata context

Sources (both must already be present on disk; no download is required):

  source1  Steam Games Metadata and Player Reviews (2020–2024)/
           └─ Game Reviews/*.csv     (23 k games; appid_count.csv format)
           └─ games.json             (rich metadata: positive/negative/tags/…)

  source2  kaggle_steam_reviews_prepared/
           └─ reviews/*.csv          (656 games; same CSV schema)
           └─ games.json             (enriched via Steam API)

When the same appid appears in both sources, source1 wins.

Pipeline stages inside --workdir (default: combined_gamedata/):

  1. games.json merge  → workdir/games.json         (source1 priority)
  2. metadata          → workdir/metadata/           (clean+filter+prepend)
  3. sentences         → workdir/sentences/          (SaT sentence split)
  4. embedded          → workdir/embedded/           (Qwen3 embed, local or cloud)
  5. h5 (optional)     → VICReg_review/h5/           (shard+merge into one HDF5)

Every stage is resumable: existing output files are skipped unless --overwrite.
Pass --only or --skip to run a subset of stages (1-4 only; use --build-h5 for 5).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# --------------------------------------------------------------------------- sources
SOURCE1_REVIEWS = SCRIPT_DIR / "Steam Games Metadata and Player Reviews (2020–2024" / "Game Reviews"
SOURCE1_GAMES_JSON = SCRIPT_DIR / "Steam Games Metadata and Player Reviews (2020–2024" / "games.json"
SOURCE2_REVIEWS = SCRIPT_DIR / "kaggle_steam_reviews_prepared" / "reviews"
SOURCE2_GAMES_JSON = SCRIPT_DIR / "kaggle_steam_reviews_prepared" / "games.json"

DEFAULT_WORKDIR = SCRIPT_DIR / "combined_gamedata"
H5_SCRIPT = PROJECT_ROOT / "VICReg_review" / "build_review_h5.py"

PIPELINE_STAGES = ("metadata", "split", "embed")


# --------------------------------------------------------------------------- helpers
def atomic_json_write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def merge_games_json(sources: list[Path], output_path: Path, overwrite: bool) -> dict:
    """Merge games.json files from multiple sources; first source wins per appid."""
    if output_path.exists() and not overwrite:
        print(f"merge_games_json: skip existing {output_path}", flush=True)
        return json.loads(output_path.read_text(encoding="utf-8"))

    merged: dict = {}
    for path in sources:
        path = Path(path)
        if not path.exists():
            print(f"  [warn] games.json not found: {path}", flush=True)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            print(f"  [warn] {path}: not a JSON object, skipping", flush=True)
            continue
        added = 0
        for appid, record in data.items():
            if str(appid) not in merged:
                merged[str(appid)] = record
                added += 1
        print(f"  games.json {path.name}: {len(data)} records, {added} new", flush=True)

    atomic_json_write(merged, output_path)
    print(f"merge_games_json: {len(merged)} total appids -> {output_path}", flush=True)
    return merged


def run_h5_build(args, workdir: Path) -> None:
    embedded_dir = workdir / "embedded"
    cmd = [
        str(args.python),
        str(H5_SCRIPT),
        "--input-dir", str(embedded_dir),
        "--games-json", str(workdir / "games.json"),
        "--workers", str(args.h5_workers),
        "--shards", str(args.h5_shards),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    print("RUN " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


# --------------------------------------------------------------------------- main
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Source directories
    parser.add_argument("--source1-reviews", type=Path, default=SOURCE1_REVIEWS,
                        help="Source1 review CSV directory (2020-2024 dataset).")
    parser.add_argument("--source1-games-json", type=Path, default=SOURCE1_GAMES_JSON)
    parser.add_argument("--source2-reviews", type=Path, default=SOURCE2_REVIEWS,
                        help="Source2 review CSV directory (Kaggle prepared).")
    parser.add_argument("--source2-games-json", type=Path, default=SOURCE2_GAMES_JSON)

    # Output
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR,
                        help="Root working directory for all pipeline outputs.")

    # Stage control
    parser.add_argument("--only", nargs="+", choices=PIPELINE_STAGES, default=None,
                        help="Run only these pipeline stages.")
    parser.add_argument("--skip", nargs="+", choices=PIPELINE_STAGES, default=[],
                        help="Skip these pipeline stages.")
    parser.add_argument("--skip-merge-games-json", action="store_true",
                        help="Skip the games.json merge step (use existing workdir/games.json).")
    parser.add_argument("--build-h5", action="store_true",
                        help="After embedding, run build_review_h5.py to produce the HDF5.")
    parser.add_argument("--overwrite", action="store_true")

    # Cleaning / filtering (note.txt spec)
    parser.add_argument("--min-length", type=int, default=300,
                        help="Minimum review character length (note.txt: >300).")
    parser.add_argument("--min-count", type=int, default=500,
                        help="Minimum kept reviews per game (note.txt: >500).")

    # Splitting
    parser.add_argument("--split-model", default="sat-3l-sm")
    parser.add_argument("--split-device", default=None)
    parser.add_argument("--chunk-size", type=int, default=2000)

    # Embedding
    parser.add_argument("--backend", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--local-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--normalize", action="store_true")

    # H5 build (only used when --build-h5 is set)
    parser.add_argument("--h5-workers", type=int, default=2)
    parser.add_argument("--h5-shards", type=int, default=8)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))

    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    run = set(args.only) if args.only else set(PIPELINE_STAGES)
    run -= set(args.skip)

    # Resolve review source dirs (skip missing ones with a warning)
    review_dirs: list[Path] = []
    for label, path in [("source1", args.source1_reviews), ("source2", args.source2_reviews)]:
        if path.exists() and any(path.glob("*.csv")):
            review_dirs.append(path)
        else:
            print(f"[warn] {label} review dir not found or empty: {path}", flush=True)

    if not review_dirs:
        raise SystemExit("No review CSV directories found. Check --source1-reviews / --source2-reviews.")

    print(
        f"=== unified game-review build ===\n"
        f"workdir  : {workdir}\n"
        f"sources  : {[str(p) for p in review_dirs]}\n"
        f"stages   : {sorted(run)}\n"
        f"backend  : {args.backend}\n",
        flush=True,
    )

    # ------------------------------------------------------------------ games.json
    if not args.skip_merge_games_json:
        print("\n--- merge games.json ---", flush=True)
        games_json_sources = []
        for path in [args.source1_games_json, args.source2_games_json]:
            if path.exists():
                games_json_sources.append(path)
        merge_games_json(games_json_sources, workdir / "games.json", args.overwrite)

    games_json = workdir / "games.json"
    if not games_json.exists():
        raise SystemExit(
            f"games.json not found at {games_json}. "
            "Run without --skip-merge-games-json or supply it manually."
        )

    # ------------------------------------------------------------------ stage 1: metadata
    if "metadata" in run:
        print("\n--- stage 1/3: metadata (clean + filter + prepend descriptions) ---", flush=True)
        # Import here so sys.path doesn't need to be modified (same package)
        from build_metadata import build_metadata
        build_metadata(
            reviews_dirs=review_dirs,
            games_json=games_json,
            output_dir=workdir / "metadata",
            min_length=args.min_length,
            min_count=args.min_count,
            with_meta=True,
            overwrite=args.overwrite,
        )

    # ------------------------------------------------------------------ stage 2: split
    if "split" in run:
        print("\n--- stage 2/3: split (SaT sentence splitter) ---", flush=True)
        from split_data import split_data
        split_data(
            input_dir=workdir / "metadata",
            output_dir=workdir / "sentences",
            model=args.split_model,
            device=args.split_device,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )

    # ------------------------------------------------------------------ stage 3: embed
    if "embed" in run:
        print("\n--- stage 3/3: embed (Qwen3 vectors) ---", flush=True)
        from embedding_data import embed_data
        embed_data(
            input_dir=workdir / "sentences",
            output_dir=workdir / "embedded",
            backend=args.backend,
            overwrite=args.overwrite,
            local_model=args.local_model,
            device=args.embed_device,
            base_url=args.base_url,
            token_file=args.token_file,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            normalize=args.normalize,
        )

    # ------------------------------------------------------------------ H5 build
    if args.build_h5:
        print("\n--- H5: shard + merge -> VICReg_review/h5/ ---", flush=True)
        run_h5_build(args, workdir)

    elapsed = time.time() - started
    print(f"\n=== build done in {elapsed:.0f}s. workdir={workdir} ===", flush=True)


if __name__ == "__main__":
    main()
