"""One-command builder for the combined game-review dataset.

Pipeline layout:

1. ``build_1.py`` builds the original 2020-2024 Steam metadata/review source.
2. ``build_2.py`` builds the Kaggle ``andrewmvd/steam-reviews`` source.
3. ``combine.py`` merges both branches by appid, keeping source 1 first.

The combined output keeps the modern 2020-2024 games used by the identity tests
while adding the larger Kaggle 2017 review pool. Every stage is resumable by
default; pass ``--overwrite`` only when you intentionally want to rebuild.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_SOURCE1_DIR = SCRIPT_DIR / "Steam Games Metadata and Player Reviews (2020–2024"
DEFAULT_SOURCE1_REVIEWS = DEFAULT_SOURCE1_DIR / "Game Reviews"
DEFAULT_SOURCE1_GAMES_JSON = DEFAULT_SOURCE1_DIR / "games.json"

DEFAULT_BUILD1_WORKDIR = SCRIPT_DIR / "build_1_gamedata"
DEFAULT_BUILD2_WORKDIR = SCRIPT_DIR / "build_2_gamedata"
DEFAULT_BUILD2_PREPARED = SCRIPT_DIR / "kaggle_steam_reviews_prepared"
DEFAULT_COMBINED_WORKDIR = SCRIPT_DIR / "combined_gamedata"
DEFAULT_KAGGLE_CACHE = SCRIPT_DIR / "kagglehub_cache"

STAGES = ("metadata", "split", "embed")
COMBINE_STAGE_MAP = {
    "metadata": "metadata",
    "split": "sentences",
    "embed": "embedded",
}


def run_command(cmd: list[str], cwd: Path = SCRIPT_DIR) -> None:
    print("RUN " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def add_stage_args(cmd: list[str], args) -> None:
    if args.only:
        cmd.extend(["--only", *args.only])
    if args.skip:
        cmd.extend(["--skip", *args.skip])


def mapped_combine_stages(stages: list[str] | None) -> list[str] | None:
    if not stages:
        return stages
    return [COMBINE_STAGE_MAP[stage] for stage in stages]


def build_1_command(args) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_1.py"),
        "--workdir",
        str(args.build1_workdir),
        "--reviews-dir",
        str(args.build1_reviews_dir),
        "--games-json",
        str(args.build1_games_json),
        "--min-length",
        str(args.min_length),
        "--min-count",
        str(args.min_count),
        "--split-model",
        args.split_model,
        "--chunk-size",
        str(args.chunk_size),
        "--backend",
        args.backend,
        "--local-model",
        args.local_model,
        "--concurrency",
        str(args.concurrency),
        "--batch-size",
        str(args.batch_size),
    ]
    add_stage_args(cmd, args)
    if args.overwrite:
        cmd.append("--overwrite")
    if args.no_meta:
        cmd.append("--no-meta")
    if args.split_device:
        cmd.extend(["--split-device", args.split_device])
    if args.embed_device:
        cmd.extend(["--embed-device", args.embed_device])
    if args.base_url:
        cmd.extend(["--base-url", args.base_url])
    if args.token_file:
        cmd.extend(["--token-file", args.token_file])
    if args.normalize:
        cmd.append("--normalize")
    return cmd


def build_2_command(args) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_2.py"),
        "--prepared-dir",
        str(args.build2_prepared_dir),
        "--workdir",
        str(args.build2_workdir),
        "--kaggle-cache",
        str(args.kaggle_cache),
        "--min-length",
        str(args.min_length),
        "--min-count",
        str(args.min_count),
        "--prepare-chunksize",
        str(args.prepare_chunksize),
        "--enrich-batch-size",
        str(args.enrich_batch_size),
        "--enrich-sleep",
        str(args.enrich_sleep),
        "--enrich-retry-sleep",
        str(args.enrich_retry_sleep),
        "--enrich-retries",
        str(args.enrich_retries),
        "--split-model",
        args.split_model,
        "--chunk-size",
        str(args.chunk_size),
        "--backend",
        args.backend,
        "--local-model",
        args.local_model,
        "--concurrency",
        str(args.concurrency),
        "--batch-size",
        str(args.batch_size),
    ]
    add_stage_args(cmd, args)
    if args.kaggle_input:
        cmd.extend(["--kaggle-input", str(args.kaggle_input)])
    if args.skip_kaggle_download:
        cmd.append("--skip-download")
    if args.skip_kaggle_prepare:
        cmd.append("--skip-prepare")
    if args.skip_kaggle_enrich:
        cmd.append("--skip-enrich")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.no_meta:
        cmd.append("--no-meta")
    if args.strict_length:
        cmd.append("--strict-length")
    else:
        cmd.append("--no-strict-length")
    if args.strict_count:
        cmd.append("--strict-count")
    else:
        cmd.append("--no-strict-count")
    if args.split_device:
        cmd.extend(["--split-device", args.split_device])
    if args.embed_device:
        cmd.extend(["--embed-device", args.embed_device])
    if args.base_url:
        cmd.extend(["--base-url", args.base_url])
    if args.token_file:
        cmd.extend(["--token-file", args.token_file])
    if args.normalize:
        cmd.append("--normalize")
    return cmd


def combine_command(args) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "combine.py"),
        "--source-workdirs",
        str(args.build1_workdir),
        str(args.build2_workdir),
        "--output-workdir",
        str(args.combined_workdir),
        "--games-jsons",
        str(args.build1_games_json),
        str(args.build2_prepared_dir / "games.json"),
        "--review-dirs",
        str(args.build1_reviews_dir),
        str(args.build2_prepared_dir / "reviews"),
    ]
    only = mapped_combine_stages(args.only)
    skip = mapped_combine_stages(args.skip)
    if only:
        cmd.extend(["--only", *only])
    if skip:
        cmd.extend(["--skip", *skip])
    if args.overwrite:
        cmd.append("--overwrite")
    if args.skip_combine_games_json:
        cmd.append("--skip-games-json")
    if args.skip_combine_reviews:
        cmd.append("--skip-reviews")
    if not args.restrict_sidecars_to_metadata:
        cmd.append("--no-restrict-sidecars-to-metadata")
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build1-workdir", type=Path, default=DEFAULT_BUILD1_WORKDIR)
    parser.add_argument("--build2-workdir", type=Path, default=DEFAULT_BUILD2_WORKDIR)
    parser.add_argument("--combined-workdir", type=Path, default=DEFAULT_COMBINED_WORKDIR)
    parser.add_argument("--build1-reviews-dir", type=Path, default=DEFAULT_SOURCE1_REVIEWS)
    parser.add_argument("--build1-games-json", type=Path, default=DEFAULT_SOURCE1_GAMES_JSON)
    parser.add_argument("--build2-prepared-dir", type=Path, default=DEFAULT_BUILD2_PREPARED)
    parser.add_argument("--kaggle-cache", type=Path, default=DEFAULT_KAGGLE_CACHE)
    parser.add_argument("--kaggle-input", type=Path, default=None)

    parser.add_argument("--only", nargs="+", choices=STAGES, default=None)
    parser.add_argument("--skip", nargs="+", choices=STAGES, default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-build1", action="store_true")
    parser.add_argument("--skip-build2", action="store_true")
    parser.add_argument("--skip-combine", action="store_true")
    parser.add_argument("--skip-kaggle-download", action="store_true")
    parser.add_argument("--skip-kaggle-prepare", action="store_true")
    parser.add_argument("--skip-kaggle-enrich", action="store_true")

    parser.add_argument("--min-length", type=int, default=300)
    parser.add_argument("--min-count", type=int, default=500)
    parser.add_argument("--strict-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-count", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-meta", action="store_true")
    parser.add_argument("--prepare-chunksize", type=int, default=200_000)

    parser.add_argument("--enrich-batch-size", type=int, default=1)
    parser.add_argument("--enrich-sleep", type=float, default=2.0)
    parser.add_argument("--enrich-retry-sleep", type=float, default=10.0)
    parser.add_argument("--enrich-retries", type=int, default=5)

    parser.add_argument("--split-model", default="sat-3l-sm")
    parser.add_argument("--split-device", default=None)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--backend", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--local-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--normalize", action="store_true")

    parser.add_argument("--skip-combine-games-json", action="store_true")
    parser.add_argument("--skip-combine-reviews", action="store_true")
    parser.add_argument("--restrict-sidecars-to-metadata", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    print(
        "=== combined game-review build ===\n"
        f"build1={args.build1_workdir}\n"
        f"build2={args.build2_workdir}\n"
        f"combined={args.combined_workdir}",
        flush=True,
    )
    if not args.skip_build1:
        print("\n--- source 1/3: build_1.py ---", flush=True)
        run_command(build_1_command(args))
    if not args.skip_build2:
        print("\n--- source 2/3: build_2.py ---", flush=True)
        run_command(build_2_command(args))
    if not args.skip_combine:
        print("\n--- source 3/3: combine.py ---", flush=True)
        run_command(combine_command(args))
    print(f"\n=== build done. combined workdir={args.combined_workdir} ===", flush=True)


if __name__ == "__main__":
    main()
