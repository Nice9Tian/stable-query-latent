"""One-command builder for the game-review dataset.

Pipeline layout:

1. ``build_1.py`` optionally builds the local 2020-2024 Steam source, if present.
2. ``build_2.py`` builds the Kaggle ``andrewmvd/steam-reviews`` source.
3. ``combine.py`` merges available branches by appid, keeping source 1 first.

The build treats ``games.json`` as an output, not an input. The Kaggle branch
creates its prepared ``games.json`` before metadata building, and the final
combine stage fills any missing records from produced metadata files. Every
stage is resumable by default; pass ``--overwrite`` only when you intentionally
want to rebuild.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_SOURCE1_DIR = SCRIPT_DIR / "Steam Games Metadata and Player Reviews (2020–2024"
DEFAULT_SOURCE1_REVIEWS = DEFAULT_SOURCE1_DIR / "Game Reviews"

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


def has_csvs(path: Path) -> bool:
    path = Path(path)
    return path.exists() and any(path.glob("*.csv"))


def has_stage_outputs(workdir: Path) -> bool:
    workdir = Path(workdir)
    return any((workdir / stage).exists() and any((workdir / stage).glob("*.json")) for stage in STAGES)


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


def combine_command(args, *, include_build1: bool, include_build2: bool) -> list[str]:
    source_workdirs = []
    games_jsons = []
    review_dirs = []

    if include_build1:
        source_workdirs.append(args.build1_workdir)
        review_dirs.append(args.build1_reviews_dir)

    if include_build2:
        source_workdirs.append(args.build2_workdir)
        review_dirs.append(args.build2_prepared_dir / "reviews")
        games_jsons.append(args.build2_prepared_dir / "games.json")

    if not source_workdirs:
        raise SystemExit("No built source workdirs are available to combine.")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "combine.py"),
        "--source-workdirs",
        *[str(path) for path in source_workdirs],
        "--output-workdir",
        str(args.combined_workdir),
    ]
    if games_jsons and not args.skip_combine_games_json:
        cmd.extend(["--games-jsons", *[str(path) for path in games_jsons]])
    if review_dirs and not args.skip_combine_reviews:
        cmd.extend(["--review-dirs", *[str(path) for path in review_dirs]])
    only = mapped_combine_stages(args.only)
    skip = mapped_combine_stages(args.skip)
    if only:
        cmd.extend(["--only", *only])
    if skip:
        cmd.extend(["--skip", *skip])
    if args.overwrite:
        cmd.append("--overwrite")
    if not args.restrict_sidecars_to_metadata:
        cmd.append("--no-restrict-sidecars-to-metadata")
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help=(
            "Final database workdir. If set, intermediate source dirs are placed "
            "under <workdir>/_build unless explicitly overridden."
        ),
    )
    parser.add_argument("--build1-workdir", type=Path, default=DEFAULT_BUILD1_WORKDIR)
    parser.add_argument("--build2-workdir", type=Path, default=DEFAULT_BUILD2_WORKDIR)
    parser.add_argument("--combined-workdir", type=Path, default=DEFAULT_COMBINED_WORKDIR)
    parser.add_argument("--build1-reviews-dir", type=Path, default=DEFAULT_SOURCE1_REVIEWS)
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


def apply_workdir_defaults(args) -> None:
    if args.workdir is None:
        return

    root = Path(args.workdir)
    build_root = root / "_build"
    if args.build1_workdir == DEFAULT_BUILD1_WORKDIR:
        args.build1_workdir = build_root / "source1"
    if args.build2_workdir == DEFAULT_BUILD2_WORKDIR:
        args.build2_workdir = build_root / "source2"
    if args.build2_prepared_dir == DEFAULT_BUILD2_PREPARED:
        args.build2_prepared_dir = build_root / "kaggle_prepared"
    if args.kaggle_cache == DEFAULT_KAGGLE_CACHE:
        args.kaggle_cache = build_root / "kagglehub_cache"
    if args.combined_workdir == DEFAULT_COMBINED_WORKDIR:
        args.combined_workdir = root


def main():
    args = parse_args()
    apply_workdir_defaults(args)

    build1_has_reviews = has_csvs(args.build1_reviews_dir)
    run_build1 = not args.skip_build1 and build1_has_reviews
    run_build2 = not args.skip_build2

    print(
        "=== combined game-review build ===\n"
        f"build1={args.build1_workdir}\n"
        f"build2={args.build2_workdir}\n"
        f"combined={args.combined_workdir}",
        flush=True,
    )
    if args.skip_build1:
        print("\n--- source 1/3: skipped by --skip-build1 ---", flush=True)
    elif not build1_has_reviews:
        print(
            f"\n--- source 1/3: skipped; no review CSVs found in {args.build1_reviews_dir} ---",
            flush=True,
        )
    else:
        print("\n--- source 1/3: build_1.py (without external games.json) ---", flush=True)
        run_command(build_1_command(args))

    if run_build2:
        print("\n--- source 2/3: build_2.py ---", flush=True)
        run_command(build_2_command(args))
    else:
        print("\n--- source 2/3: skipped by --skip-build2 ---", flush=True)

    include_build1 = run_build1 or has_stage_outputs(args.build1_workdir)
    include_build2 = run_build2 or has_stage_outputs(args.build2_workdir)

    if not args.skip_combine:
        print("\n--- source 3/3: combine.py ---", flush=True)
        run_command(combine_command(args, include_build1=include_build1, include_build2=include_build2))
    else:
        print("\n--- source 3/3: skipped by --skip-combine ---", flush=True)
    print(f"\n=== build done. combined workdir={args.combined_workdir} ===", flush=True)


if __name__ == "__main__":
    main()
