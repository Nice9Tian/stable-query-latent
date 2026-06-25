"""Combine multiple game-review pipeline outputs by appid.

The normal pipeline produces a workdir with:

    metadata/*.json
    sentences/*.json
    embedded/*.json

This script merges one or more such workdirs into a single workdir while
deduplicating by Steam appid. Inputs are ordered by priority: if two sources
contain the same appid, the first source wins. That default keeps the newer
2020-2024 source ahead of the older Kaggle 2017 source.

It can also merge companion ``games.json`` metadata and raw ``reviews/*.csv`` so
downstream H5 building, TAG labels, and recommendation-rate probes stay aligned.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

STAGES = ("metadata", "sentences", "embedded")


def appid_from_path(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        shutil.copy2(src, tmp)
        tmp.replace(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def stage_input_dirs(args, stage: str) -> list[Path]:
    explicit = getattr(args, f"{stage}_dirs")
    if explicit:
        return [Path(path) for path in explicit]
    return [Path(workdir) / stage for workdir in args.source_workdirs]


def combine_stage(stage: str, input_dirs: list[Path], output_dir: Path, overwrite: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, dict] = {}
    duplicates = []
    missing_dirs = []

    for source_index, input_dir in enumerate(input_dirs):
        input_dir = Path(input_dir)
        if not input_dir.exists():
            missing_dirs.append(str(input_dir))
            continue
        for path in sorted(input_dir.glob("*.json")):
            appid = appid_from_path(path)
            if appid in selected:
                duplicates.append({
                    "appid": appid,
                    "kept": selected[appid]["source"],
                    "skipped": str(path),
                })
                continue
            selected[appid] = {
                "source": str(path),
                "source_index": source_index,
                "filename": path.name,
            }

    copied = skipped_existing = 0
    for appid, info in selected.items():
        src = Path(info["source"])
        dst = output_dir / src.name
        if dst.exists() and not overwrite:
            skipped_existing += 1
            continue
        atomic_copy(src, dst)
        copied += 1

    return {
        "stage": stage,
        "output_dir": str(output_dir),
        "input_dirs": [str(path) for path in input_dirs],
        "selected_files": len(selected),
        "copied_files": copied,
        "skipped_existing": skipped_existing,
        "duplicate_appids": duplicates,
        "missing_dirs": missing_dirs,
    }


def appids_in_dirs(input_dirs: list[Path], glob_pattern: str = "*.json") -> set[str]:
    appids = set()
    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            continue
        for path in input_dir.glob(glob_pattern):
            appids.add(appid_from_path(path))
    return appids


def combine_games_json(games_jsons: list[Path], output_path: Path, overwrite: bool,
                       allowed_appids: set[str] | None = None) -> dict:
    selected = {}
    duplicates = []
    missing = []
    for source_index, path in enumerate(games_jsons):
        path = Path(path)
        if not path.exists():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} is not a JSON object keyed by appid.")
        for appid, record in payload.items():
            appid = str(appid)
            if allowed_appids is not None and appid not in allowed_appids:
                continue
            if appid in selected:
                duplicates.append({
                    "appid": appid,
                    "kept": selected[appid]["source"],
                    "skipped": str(path),
                })
                continue
            selected[appid] = {
                "source": str(path),
                "source_index": source_index,
                "record": record,
            }

    if output_path.exists() and not overwrite:
        copied = False
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_name(output_path.name + ".tmp")
        try:
            out = {appid: info["record"] for appid, info in selected.items()}
            tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            tmp.replace(output_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        copied = True

    return {
        "output_path": str(output_path),
        "input_jsons": [str(path) for path in games_jsons],
        "allowed_appids": len(allowed_appids) if allowed_appids is not None else None,
        "selected_games": len(selected),
        "duplicate_appids": duplicates,
        "missing_inputs": missing,
        "written": copied,
    }


def combine_reviews(review_dirs: list[Path], output_dir: Path, overwrite: bool,
                    allowed_appids: set[str] | None = None) -> dict:
    if not review_dirs:
        return {}
    return combine_file_dirs(
        input_dirs=review_dirs,
        output_dir=output_dir,
        glob_pattern="*.csv",
        label="reviews",
        overwrite=overwrite,
        allowed_appids=allowed_appids,
    )


def combine_file_dirs(input_dirs: list[Path], output_dir: Path, glob_pattern: str,
                      label: str, overwrite: bool, allowed_appids: set[str] | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, dict] = {}
    duplicates = []
    missing_dirs = []
    for source_index, input_dir in enumerate(input_dirs):
        input_dir = Path(input_dir)
        if not input_dir.exists():
            missing_dirs.append(str(input_dir))
            continue
        for path in sorted(input_dir.glob(glob_pattern)):
            appid = appid_from_path(path)
            if allowed_appids is not None and appid not in allowed_appids:
                continue
            if appid in selected:
                duplicates.append({
                    "appid": appid,
                    "kept": selected[appid]["source"],
                    "skipped": str(path),
                })
                continue
            selected[appid] = {
                "source": str(path),
                "source_index": source_index,
                "filename": path.name,
            }

    copied = skipped_existing = 0
    for info in selected.values():
        src = Path(info["source"])
        dst = output_dir / src.name
        if dst.exists() and not overwrite:
            skipped_existing += 1
            continue
        atomic_copy(src, dst)
        copied += 1
    return {
        "label": label,
        "output_dir": str(output_dir),
        "input_dirs": [str(path) for path in input_dirs],
        "allowed_appids": len(allowed_appids) if allowed_appids is not None else None,
        "selected_files": len(selected),
        "copied_files": copied,
        "skipped_existing": skipped_existing,
        "duplicate_appids": duplicates,
        "missing_dirs": missing_dirs,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workdirs", type=Path, nargs="+", default=[],
                        help="Pipeline workdirs to merge, in priority order.")
    parser.add_argument("--output-workdir", type=Path, required=True)
    parser.add_argument("--only", nargs="+", choices=STAGES, default=None)
    parser.add_argument("--skip", nargs="+", choices=STAGES, default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--sentences-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--embedded-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--games-jsons", type=Path, nargs="+", default=None)
    parser.add_argument("--review-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--restrict-sidecars-to-metadata", action=argparse.BooleanOptionalAction, default=True,
                        help="Restrict games.json/reviews sidecars to appids present in metadata inputs.")
    parser.add_argument("--skip-games-json", action="store_true")
    parser.add_argument("--skip-reviews", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    stages = set(args.only) if args.only else set(STAGES)
    stages -= set(args.skip)

    if not args.source_workdirs and not any(
        getattr(args, f"{stage}_dirs") for stage in STAGES
    ) and not args.games_jsons and not args.review_dirs:
        raise SystemExit(
            "Provide --source-workdirs or explicit --metadata-dirs/--sentences-dirs/"
            "--embedded-dirs/--games-jsons/--review-dirs."
        )

    results = []
    metadata_input_dirs = stage_input_dirs(args, "metadata")
    allowed_sidecar_appids = (
        appids_in_dirs(metadata_input_dirs)
        if args.restrict_sidecars_to_metadata
        else None
    )
    games_result = None
    if args.games_jsons and not args.skip_games_json:
        games_result = combine_games_json(
            [Path(path) for path in args.games_jsons],
            args.output_workdir / "games.json",
            args.overwrite,
            allowed_appids=allowed_sidecar_appids,
        )
        print(
            f"combine games.json: selected={games_result['selected_games']} "
            f"duplicates={len(games_result['duplicate_appids'])}",
            flush=True,
        )

    reviews_result = None
    if args.review_dirs and not args.skip_reviews:
        reviews_result = combine_reviews(
            [Path(path) for path in args.review_dirs],
            args.output_workdir / "reviews",
            args.overwrite,
            allowed_appids=allowed_sidecar_appids,
        )
        print(
            f"combine reviews: selected={reviews_result['selected_files']} "
            f"copied={reviews_result['copied_files']} duplicates={len(reviews_result['duplicate_appids'])}",
            flush=True,
        )

    for stage in STAGES:
        if stage not in stages:
            continue
        result = combine_stage(
            stage,
            stage_input_dirs(args, stage),
            args.output_workdir / stage,
            args.overwrite,
        )
        results.append(result)
        print(
            f"combine {stage}: selected={result['selected_files']} "
            f"copied={result['copied_files']} duplicates={len(result['duplicate_appids'])}",
            flush=True,
        )

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_workdirs": [str(path) for path in args.source_workdirs],
        "output_workdir": str(args.output_workdir),
        "games_json": games_result,
        "reviews": reviews_result,
        "stages": results,
    }
    manifest_path = args.output_workdir / "combine_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(manifest_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    print(f"combined workdir -> {args.output_workdir}", flush=True)


if __name__ == "__main__":
    main()
