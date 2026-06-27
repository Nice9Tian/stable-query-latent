"""Unified game-review build: download -> clean -> split -> text H5 -> embedding H5.

Sources:

  source1  Steam Games Metadata and Player Reviews (2020-2024)
           Game Reviews/*.csv + games.json, downloaded from Mendeley.

  source2  Kaggle andrewmvd/steam-reviews
           prepared into reviews/*.csv + enriched games.json.

When the same appid appears in both sources, source1 wins. The clean stage keeps
long reviews, drops games below the review-count threshold, and prepends the
three Steam description fields. The split stage preserves review_id/sentence_id.

Durable corpus artifacts:

  <data-dir>/text_h5.h5       all sentence text + review/game metadata
  <data-dir>/embedding_h5.h5  same metadata + Qwen sentence vectors

No per-game embedded JSON, no NPZ corpus, and no separate conversion step.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SOURCE1_DOWNLOAD_URL = "https://data.mendeley.com/public-api/zip/jxy85cr3th/download/2"
KAGGLE_DATASET = "andrewmvd/steam-reviews"

DEFAULT_DATA_DIR = SCRIPT_DIR
DEFAULT_TEXT_H5 = DEFAULT_DATA_DIR / "text_h5.h5"
DEFAULT_EMBEDDING_H5 = DEFAULT_DATA_DIR / "embedding_h5.h5"
DEFAULT_TAP_MAPPING = PROJECT_ROOT / "VICReg_review" / "tags" / "tap_mapping.json"

PIPELINE_STAGES = ("metadata", "split", "text-h5", "embed-h5")


def atomic_json_write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def source1_done(source1_dir: Path) -> bool:
    reviews_dir = source1_dir / "Game Reviews"
    return (
        (source1_dir / "games.json").exists()
        and reviews_dir.exists()
        and any(reviews_dir.glob("*.csv"))
    )


def download_source1(source1_dir: Path, zip_cache: Path) -> bool:
    if source1_done(source1_dir):
        print(f"source1: already present at {source1_dir}", flush=True)
        return True

    if not zip_cache.exists():
        print(f"source1: downloading from Mendeley ...\n  {SOURCE1_DOWNLOAD_URL}", flush=True)
        zip_cache.parent.mkdir(parents=True, exist_ok=True)
        tmp_zip = zip_cache.with_suffix(".zip.tmp")
        try:
            req = urllib.request.Request(
                SOURCE1_DOWNLOAD_URL,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req) as resp:
                total = int(resp.headers.get("content-length") or 0)
                downloaded = 0
                chunk = 1 << 20
                with tmp_zip.open("wb") as file:
                    while True:
                        block = resp.read(chunk)
                        if not block:
                            break
                        file.write(block)
                        downloaded += len(block)
                        if total > 0 and downloaded % (500 << 20) < chunk:
                            pct = downloaded * 100 // total
                            print(f"  ... {pct}% ({downloaded >> 20} MB)", flush=True)
            tmp_zip.replace(zip_cache)
        except Exception as exc:
            tmp_zip.unlink(missing_ok=True)
            print(f"[warn] Mendeley download failed: {exc}\nsource1 will be skipped.", flush=True)
            return False
        print(f"source1: downloaded -> {zip_cache}", flush=True)
    else:
        print(f"source1: using cached zip {zip_cache}", flush=True)

    print(f"source1: extracting to {source1_dir.parent} ...", flush=True)
    try:
        with zipfile.ZipFile(zip_cache, "r") as zf:
            zf.extractall(source1_dir.parent)
    except Exception as exc:
        print(f"[warn] extraction failed: {exc}\nsource1 will be skipped.", flush=True)
        return False

    extracted = source1_dir
    if not extracted.is_dir():
        for candidate in source1_dir.parent.iterdir():
            if candidate.is_dir() and (candidate / "games.json").exists():
                extracted = candidate
                break

    if not extracted.is_dir():
        print(
            f"[warn] outer zip extracted but no folder with games.json found under {source1_dir.parent}.",
            flush=True,
        )
        return False

    inner_zip = extracted / "Game Reviews.zip"
    reviews_dir = extracted / "Game Reviews"
    if inner_zip.exists() and not (reviews_dir.exists() and any(reviews_dir.glob("*.csv"))):
        print(f"source1: extracting inner zip {inner_zip.name} ...", flush=True)
        try:
            with zipfile.ZipFile(inner_zip, "r") as zf:
                zf.extractall(extracted)
        except Exception as exc:
            print(f"[warn] inner zip extraction failed: {exc}\nsource1 will be skipped.", flush=True)
            return False

    if extracted != source1_dir:
        print(f"source1: renaming {extracted.name!r} -> {source1_dir.name!r}", flush=True)
        extracted.rename(source1_dir)

    if not source1_done(source1_dir):
        print(
            f"[warn] extraction finished but Game Reviews/*.csv still not found under {source1_dir}.",
            flush=True,
        )
        return False

    print(f"source1: ready ({source1_dir})", flush=True)
    return True


def prepared_done(prepared_dir: Path) -> bool:
    return (
        (prepared_dir / "prepare_manifest.json").exists()
        and (prepared_dir / "games.json").exists()
        and any((prepared_dir / "reviews").glob("*.csv"))
    )


def download_and_prepare_kaggle(args, source2_dir: Path, kaggle_cache: Path) -> bool:
    prepared_dir = source2_dir
    if prepared_done(prepared_dir) and not args.overwrite:
        print(f"source2: prepared data already exists at {prepared_dir}", flush=True)
        return True

    kaggle_input = getattr(args, "kaggle_input", None)
    if kaggle_input is None:
        try:
            import kagglehub
        except ImportError:
            print(
                "[warn] kagglehub not installed. Install with:\n"
                f"  {sys.executable} -m pip install kagglehub\n"
                "source2 will be skipped.",
                flush=True,
            )
            return False

        old_cache = os.environ.get("KAGGLEHUB_CACHE")
        os.environ["KAGGLEHUB_CACHE"] = str(kaggle_cache)
        try:
            print(f"source2: downloading {KAGGLE_DATASET} ...", flush=True)
            kaggle_input = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        except Exception as exc:
            print(f"[warn] Kaggle download failed: {exc}\nsource2 will be skipped.", flush=True)
            return False
        finally:
            if old_cache is None:
                os.environ.pop("KAGGLEHUB_CACHE", None)
            else:
                os.environ["KAGGLEHUB_CACHE"] = old_cache
        print(f"source2: downloaded to {kaggle_input}", flush=True)

    cmd = [
        str(args.python),
        str(SCRIPT_DIR / "prepare_kaggle_steam_reviews.py"),
        "--input",
        str(kaggle_input),
        "--output-dir",
        str(prepared_dir),
        "--min-length",
        str(args.min_length),
        "--min-count",
        str(args.min_count),
        "--chunksize",
        str(args.prepare_chunksize),
    ]
    cmd.append("--strict-length" if args.strict_length else "--no-strict-length")
    cmd.append("--strict-count" if args.strict_count else "--no-strict-count")
    if args.overwrite:
        cmd.append("--overwrite")
    print("RUN " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=True)

    if not args.skip_enrich:
        enrich_cmd = [
            str(args.python),
            str(SCRIPT_DIR / "enrich_steam_store_metadata.py"),
            "--games-json",
            str(prepared_dir / "games.json"),
            "--batch-size",
            str(args.enrich_batch_size),
            "--sleep",
            str(args.enrich_sleep),
            "--retry-sleep",
            str(args.enrich_retry_sleep),
            "--retries",
            str(args.enrich_retries),
        ]
        if args.overwrite:
            enrich_cmd.append("--overwrite-cache")
        print("RUN " + " ".join(str(c) for c in enrich_cmd), flush=True)
        subprocess.run(enrich_cmd, cwd=str(SCRIPT_DIR), check=True)

    return prepared_done(prepared_dir)


def merge_games_json(sources: list[Path], output_path: Path, overwrite: bool) -> dict:
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
            key = str(appid)
            if key not in merged:
                merged[key] = record
                added += 1
        print(f"  {path}: {len(data)} records, {added} new", flush=True)

    atomic_json_write(merged, output_path)
    print(f"merge_games_json: {len(merged)} total appids -> {output_path}", flush=True)
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Root directory for downloads and outputs: games.json, metadata/, "
            "sentences/, text_h5.h5, embedding_h5.h5."
        ),
    )

    parser.add_argument("--skip-source1", action="store_true")
    parser.add_argument("--skip-source1-download", action="store_true")
    parser.add_argument("--skip-source2", action="store_true")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Kaggle download and use existing prepared source2 data.")
    parser.add_argument("--skip-enrich", action="store_true")
    parser.add_argument("--kaggle-input", type=Path, default=None)

    parser.add_argument("--prepare-chunksize", type=int, default=200_000)
    parser.add_argument("--strict-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-count", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enrich-batch-size", type=int, default=1)
    parser.add_argument("--enrich-sleep", type=float, default=2.0)
    parser.add_argument("--enrich-retry-sleep", type=float, default=10.0)
    parser.add_argument("--enrich-retries", type=int, default=5)

    parser.add_argument("--only", nargs="+", choices=PIPELINE_STAGES, default=None)
    parser.add_argument("--skip", nargs="+", choices=PIPELINE_STAGES, default=[])
    parser.add_argument("--skip-merge-games-json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--min-length", type=int, default=300)
    parser.add_argument("--min-count", type=int, default=500)

    parser.add_argument("--split-model", default="sat-3l-sm")
    parser.add_argument("--split-device", default=None)
    parser.add_argument("--chunk-size", type=int, default=2000)

    parser.add_argument("--text-h5", type=Path, default=None)
    parser.add_argument("--embedding-h5", type=Path, default=None)
    parser.add_argument("--limit-files", type=int, default=0,
                        help="Debug limit for sentence JSON files when building text H5.")
    parser.add_argument("--text-chunk-rows", type=int, default=8192)
    parser.add_argument("--tap-mapping", type=Path, default=DEFAULT_TAP_MAPPING)
    parser.add_argument("--no-tap-labels", action="store_true")

    parser.add_argument("--backend", choices=["local", "cloud"], default="local")
    parser.add_argument("--local-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-in-flight", type=int, default=None)
    parser.add_argument("--read-batch-size", type=int, default=4096)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--embedding-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--embedding-chunk-rows", type=int, default=2048)
    parser.add_argument("--embedding-compression", choices=["none", "gzip", "lzf"], default="none")
    parser.add_argument("--gzip-level", type=int, default=1)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    source1_dir = data_dir / "Steam Games Metadata and Player Reviews (2020–2024"
    source1_zip = data_dir / "mendeley_steam_reviews.zip"
    source2_dir = data_dir / "kaggle_steam_reviews_prepared"
    kaggle_cache = data_dir / "kagglehub_cache"
    metadata_dir = data_dir / "metadata"
    sentences_dir = data_dir / "sentences"
    games_json_path = data_dir / "games.json"
    text_h5 = args.text_h5 or (data_dir / "text_h5.h5")
    embedding_h5 = args.embedding_h5 or (data_dir / "embedding_h5.h5")

    run = set(args.only) if args.only else set(PIPELINE_STAGES)
    run -= set(args.skip)

    review_dirs: list[Path] = []
    games_json_sources: list[Path] = []

    if not args.skip_source1:
        s1_ready = source1_done(source1_dir)
        if not s1_ready and not args.skip_source1_download:
            s1_ready = download_source1(source1_dir, source1_zip)
        if s1_ready:
            s1_reviews = source1_dir / "Game Reviews"
            if s1_reviews.exists() and any(s1_reviews.glob("*.csv")):
                review_dirs.append(s1_reviews)
            s1_games_json = source1_dir / "games.json"
            if s1_games_json.exists():
                games_json_sources.append(s1_games_json)
        else:
            print("[warn] source1 not available, continuing without it.", flush=True)

    if not args.skip_source2:
        s2_ready = False
        if not args.skip_download:
            s2_ready = download_and_prepare_kaggle(args, source2_dir, kaggle_cache)
        elif prepared_done(source2_dir):
            s2_ready = True
        else:
            print(
                f"[warn] source2 prepared data not found at {source2_dir}. "
                "Run without --skip-download to fetch it.",
                flush=True,
            )

        if s2_ready:
            s2_reviews = source2_dir / "reviews"
            if s2_reviews.exists() and any(s2_reviews.glob("*.csv")):
                review_dirs.append(s2_reviews)
            s2_games_json = source2_dir / "games.json"
            if s2_games_json.exists():
                games_json_sources.append(s2_games_json)

    if not review_dirs:
        raise SystemExit(
            "No review CSV directories found. Both sources failed or were skipped.\n"
            f"  source1: {SOURCE1_DOWNLOAD_URL}\n"
            "  source2: install kagglehub + configure Kaggle credentials, then re-run."
        )

    print(
        f"=== unified game-review build ===\n"
        f"data-dir    : {data_dir}\n"
        f"text-h5     : {text_h5}\n"
        f"embedding-h5: {embedding_h5}\n"
        f"sources     : {[str(p) for p in review_dirs]}\n"
        f"stages      : {sorted(run)}\n"
        f"backend     : {args.backend}\n",
        flush=True,
    )

    if not args.skip_merge_games_json:
        print("\n--- merge games.json ---", flush=True)
        merge_games_json(games_json_sources, games_json_path, args.overwrite)

    if not games_json_path.exists():
        raise SystemExit(
            f"games.json not found at {games_json_path}. "
            "Run without --skip-merge-games-json or supply it manually."
        )

    if "metadata" in run:
        print("\n--- stage 1/4: metadata (clean + filter + prepend descriptions) ---", flush=True)
        from build_metadata import build_metadata

        build_metadata(
            reviews_dirs=review_dirs,
            games_json=games_json_path,
            output_dir=metadata_dir,
            min_length=args.min_length,
            min_count=args.min_count,
            with_meta=True,
            overwrite=args.overwrite,
        )

    if "split" in run:
        print("\n--- stage 2/4: split (SaT sentence splitter) ---", flush=True)
        from split_data import split_data

        split_data(
            input_dir=metadata_dir,
            output_dir=sentences_dir,
            model=args.split_model,
            device=args.split_device,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )

    if "text-h5" in run:
        print("\n--- stage 3/4: text-h5 (unified text corpus) ---", flush=True)
        from h5_corpus import build_text_h5

        build_text_h5(
            sentences_dir=sentences_dir,
            games_json=games_json_path,
            output_h5=text_h5,
            overwrite=args.overwrite,
            limit_files=args.limit_files,
            chunk_rows=args.text_chunk_rows,
            tap_mapping=args.tap_mapping,
            no_tap_labels=args.no_tap_labels,
            reviews_dirs=review_dirs,
            label_min_length=args.min_length,
        )

    if "embed-h5" in run:
        print("\n--- stage 4/4: embed-h5 (stream text H5 -> embedding H5) ---", flush=True)
        from embedding_data import embed_data

        embed_data(
            input_h5=text_h5,
            output_h5=embedding_h5,
            backend=args.backend,
            overwrite=args.overwrite,
            local_model=args.local_model,
            device=args.embed_device,
            base_url=args.base_url,
            token_file=args.token_file,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            max_in_flight=args.max_in_flight,
            read_batch_size=args.read_batch_size,
            normalize=args.normalize,
            dtype=args.embedding_dtype,
            chunk_rows=args.embedding_chunk_rows,
            compression=args.embedding_compression,
            gzip_level=args.gzip_level,
        )

    elapsed = time.time() - started
    print(
        f"\n=== build done in {elapsed:.0f}s ===\n"
        f"text-h5     : {text_h5}\n"
        f"embedding-h5: {embedding_h5}",
        flush=True,
    )


if __name__ == "__main__":
    main()
