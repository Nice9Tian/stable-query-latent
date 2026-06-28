"""Pre-embed AO/Cyberpunk disturbance texts into eval npz caches.

The final sweep uses two text-eval caches:

* ``test_case_embeddings.npz`` for identity retrieval and same-game cosine.
* ``text_variant_embedding_cache.npz`` for real-text TAG recall/drop metrics.

This script builds those caches before training/evaluation so later tests only
load vectors from npz files instead of embedding raw text on the fly.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "game_review_data") not in sys.path:
    sys.path.insert(0, str(ROOT / "game_review_data"))

from game_review_data.embedding_data import DEFAULT_LOCAL_MODEL, LocalEmbedder  # noqa: E402

DEFAULT_H5 = ROOT / "game_review_data" / "embedding_h5.h5"
DEFAULT_OUT_DIR = SCRIPT_DIR / "heads" / "data_view_sweep"
DEFAULT_TEXT_VARIANT_DIR = ROOT

TEST_CASES = (
    ("Cyberpunk 2077", "1091500", "neutral", ROOT / "2077_text.txt"),
    ("Cyberpunk 2077", "1091500", "positive", ROOT / "2077_text_postive.txt"),
    ("Cyberpunk 2077", "1091500", "negative", ROOT / "2077_text_negative.txt"),
    ("Cyberpunk 2077", "1091500", "noname", ROOT / "2077_noname.txt"),
    ("Across the Obelisk", "1385380", "neutral", ROOT / "AO_text.txt"),
    ("Across the Obelisk", "1385380", "positive", ROOT / "AO_text_postive.txt"),
    ("Across the Obelisk", "1385380", "negative", ROOT / "AO_text_negative.txt"),
    ("Across the Obelisk", "1385380", "noname", ROOT / "AO_text_noname.txt"),
)

VARIANTS = ("positive", "neutral", "negative", "noname")
VARIANT_ALIASES = {
    "positive": ("positive", "postive", "pos"),
    "neutral": ("neutral", "middle", "base"),
    "negative": ("negative", "neg"),
    "noname": ("noname", "no_name", "nameless", "deidentified", "deidentified_names"),
}


@dataclass(frozen=True)
class TestCaseRecord:
    game: str
    appid: str
    sentiment: str
    path: Path


@dataclass(frozen=True)
class VariantRecord:
    appid: str
    name: str
    variant: str
    path: Path


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def split_text(text: str, max_sentences: int) -> list[str]:
    parts = re.split(r"(?:\r?\n)+|(?<=[.!?。！？；;])\s*", str(text).strip())
    sentences = [part.strip() for part in parts if part.strip()]
    if not sentences and str(text).strip():
        sentences = [str(text).strip()]
    return sentences[:max_sentences]


def atomic_npz_write(path: Path, **payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_npz_payload(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def test_cases_from_defaults() -> list[TestCaseRecord]:
    return [
        TestCaseRecord(game, str(appid), sentiment, Path(path))
        for game, appid, sentiment, path in TEST_CASES
        if Path(path).exists()
    ]


def load_h5_names(h5_path: Path) -> tuple[list[str], list[str]]:
    with h5py.File(h5_path, "r") as h5:
        names = [decode(x) for x in h5["game_names"][:]]
        if "appids" in h5:
            appids = [decode(x) for x in h5["appids"][:]]
        else:
            appids = [name.split("_", 1)[0] for name in names]
    return names, appids


def legacy_text_path(root: Path, appid: str, variant: str) -> Path | None:
    legacy = {
        "1091500": {
            "positive": root / "2077_text_postive.txt",
            "neutral": root / "2077_text.txt",
            "negative": root / "2077_text_negative.txt",
            "noname": root / "2077_noname.txt",
        },
        "1385380": {
            "positive": root / "AO_text_postive.txt",
            "neutral": root / "AO_text.txt",
            "negative": root / "AO_text_negative.txt",
            "noname": root / "AO_text_noname.txt",
        },
    }
    path = legacy.get(str(appid), {}).get(variant)
    return path if path is not None and path.exists() else None


def find_variant_path(root: Path, appid: str, variant: str) -> Path | None:
    aliases = VARIANT_ALIASES[variant]
    candidates = []
    for alias in aliases:
        candidates.extend([
            root / str(appid) / f"{alias}.txt",
            root / str(appid) / f"text_{alias}.txt",
            root / f"{appid}_{alias}.txt",
            root / f"{appid}_text_{alias}.txt",
            root / f"{appid}_{alias}_text.txt",
        ])
    for path in candidates:
        if path.exists():
            return path
    return legacy_text_path(root, appid, variant)


def discover_variant_records(root: Path, names: list[str], appids: list[str]) -> list[VariantRecord]:
    if not Path(root).exists():
        return []
    records = []
    for name, appid in zip(names, appids):
        for variant in VARIANTS:
            path = find_variant_path(Path(root), str(appid), variant)
            if path is not None:
                records.append(VariantRecord(str(appid), name, variant, path))
    return records


def _paths(records: Iterable) -> list[Path]:
    return [Path(record.path).resolve() for record in records]


def _source_signature(paths: list[Path]) -> tuple[list[str], list[int], list[int]]:
    return (
        [str(path) for path in paths],
        [int(path.stat().st_mtime_ns) for path in paths],
        [int(path.stat().st_size) for path in paths],
    )


def _cache_is_fresh(
    cache_path: Path,
    paths: list[Path],
    local_model: str,
    max_text_sentences: int,
    variants: list[str] | None = None,
) -> bool:
    if not Path(cache_path).exists():
        return False
    data = np.load(cache_path, allow_pickle=True)
    expected_paths, expected_mtime_ns, expected_sizes = _source_signature(paths)
    cached_paths = [str(x) for x in data["paths"]] if "paths" in data else []
    cached_mtime_ns = [int(x) for x in data["source_mtime_ns"]] if "source_mtime_ns" in data else []
    cached_sizes = [int(x) for x in data["source_sizes"]] if "source_sizes" in data else []
    cached_model = str(data["embedding_model"][0]) if "embedding_model" in data else None
    cached_max_sentences = int(data["max_text_sentences"][0]) if "max_text_sentences" in data else None
    if (
        cached_paths != expected_paths
        or cached_mtime_ns != expected_mtime_ns
        or cached_sizes != expected_sizes
        or cached_model != str(local_model)
        or cached_max_sentences != int(max_text_sentences)
    ):
        return False
    if variants is not None:
        cached_variants = [str(x) for x in data["variants"]] if "variants" in data else []
        if cached_variants != [str(x) for x in variants]:
            return False
    return True


def _embed_record_paths(
    records: Iterable,
    local_model: str,
    device: str | None,
    batch_size: int,
    max_text_sentences: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    records = list(records)
    embedder = LocalEmbedder(local_model, device=device, batch_size=batch_size)
    vectors = []
    offsets = [0]
    sentence_counts = []
    texts = []
    try:
        for record in records:
            text = Path(record.path).read_text(encoding="utf-8")
            texts.append(text)
            sentences = split_text(text, int(max_text_sentences))
            if not sentences:
                raise ValueError(f"{record.path} contains no text.")
            embedded = np.asarray(embedder.embed(sentences), dtype=np.float32)
            vectors.append(embedded)
            offsets.append(offsets[-1] + embedded.shape[0])
            sentence_counts.append(len(sentences))
            label = getattr(record, "variant", getattr(record, "sentiment", "text"))
            print(
                f"disturbtion embed appid={record.appid} kind={label} sentences={len(sentences)}",
                flush=True,
            )
    finally:
        del embedder
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return (
        np.concatenate(vectors, axis=0).astype(np.float32),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(sentence_counts, dtype=np.int32),
        np.asarray(texts, dtype=object),
    )


def ensure_test_case_cache(
    cache_path: Path,
    local_model: str = DEFAULT_LOCAL_MODEL,
    device: str | None = None,
    batch_size: int = 32,
    max_text_sentences: int = 4096,
    rebuild: bool = False,
) -> Path:
    records = test_cases_from_defaults()
    if not records:
        if Path(cache_path).exists() and not rebuild:
            print(f"disturbtion test-case cache exists; source texts not required: {cache_path}", flush=True)
            return Path(cache_path)
        raise FileNotFoundError("No AO/Cyberpunk test-case text files were found.")
    paths = _paths(records)
    if not rebuild and _cache_is_fresh(cache_path, paths, local_model, max_text_sentences):
        print(f"disturbtion test-case cache is fresh: {cache_path}", flush=True)
        return Path(cache_path)

    vectors, offsets, sentence_counts, texts = _embed_record_paths(
        records, local_model, device, batch_size, max_text_sentences
    )
    source_paths, source_mtime_ns, source_sizes = _source_signature(paths)
    atomic_npz_write(
        Path(cache_path),
        kind=np.asarray(["disturbtion_test_cases"], dtype=object),
        vectors=vectors,
        offsets=offsets,
        games=np.asarray([record.game for record in records], dtype=object),
        appids=np.asarray([record.appid for record in records], dtype=object),
        sentiments=np.asarray([record.sentiment for record in records], dtype=object),
        paths=np.asarray(source_paths, dtype=object),
        texts=texts,
        source_mtime_ns=np.asarray(source_mtime_ns, dtype=np.int64),
        source_sizes=np.asarray(source_sizes, dtype=np.int64),
        sentence_counts=sentence_counts,
        embedding_model=np.asarray([str(local_model)], dtype=object),
        max_text_sentences=np.asarray([int(max_text_sentences)], dtype=np.int32),
        created_at=np.asarray([time.strftime("%Y-%m-%dT%H:%M:%S")], dtype=object),
    )
    print(f"wrote disturbtion test-case cache -> {cache_path}", flush=True)
    return Path(cache_path)


def ensure_text_variant_cache(
    cache_path: Path,
    records: Iterable,
    local_model: str = DEFAULT_LOCAL_MODEL,
    device: str | None = None,
    batch_size: int = 32,
    max_text_sentences: int = 4096,
    rebuild: bool = False,
) -> Path | None:
    records = list(records)
    if not records:
        if Path(cache_path).exists() and not rebuild:
            print(f"disturbtion text-variant cache exists; source texts not required: {cache_path}", flush=True)
            return Path(cache_path)
        print("disturbtion text-variant cache skipped: no text variant records found.", flush=True)
        return None
    paths = _paths(records)
    variants = [str(record.variant) for record in records]
    if not rebuild and _cache_is_fresh(cache_path, paths, local_model, max_text_sentences, variants=variants):
        print(f"disturbtion text-variant cache is fresh: {cache_path}", flush=True)
        return Path(cache_path)

    vectors, offsets, _sentence_counts, texts = _embed_record_paths(
        records, local_model, device, batch_size, max_text_sentences
    )
    source_paths, source_mtime_ns, source_sizes = _source_signature(paths)
    atomic_npz_write(
        Path(cache_path),
        kind=np.asarray(["disturbtion_text_variants"], dtype=object),
        vectors=vectors,
        offsets=offsets,
        appids=np.asarray([record.appid for record in records], dtype=object),
        names=np.asarray([record.name for record in records], dtype=object),
        variants=np.asarray(variants, dtype=object),
        paths=np.asarray(source_paths, dtype=object),
        texts=texts,
        source_mtime_ns=np.asarray(source_mtime_ns, dtype=np.int64),
        source_sizes=np.asarray(source_sizes, dtype=np.int64),
        embedding_model=np.asarray([str(local_model)], dtype=object),
        max_text_sentences=np.asarray([int(max_text_sentences)], dtype=np.int32),
        created_at=np.asarray([time.strftime("%Y-%m-%dT%H:%M:%S")], dtype=object),
    )
    print(f"wrote disturbtion text-variant cache -> {cache_path}", flush=True)
    return Path(cache_path)


def ensure_eval_caches(args) -> None:
    out_dir = Path(args.out_dir)
    test_case_cache = Path(args.test_case_cache or (out_dir / "test_case_embeddings.npz"))
    text_variant_cache = Path(args.text_variant_cache or (out_dir / "text_variant_embedding_cache.npz"))

    if not args.skip_test_cases:
        ensure_test_case_cache(
            test_case_cache,
            local_model=args.local_model,
            device=args.device,
            batch_size=args.batch_size,
            max_text_sentences=args.max_text_sentences,
            rebuild=args.rebuild or args.rebuild_test_case_cache,
        )

    if not args.skip_text_variants:
        names, appids = load_h5_names(Path(args.h5))
        records = discover_variant_records(Path(args.text_variant_dir), names, appids)
        ensure_text_variant_cache(
            text_variant_cache,
            records,
            local_model=args.local_model,
            device=args.device,
            batch_size=args.batch_size,
            max_text_sentences=args.max_text_sentences,
            rebuild=args.rebuild or args.rebuild_text_variant_cache,
        )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default=DEFAULT_H5, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--text-variant-dir", default=DEFAULT_TEXT_VARIANT_DIR, type=Path)
    parser.add_argument("--test-case-cache", default=None, type=Path)
    parser.add_argument("--text-variant-cache", default=None, type=Path)
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-text-sentences", type=int, default=4096)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--rebuild-test-case-cache", action="store_true")
    parser.add_argument("--rebuild-text-variant-cache", action="store_true")
    parser.add_argument("--skip-test-cases", action="store_true")
    parser.add_argument("--skip-text-variants", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ensure_eval_caches(parse_args(argv))


if __name__ == "__main__":
    main()
