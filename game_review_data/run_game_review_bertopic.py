"""Run BERTopic on game_review_cleaned_3_sentences and write a Markdown report.

The sentence directory stores one huge JSON object per game and each sentence
already has a Qwen embedding vector. A full in-memory BERTopic fit over the
entire 100GB sentence/vector directory is not practical on a 32GB workstation,
so this script builds a deterministic per-game sentence sample by default.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import ijson
import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "game_review_cleaned_3_sentences"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "game_review_bertopic.md"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "bertopic_cache"


def markdown_escape(value: object, max_chars: int | None = None) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if max_chars and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text.replace("|", "\\|")


def topic_words(topic_model: BERTopic, topic_id: int, limit: int = 10) -> str:
    words = []
    for word, _ in topic_model.get_topic(topic_id) or []:
        clean_word = " ".join(str(word).split())
        if clean_word:
            words.append(clean_word)
        if len(words) >= limit:
            break
    return ", ".join(words) if words else "no valid tokens"


def atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("wb") as file:
            np.save(file, array)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def package_versions() -> dict[str, str]:
    packages = [
        "bertopic",
        "hdbscan",
        "umap-learn",
        "ijson",
        "numpy",
        "scikit-learn",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def input_size_gb(input_dir: Path) -> float:
    total = sum(path.stat().st_size for path in input_dir.glob("*.json"))
    return total / (1024 ** 3)


def cache_paths(cache_dir: Path, max_docs: int, min_chars: int, skip_metadata: bool) -> tuple[Path, Path, Path]:
    tag = f"balanced_prefix_n{max_docs}_minchars{min_chars}_skipmeta{int(skip_metadata)}"
    return (
        cache_dir / f"{tag}_docs.json",
        cache_dir / f"{tag}_embeddings.npy",
        cache_dir / f"{tag}_meta.json",
    )


def load_cached_sample(docs_path: Path, embeddings_path: Path, meta_path: Path):
    if not (docs_path.exists() and embeddings_path.exists() and meta_path.exists()):
        return None
    records = json.loads(docs_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path, mmap_mode=None)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    docs = [record["text"] for record in records]
    return records, docs, embeddings, meta


def iter_sentence_items(path: Path, skip_metadata: bool):
    with path.open("rb") as file:
        for review_id, sentences in ijson.kvitems(file, "", use_float=True):
            if skip_metadata:
                try:
                    if int(review_id) < 3:
                        continue
                except ValueError:
                    pass
            if not isinstance(sentences, dict):
                continue
            for sentence_id, item in sentences.items():
                if not isinstance(item, dict):
                    continue
                text = item.get("sentence_text")
                vector = item.get("vector")
                if text and vector:
                    yield review_id, sentence_id, str(text), vector


def build_balanced_prefix_sample(
    input_dir: Path,
    cache_dir: Path,
    max_docs: int,
    min_chars: int,
    skip_metadata: bool,
    overwrite_cache: bool,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    docs_path, embeddings_path, meta_path = cache_paths(cache_dir, max_docs, min_chars, skip_metadata)
    if not overwrite_cache:
        cached = load_cached_sample(docs_path, embeddings_path, meta_path)
        if cached:
            print(f"Loaded cached sample: {len(cached[1])} docs from {cache_dir}")
            return cached

    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON files found in {input_dir}")

    per_file_limit = max(1, math.ceil(max_docs / len(files)))
    records: list[dict[str, str]] = []
    vectors: list[np.ndarray] = []
    file_counts: dict[str, int] = {}
    skipped_short = 0
    skipped_bad_vector = 0
    started = time.time()
    dim = None

    print(
        f"Building balanced prefix sample: files={len(files)} max_docs={max_docs} "
        f"per_file_limit={per_file_limit}"
    )
    for file_index, path in enumerate(files, start=1):
        kept_for_file = 0
        for review_id, sentence_id, text, vector in iter_sentence_items(path, skip_metadata=skip_metadata):
            if kept_for_file >= per_file_limit or len(records) >= max_docs:
                break
            clean_text = " ".join(text.split())
            if len(clean_text) < min_chars:
                skipped_short += 1
                continue
            array = np.asarray(vector, dtype=np.float32)
            if array.ndim != 1:
                skipped_bad_vector += 1
                continue
            if dim is None:
                dim = int(array.shape[0])
            if int(array.shape[0]) != dim:
                skipped_bad_vector += 1
                continue
            records.append(
                {
                    "text": clean_text,
                    "file": path.name,
                    "review_id": str(review_id),
                    "sentence_id": str(sentence_id),
                }
            )
            vectors.append(array)
            kept_for_file += 1
        file_counts[path.name] = kept_for_file
        if file_index % 10 == 0 or file_index == len(files):
            elapsed = time.time() - started
            print(
                f"[{file_index}/{len(files)}] sampled={len(records)} "
                f"elapsed={elapsed:.1f}s last={path.name}:{kept_for_file}"
            )
        if len(records) >= max_docs:
            break

    if not records:
        raise ValueError("No usable sentence/vector pairs were sampled.")

    embeddings = np.vstack(vectors).astype(np.float32, copy=False)
    meta = {
        "sample_method": "balanced_prefix_per_game",
        "input_dir": str(input_dir.resolve()),
        "input_files": len(files),
        "input_size_gb": round(input_size_gb(input_dir), 3),
        "max_docs": max_docs,
        "sampled_docs": len(records),
        "embedding_dim": int(embeddings.shape[1]),
        "per_file_limit": per_file_limit,
        "min_chars": min_chars,
        "skip_metadata_review_ids_lt_3": skip_metadata,
        "skipped_short": skipped_short,
        "skipped_bad_vector": skipped_bad_vector,
        "file_counts": file_counts,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    atomic_write_text(docs_path, json.dumps(records, ensure_ascii=False))
    atomic_save_npy(embeddings_path, embeddings)
    atomic_write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Cached sample under {cache_dir}")
    return records, [record["text"] for record in records], embeddings, meta


def fit_bertopic(docs: list[str], embeddings: np.ndarray, min_cluster_size: int, random_state: int):
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        low_memory=True,
        random_state=random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
        core_dist_n_jobs=1,
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=50_000,
        token_pattern=r"(?u)\b[^\W\d_][^\W_]{1,}\b",
    )
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings)
    return topic_model, topics


def build_report(
    output_md: Path,
    topic_model: BERTopic,
    topics: list[int],
    docs: list[str],
    records: list[dict[str, str]],
    sample_meta: dict,
    min_cluster_size: int,
    random_state: int,
    fit_seconds: float,
) -> None:
    topic_info = topic_model.get_topic_info()
    topic_counts = defaultdict(int)
    examples: dict[int, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for topic, doc, record in zip(topics, docs, records):
        topic_counts[int(topic)] += 1
        if len(examples[int(topic)]) < 3:
            examples[int(topic)].append((doc, record))

    non_outlier_topics = [int(topic) for topic in topic_counts if int(topic) != -1]
    outlier_count = int(topic_counts.get(-1, 0))

    lines: list[str] = []
    lines.append("# Game Review BERTopic")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Input: `{markdown_escape(sample_meta['input_dir'])}`")
    lines.append(f"- Input files: {sample_meta['input_files']}")
    lines.append(f"- Input directory size: {sample_meta['input_size_gb']} GB")
    lines.append(f"- Sample method: `{sample_meta['sample_method']}`")
    lines.append(f"- Sampled documents: {sample_meta['sampled_docs']}")
    lines.append(f"- Embedding dimension: {sample_meta['embedding_dim']}")
    lines.append(f"- HDBSCAN `min_cluster_size`: {min_cluster_size}")
    lines.append(f"- Random state: {random_state}")
    lines.append(f"- Fit time: {fit_seconds:.1f} seconds")
    lines.append("")
    lines.append(
        "Note: the source sentence/vector directory is about 100 GB. This run uses "
        "a deterministic per-game prefix sample and skips metadata records with "
        "`review_id < 3` before fitting BERTopic. It is a practical topic snapshot, "
        "not a full all-sentence HDBSCAN fit."
    )
    lines.append("")

    lines.append("## Environment")
    lines.append("")
    lines.append("| Package | Version |")
    lines.append("|---|---:|")
    for package, version in package_versions().items():
        lines.append(f"| `{package}` | `{version}` |")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Topics excluding outliers: {len(non_outlier_topics)}")
    lines.append(f"- Outlier documents: {outlier_count}")
    lines.append(f"- Outlier rate: {outlier_count / max(1, len(topics)):.2%}")
    lines.append("")

    lines.append("## Topic Table")
    lines.append("")
    lines.append("| Topic | Count | Top Words |")
    lines.append("|---:|---:|---|")
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            words = "outliers"
        else:
            words = topic_words(topic_model, topic_id)
        lines.append(
            f"| {topic_id} | {int(row['Count'])} | {markdown_escape(words, max_chars=220)} |"
        )
    lines.append("")

    lines.append("## Top Topic Examples")
    lines.append("")
    for _, row in topic_info[topic_info["Topic"] != -1].head(30).iterrows():
        topic_id = int(row["Topic"])
        words = topic_words(topic_model, topic_id)
        lines.append(f"### Topic {topic_id} ({int(row['Count'])} docs)")
        lines.append("")
        lines.append(f"Top words: {markdown_escape(words)}")
        lines.append("")
        for doc, record in examples.get(topic_id, []):
            source = f"{record['file']} review={record['review_id']} sentence={record['sentence_id']}"
            lines.append(f"- `{markdown_escape(source)}`: {markdown_escape(doc, max_chars=450)}")
        lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_md, "\n".join(lines).rstrip() + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, type=Path)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD, type=Path)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, type=Path)
    parser.add_argument("--max-docs", default=100_000, type=int)
    parser.add_argument("--min-chars", default=20, type=int)
    parser.add_argument("--min-cluster-size", default=100, type=int)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--include-metadata-records", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, docs, embeddings, sample_meta = build_balanced_prefix_sample(
        input_dir=args.input_dir,
        cache_dir=args.cache_dir,
        max_docs=args.max_docs,
        min_chars=args.min_chars,
        skip_metadata=not args.include_metadata_records,
        overwrite_cache=args.overwrite_cache,
    )
    print(
        f"Fitting BERTopic: docs={len(docs)} dim={embeddings.shape[1]} "
        f"min_cluster_size={args.min_cluster_size}"
    )
    started = time.time()
    topic_model, topics = fit_bertopic(
        docs,
        embeddings,
        min_cluster_size=args.min_cluster_size,
        random_state=args.random_state,
    )
    fit_seconds = time.time() - started
    build_report(
        output_md=args.output_md,
        topic_model=topic_model,
        topics=topics,
        docs=docs,
        records=records,
        sample_meta=sample_meta,
        min_cluster_size=args.min_cluster_size,
        random_state=args.random_state,
        fit_seconds=fit_seconds,
    )
    print(f"Wrote {args.output_md.resolve()}")


if __name__ == "__main__":
    main()
