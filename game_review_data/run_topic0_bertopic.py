"""Run a second-level BERTopic model on parent Topic 0 from berttopic_2.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from run_game_review_bertopic import (
    DEFAULT_CACHE_DIR,
    DEFAULT_INPUT_H5,
    PROJECT_ROOT,
    atomic_save_npy,
    atomic_write_text,
    build_balanced_prefix_sample,
    markdown_escape,
    package_versions,
    topic_words,
)


DEFAULT_OUTPUT_MD = PROJECT_ROOT / "berttopic_topic0.md"


def parent_topics_cache_path(
    cache_dir: Path,
    sample_meta: dict,
    max_docs: int,
    min_chars: int,
    skip_metadata: bool,
    n_neighbors: int,
    min_cluster_size: int,
    vectorizer_min_df: int,
    random_state: int,
) -> Path:
    corpus_format = sample_meta.get("corpus_format", "json")
    input_key = sample_meta.get("input_h5") or "missing_h5"
    suffix = hashlib.sha1(str(input_key).encode("utf-8")).hexdigest()[:8]
    tag = (
        f"parent_topics_{corpus_format}_{suffix}_n{max_docs}"
        f"_minchars{min_chars}_skipmeta{int(skip_metadata)}"
        f"_neighbors{n_neighbors}_mcs{min_cluster_size}"
        f"_vmin{vectorizer_min_df}_seed{random_state}.npy"
    )
    return cache_dir / tag


def fit_bertopic(
    docs: list[str],
    embeddings: np.ndarray,
    n_neighbors: int,
    min_cluster_size: int,
    min_samples: int | None,
    cluster_selection_method: str,
    vectorizer_min_df: int,
    random_state: int,
):
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        low_memory=True,
        random_state=random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        prediction_data=False,
        core_dist_n_jobs=1,
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=vectorizer_min_df,
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
    return topic_model, np.asarray(topics, dtype=np.int32)


def get_parent_topics(args, docs: list[str], embeddings: np.ndarray, sample_meta: dict) -> np.ndarray:
    path = parent_topics_cache_path(
        args.cache_dir,
        sample_meta,
        args.max_docs,
        args.min_chars,
        not args.include_metadata_records,
        args.parent_n_neighbors,
        args.parent_min_cluster_size,
        args.parent_vectorizer_min_df,
        args.random_state,
    )
    if path.exists() and not args.overwrite_parent_topics:
        topics = np.load(path)
        if len(topics) == len(docs):
            print(f"Loaded cached parent topics from {path}")
            return topics
        print(f"Ignoring stale parent topic cache: {path}")

    print(
        "Fitting parent BERTopic for Topic 0 selection: "
        f"docs={len(docs)} n_neighbors={args.parent_n_neighbors} "
        f"min_cluster_size={args.parent_min_cluster_size}"
    )
    started = time.time()
    _, topics = fit_bertopic(
        docs,
        embeddings,
        n_neighbors=args.parent_n_neighbors,
        min_cluster_size=args.parent_min_cluster_size,
        min_samples=None,
        cluster_selection_method="eom",
        vectorizer_min_df=args.parent_vectorizer_min_df,
        random_state=args.random_state,
    )
    atomic_save_npy(path, topics)
    meta_path = path.with_suffix(".json")
    parent_counts = {str(topic): int((topics == topic).sum()) for topic in sorted(set(topics.tolist()))}
    atomic_write_text(
        meta_path,
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "fit_seconds": round(time.time() - started, 1),
                "n_neighbors": args.parent_n_neighbors,
                "min_cluster_size": args.parent_min_cluster_size,
                "vectorizer_min_df": args.parent_vectorizer_min_df,
                "random_state": args.random_state,
                "corpus_format": sample_meta.get("corpus_format"),
                "input_h5": sample_meta.get("input_h5"),
                "topic_counts": parent_counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    print(f"Cached parent topics to {path}")
    return topics


def build_report(
    output_md: Path,
    parent_topics: np.ndarray,
    parent_topic: int,
    topic_model: BERTopic,
    sub_topics: np.ndarray,
    docs: list[str],
    records: list[dict[str, str]],
    sample_meta: dict,
    args,
    fit_seconds: float,
) -> None:
    topic_info = topic_model.get_topic_info()
    topic_counts = defaultdict(int)
    examples: dict[int, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for topic, doc, record in zip(sub_topics, docs, records):
        topic_counts[int(topic)] += 1
        if len(examples[int(topic)]) < 3:
            examples[int(topic)].append((doc, record))

    non_outlier_topics = [int(topic) for topic in topic_counts if int(topic) != -1]
    outlier_count = int(topic_counts.get(-1, 0))
    parent_count = int((parent_topics == parent_topic).sum())
    parent_outliers = int((parent_topics == -1).sum())

    lines: list[str] = []
    lines.append("# Game Review BERTopic Topic 0 Refinement")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Input H5: `{markdown_escape(sample_meta['input_h5'])}`")
    lines.append(f"- Corpus format: `{sample_meta.get('corpus_format', 'json')}`")
    lines.append(f"- Base sample documents: {sample_meta['sampled_docs']}")
    lines.append(f"- Parent topic selected: {parent_topic}")
    lines.append(f"- Parent Topic {parent_topic} documents: {parent_count}")
    lines.append(f"- Parent outlier documents: {parent_outliers}")
    lines.append(f"- Embedding dimension: {sample_meta['embedding_dim']}")
    lines.append("")
    lines.append("## Parent Model")
    lines.append("")
    lines.append(f"- UMAP `n_neighbors`: {args.parent_n_neighbors}")
    lines.append(f"- HDBSCAN `min_cluster_size`: {args.parent_min_cluster_size}")
    lines.append("- HDBSCAN `min_samples`: default")
    lines.append("- HDBSCAN `cluster_selection_method`: `eom`")
    lines.append(f"- CountVectorizer `min_df`: {args.parent_vectorizer_min_df}")
    lines.append("")
    lines.append("## Subtopic Model")
    lines.append("")
    lines.append(f"- Documents: {len(docs)}")
    lines.append(f"- UMAP `n_neighbors`: {args.sub_n_neighbors}")
    lines.append(f"- HDBSCAN `min_cluster_size`: {args.sub_min_cluster_size}")
    lines.append(f"- HDBSCAN `min_samples`: {args.sub_min_samples}")
    lines.append(f"- HDBSCAN `cluster_selection_method`: `{args.sub_cluster_selection_method}`")
    lines.append(f"- CountVectorizer `min_df`: {args.sub_vectorizer_min_df}")
    lines.append(f"- Random state: {args.random_state}")
    lines.append(f"- Fit time: {fit_seconds:.1f} seconds")
    lines.append("")
    lines.append(
        "Note: these topic IDs are second-level IDs inside parent Topic 0. "
        "They are not the same namespace as the parent model topic IDs."
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
    lines.append(f"- Subtopics excluding outliers: {len(non_outlier_topics)}")
    lines.append(f"- Outlier documents: {outlier_count}")
    lines.append(f"- Outlier rate: {outlier_count / max(1, len(sub_topics)):.2%}")
    lines.append("")

    lines.append("## Subtopic Table")
    lines.append("")
    lines.append("| Subtopic | Count | Top Words |")
    lines.append("|---:|---:|---|")
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        words = "outliers" if topic_id == -1 else topic_words(topic_model, topic_id)
        lines.append(f"| {topic_id} | {int(row['Count'])} | {markdown_escape(words, max_chars=220)} |")
    lines.append("")

    lines.append("## Top Subtopic Examples")
    lines.append("")
    for _, row in topic_info[topic_info["Topic"] != -1].head(args.example_topics).iterrows():
        topic_id = int(row["Topic"])
        words = topic_words(topic_model, topic_id)
        lines.append(f"### Subtopic {topic_id} ({int(row['Count'])} docs)")
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
    parser.add_argument("--input-h5", "--input-dir", dest="input_h5", default=DEFAULT_INPUT_H5, type=Path)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, type=Path)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD, type=Path)
    parser.add_argument("--max-docs", default=100_000, type=int)
    parser.add_argument("--min-chars", default=20, type=int)
    parser.add_argument("--parent-topic", default=0, type=int)
    parser.add_argument("--parent-n-neighbors", default=100, type=int)
    parser.add_argument("--parent-min-cluster-size", default=100, type=int)
    parser.add_argument("--parent-vectorizer-min-df", default=1, type=int)
    parser.add_argument("--sub-n-neighbors", default=30, type=int)
    parser.add_argument("--sub-min-cluster-size", default=50, type=int)
    parser.add_argument("--sub-min-samples", default=10, type=int)
    parser.add_argument("--sub-cluster-selection-method", choices=["eom", "leaf"], default="leaf")
    parser.add_argument("--sub-vectorizer-min-df", default=5, type=int)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--example-topics", default=40, type=int)
    parser.add_argument("--include-metadata-records", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite-parent-topics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, docs, embeddings, sample_meta = build_balanced_prefix_sample(
        input_h5=args.input_h5,
        cache_dir=args.cache_dir,
        max_docs=args.max_docs,
        min_chars=args.min_chars,
        skip_metadata=not args.include_metadata_records,
        overwrite_cache=args.overwrite_cache,
    )
    parent_topics = get_parent_topics(args, docs, embeddings, sample_meta)
    mask = parent_topics == args.parent_topic
    selected_count = int(mask.sum())
    if not selected_count:
        raise ValueError(f"Parent topic {args.parent_topic} has no documents.")

    sub_docs = [doc for doc, keep in zip(docs, mask) if keep]
    sub_records = [record for record, keep in zip(records, mask) if keep]
    sub_embeddings = embeddings[mask]
    print(
        f"Fitting subtopic BERTopic: parent_topic={args.parent_topic} "
        f"docs={len(sub_docs)} n_neighbors={args.sub_n_neighbors} "
        f"min_cluster_size={args.sub_min_cluster_size} "
        f"min_samples={args.sub_min_samples} "
        f"method={args.sub_cluster_selection_method}"
    )
    started = time.time()
    topic_model, sub_topics = fit_bertopic(
        sub_docs,
        sub_embeddings,
        n_neighbors=args.sub_n_neighbors,
        min_cluster_size=args.sub_min_cluster_size,
        min_samples=args.sub_min_samples,
        cluster_selection_method=args.sub_cluster_selection_method,
        vectorizer_min_df=args.sub_vectorizer_min_df,
        random_state=args.random_state,
    )
    fit_seconds = time.time() - started
    build_report(
        output_md=args.output_md,
        parent_topics=parent_topics,
        parent_topic=args.parent_topic,
        topic_model=topic_model,
        sub_topics=sub_topics,
        docs=sub_docs,
        records=sub_records,
        sample_meta=sample_meta,
        args=args,
        fit_seconds=fit_seconds,
    )
    print(f"Wrote {args.output_md.resolve()}")


if __name__ == "__main__":
    main()
