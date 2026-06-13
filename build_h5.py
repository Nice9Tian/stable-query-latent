import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_SCORE_COLUMNS = [
    "psychological_meaning",
    "psychological_mastery",
    "psychological_curiosity",
    "psychological_autonomy",
    "psychological_immersion",
    "functional_progress_feedback",
    "functional_ease_of_control",
    "functional_audiovisual_appeal",
    "functional_goals_and_rules",
    "functional_challenge",
]

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_script_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def write_string_dataset(h5, name, values):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.asarray(values, dtype=object), dtype=string_dtype)


def average_benchmark_by_game(benchmark, score_columns):
    for column in score_columns:
        benchmark[column] = pd.to_numeric(benchmark[column], errors="raise")

    metadata_columns = [
        column
        for column in benchmark.columns
        if column not in set(score_columns) | {"game_id", "sample_index"}
    ]
    aggregations = {column: (column, "first") for column in metadata_columns}
    aggregations.update({column: (column, "mean") for column in score_columns})
    aggregations["score_sample_count"] = ("game_id", "size")

    averaged = (
        benchmark.groupby("game_id", sort=False, as_index=False)
        .agg(**aggregations)
        .reset_index(drop=True)
    )
    averaged["sample_index"] = np.arange(len(averaged), dtype=np.int64)
    return averaged


def build_h5(
    benchmark_csv,
    sentence_metadata_csv,
    embeddings_npy,
    output_h5,
    score_columns,
):
    benchmark_csv = resolve_script_path(benchmark_csv)
    sentence_metadata_csv = resolve_script_path(sentence_metadata_csv)
    embeddings_npy = resolve_script_path(embeddings_npy)
    output_h5 = resolve_script_path(output_h5)

    benchmark = pd.read_csv(
        benchmark_csv,
        dtype={"game_id": str, "genre_id": str, "game_name": str, "genre_name": str},
    )
    sentence_metadata = pd.read_csv(
        sentence_metadata_csv,
        dtype={"game_id": str, "genre_id": str, "game_name": str, "genre_name": str},
    )
    embeddings = np.load(embeddings_npy).astype(np.float32)

    missing_scores = [column for column in score_columns if column not in benchmark.columns]
    if missing_scores:
        raise ValueError(f"Missing score columns in benchmark: {missing_scores}")

    benchmark = average_benchmark_by_game(benchmark, score_columns)

    if len(sentence_metadata) != len(embeddings):
        raise ValueError(
            "sentence_metadata row count does not match sentence_embeddings.npy: "
            f"{len(sentence_metadata)} != {len(embeddings)}"
        )

    missing_game_ids = sorted(set(benchmark["game_id"]) - set(sentence_metadata["game_id"]))
    if missing_game_ids:
        raise ValueError(
            "Some benchmark game_id values do not appear in sentence metadata: "
            f"{missing_game_ids[:20]}"
        )

    game_to_embedding_indices = {
        game_id: group.sort_values("sentence_index")["embedding_index"].to_numpy(dtype=np.int64)
        for game_id, group in sentence_metadata.groupby("game_id", sort=False)
    }
    sequence_lengths = benchmark["game_id"].map(
        lambda game_id: len(game_to_embedding_indices[game_id])
    ).to_numpy(dtype=np.int64)

    sample_count = len(benchmark)
    max_sequence_length = int(sequence_lengths.max())
    embedding_dim = int(embeddings.shape[1])
    inputs = np.zeros((sample_count, max_sequence_length, embedding_dim), dtype=np.float32)
    key_padding_mask = np.ones((sample_count, max_sequence_length), dtype=np.bool_)

    for row_index, game_id in enumerate(benchmark["game_id"]):
        embedding_indices = game_to_embedding_indices[game_id]
        length = len(embedding_indices)
        inputs[row_index, :length] = embeddings[embedding_indices]
        key_padding_mask[row_index, :length] = False

    targets = benchmark[score_columns].apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float32
    )

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as h5:
        h5.attrs["source_benchmark_csv"] = str(benchmark_csv.resolve())
        h5.attrs["source_sentence_metadata_csv"] = str(sentence_metadata_csv.resolve())
        h5.attrs["source_embeddings_npy"] = str(embeddings_npy.resolve())
        h5.attrs["score_columns_json"] = json.dumps(score_columns)
        h5.attrs["input_layout"] = "game_average_sample, sentence_token, embedding_dim"
        h5.attrs["target_layout"] = "game_average_score"

        h5.create_dataset("inputs", data=inputs, compression="gzip", compression_opts=4)
        h5.create_dataset(
            "key_padding_mask",
            data=key_padding_mask,
            compression="gzip",
            compression_opts=4,
        )
        h5.create_dataset("targets", data=targets, compression="gzip", compression_opts=4)
        h5.create_dataset("sequence_lengths", data=sequence_lengths)
        h5.create_dataset(
            "benchmark_row_index",
            data=np.arange(sample_count, dtype=np.int64),
        )

        write_string_dataset(h5, "score_columns", score_columns)
        write_string_dataset(h5, "benchmark_game_id", benchmark["game_id"].tolist())
        write_string_dataset(h5, "benchmark_game_name", benchmark["game_name"].astype(str).tolist())
        write_string_dataset(h5, "benchmark_genre_id", benchmark["genre_id"].astype(str).tolist())
        write_string_dataset(h5, "benchmark_genre_name", benchmark["genre_name"].astype(str).tolist())
        h5.create_dataset(
            "benchmark_sample_index",
            data=benchmark["sample_index"].to_numpy(dtype=np.int64),
        )
        h5.create_dataset(
            "benchmark_score_sample_count",
            data=benchmark["score_sample_count"].to_numpy(dtype=np.int64),
        )

        metadata_group = h5.create_group("sentence_metadata")
        for column in sentence_metadata.columns:
            values = sentence_metadata[column]
            numeric_values = pd.to_numeric(values, errors="coerce")
            if numeric_values.notna().all():
                metadata_group.create_dataset(column, data=numeric_values.to_numpy())
            else:
                write_string_dataset(metadata_group, column, values.astype(str).tolist())

    return {
        "path": str(output_h5),
        "samples": sample_count,
        "max_sequence_length": max_sequence_length,
        "embedding_dim": embedding_dim,
        "output_dim": len(score_columns),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Package benchmark.csv, sentence metadata, and sentence embeddings into HDF5."
    )
    parser.add_argument("--benchmark-csv", default="benchmark.csv")
    parser.add_argument(
        "--sentence-metadata-csv",
        default="pseudo_text_sentence_embeddings/sentence_metadata.csv",
    )
    parser.add_argument(
        "--embeddings-npy",
        default="pseudo_text_sentence_embeddings/sentence_embeddings.npy",
    )
    parser.add_argument("--output-h5", default="benchmark_sentence_latent_query.h5")
    parser.add_argument("--score-columns", nargs="*", default=DEFAULT_SCORE_COLUMNS)
    args = parser.parse_args()

    summary = build_h5(
        args.benchmark_csv,
        args.sentence_metadata_csv,
        args.embeddings_npy,
        args.output_h5,
        args.score_columns,
    )
    print(
        "wrote {path}: samples={samples}, max_tokens={max_sequence_length}, "
        "embedding_dim={embedding_dim}, output_dim={output_dim}".format(**summary)
    )


if __name__ == "__main__":
    main()
