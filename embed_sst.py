"""Embed the SST `default_*` splits, keeping each sentence's label next to its
embedding.

For every ``sst/clean/default_<split>.csv`` (split = test/dev/train) this reads
the ``sentence`` and ``label`` columns, embeds the sentences with the remote TEI
endpoint, and writes one JSON file per split to
``sst/clean/sentence_embeddings/default_<split>.json`` as a list of records:

    {"sentence": "...", "label": 0.5, "embedding": [ ... 1024 floats ... ]}

The robust client (token from ``tokenAPI.txt``, batched concurrent requests,
transient-error retries) and the bounded streaming writer are reused from
``cloud_embedding.py`` so memory stays flat and disk writes are spread out.
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from cloud_embedding import (
    DEFAULT_BASE_URL,
    DEFAULT_TOKEN_FILE,
    EmbeddingClient,
    chunked,
    load_token,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = "sst/clean"
DEFAULT_OUTPUT_DIR = "sst/clean/sentence_embeddings"
DEFAULT_SPLITS = ["test", "dev", "train"]
DEFAULT_CONCURRENCY = 4
DEFAULT_BATCH_SIZE = 32


def resolve_script_relative(path):
    path = Path(path)
    return path if path.is_absolute() else SCRIPT_DIR / path


def load_rows(csv_path):
    """Return ordered [{'sentence', 'label'}], skipping rows with empty text."""
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            if not sentence:
                continue
            rows.append({"sentence": sentence, "label": float(row["label"])})
    return rows


def embed_records_streaming(records, client, executor, batch_size, out_file, max_in_flight):
    """Embed records' sentences and stream {sentence,label,embedding} objects into
    ``out_file`` as a JSON array, in order, bounding memory to ~max_in_flight batches.
    Returns (written_count, embedding_dim).
    """
    batches = [batch for _, batch in chunked(records, batch_size)]
    n = len(batches)

    in_flight = {}      # future -> batch index
    completed = {}      # batch index -> vectors
    next_submit = 0
    next_write = 0
    count = 0
    dim = 0
    separator = ""
    out_file.write("[")

    def fill_window():
        nonlocal next_submit
        while next_submit < n and len(in_flight) < max_in_flight:
            sentences = [record["sentence"] for record in batches[next_submit]]
            future = executor.submit(client.embed_batch, sentences)
            in_flight[future] = next_submit
            next_submit += 1

    fill_window()
    while in_flight:
        done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
        for future in done:
            completed[in_flight.pop(future)] = future.result()

        while next_write in completed:
            vectors = completed.pop(next_write)
            for record, vector in zip(batches[next_write], vectors):
                out_file.write(separator)
                out_file.write(
                    json.dumps(
                        {"sentence": record["sentence"], "label": record["label"], "embedding": vector},
                        ensure_ascii=False,
                    )
                )
                separator = ","
                count += 1
                if dim == 0 and vector:
                    dim = len(vector)
            next_write += 1

        fill_window()

    out_file.write("]")
    return count, dim


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--concurrency", default=DEFAULT_CONCURRENCY, type=int)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--max-in-flight", default=None, type=int)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_in_flight is None:
        args.max_in_flight = args.concurrency
    input_dir = resolve_script_relative(args.input_dir).resolve()
    output_dir = resolve_script_relative(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    token = load_token(args.token_file)
    client = EmbeddingClient(args.base_url, token, normalize=args.normalize)

    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        for split in args.splits:
            csv_path = input_dir / f"default_{split}.csv"
            output_path = output_dir / f"default_{split}.json"
            if not csv_path.exists():
                print(f"[{split}] SKIP: {csv_path} not found", file=sys.stderr)
                continue
            if output_path.exists() and not args.overwrite:
                print(f"[{split}] Skipping (already embedded): {output_path.name}")
                continue

            records = load_rows(csv_path)
            if not records:
                output_path.write_text("[]", encoding="utf-8")
                print(f"[{split}] empty, wrote []")
                continue

            started = time.time()
            tmp_path = output_path.with_suffix(".json.tmp")
            try:
                with tmp_path.open("w", encoding="utf-8") as file:
                    count, dim = embed_records_streaming(
                        records, client, executor, args.batch_size, file, args.max_in_flight
                    )
                tmp_path.replace(output_path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise

            print(
                f"[{split}] {len(records)} sentences -> {count} records "
                f"(label+embedding, dim {dim}) in {time.time() - started:.1f}s -> {output_path.name}"
            )
    finally:
        executor.shutdown(wait=True)

    print(f"Done. SST embeddings written to {output_dir}")


if __name__ == "__main__":
    main()
