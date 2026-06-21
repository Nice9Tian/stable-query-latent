"""Attach an embedding vector to every sentence in the nested
``game_review_cleaned_3_sentences`` files, augmenting them in place to:

    { "<review_id>": { "sentence_1": {"sentence_text": ..., "vector": [...]},
                        "sentence_2": {"sentence_text": ..., "vector": [...]}, ... }, ... }

For each file the nested structure is flattened in (review_id, sentence_id) order.
If the matching flat vector file in ``game_review_cleaned_3_sentence_embeddings`` has
exactly the same number of vectors (verified to align 1:1, cosine 1.0), those vectors
are reused; otherwise the sentences are (re)embedded via the cloud endpoint.

Reuse is the bulk of the work and is pure CPU/IO, so it runs across a process pool.
To stay within RAM on 60+ GB of vectors, reuse never parses the float values: it
splices each vector's raw JSON bytes from the flat file straight into the output
(zero-copy memoryview), streaming the result to disk. Progress is tracked in a
manifest so the run resumes cheaply.
"""

import argparse
import json
import mmap
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path

import orjson

from cloud_embedding import (
    DEFAULT_BASE_URL,
    DEFAULT_TOKEN_FILE,
    EmbeddingClient,
    chunked,
    load_token,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SENTENCES_DIR = "game_review_cleaned_3_sentences"
DEFAULT_VECTORS_DIR = "game_review_cleaned_3_sentence_embeddings"
MANIFEST_NAME = "embed_nested_progress.json"
DEFAULT_WORKERS = 10
DEFAULT_CONCURRENCY = 256
DEFAULT_BATCH_SIZE = 32


def flatten_positions(nested):
    """Ordered [(review_id, sentence_key)] over the nested dict (insertion order)."""
    return [(rid, skey) for rid, sentences in nested.items() for skey in sentences]


def split_top_level_vector_spans(data):
    """Return [(start, end)] byte spans of each top-level vector in a JSON array of
    arrays. Vectors contain only numbers (no nested brackets), so each vector runs
    from an inner '[' to the very next ']'. Uses C-level bytes/mmap .find so a 2 GB
    file isn't scanned one Python byte at a time. ``data`` may be bytes or an mmap."""
    spans = []
    i = data.find(b"[", data.find(b"[") + 1)  # first inner '['
    while i != -1:
        j = data.find(b"]", i + 1)            # the next ']' closes this vector
        if j == -1:
            break
        spans.append((i, j + 1))
        i = data.find(b"[", j + 1)
    return spans


def reuse_worker(task):
    """Process pool worker: reuse existing vectors for one file via raw byte splicing.
    The flat vector file is mmap'd (low resident memory, zero-copy slices) and the
    augmented nested structure is streamed to disk. Returns
    ('reused', name, n, dim) or ('needs_embed', name, n)."""
    name, sentences_dir, vectors_dir = task
    sentences_path = Path(sentences_dir) / name
    vectors_path = Path(vectors_dir) / name

    nested = orjson.loads(sentences_path.read_bytes())
    positions = flatten_positions(nested)
    n_sentences = len(positions)

    if not vectors_path.exists():
        return ("needs_embed", name, n_sentences)

    with open(vectors_path, "rb") as vector_file:
        mm = mmap.mmap(vector_file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            spans = split_top_level_vector_spans(mm)
            if len(spans) != n_sentences:
                return ("needs_embed", name, n_sentences)

            view = memoryview(mm)
            try:
                first = bytes(view[spans[0][0]:spans[0][1]])
                dim = first.count(b",") + 1 if first.strip(b"[] ") else 0

                tmp_path = sentences_path.with_suffix(".json.tmp")
                try:
                    with open(tmp_path, "wb", buffering=1 << 20) as out:
                        out.write(b"{")
                        pos = 0
                        first_review = True
                        for review_id, sentences in nested.items():
                            if not first_review:
                                out.write(b",")
                            first_review = False
                            out.write(orjson.dumps(review_id))
                            out.write(b":{")
                            first_sentence = True
                            for sentence_key, sentence in sentences.items():
                                if not first_sentence:
                                    out.write(b",")
                                first_sentence = False
                                out.write(orjson.dumps(sentence_key))
                                out.write(b':{"sentence_text":')
                                out.write(orjson.dumps(sentence["sentence_text"]))
                                out.write(b',"vector":')
                                start, end = spans[pos]
                                out.write(view[start:end])
                                out.write(b"}")
                                pos += 1
                            out.write(b"}")
                        out.write(b"}")
                    tmp_path.replace(sentences_path)
                except BaseException:
                    tmp_path.unlink(missing_ok=True)
                    raise
            finally:
                view.release()
        finally:
            mm.close()

    return ("reused", name, n_sentences, dim)


def embed_texts_in_order(texts, client, executor, batch_size, max_in_flight):
    """Embed texts concurrently, returning vectors in input order."""
    batches = [batch for _, batch in chunked(texts, batch_size)]
    n = len(batches)
    results = [None] * n
    in_flight = {}
    next_submit = 0

    def fill_window():
        nonlocal next_submit
        while next_submit < n and len(in_flight) < max_in_flight:
            future = executor.submit(client.embed_batch, batches[next_submit])
            in_flight[future] = next_submit
            next_submit += 1

    fill_window()
    while in_flight:
        done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
        for future in done:
            results[in_flight.pop(future)] = future.result()
        fill_window()

    vectors = []
    for batch_vectors in results:
        vectors.extend(batch_vectors)
    return vectors


def embed_file_in_main(name, sentences_dir, args, client, thread_executor):
    """Fallback path for files whose vector count didn't match: embed via the cloud
    endpoint and write the augmented nested file in place."""
    sentences_path = sentences_dir / name
    nested = orjson.loads(sentences_path.read_bytes())
    positions = flatten_positions(nested)
    texts = [nested[rid][skey]["sentence_text"] for rid, skey in positions]
    vectors = embed_texts_in_order(texts, client, thread_executor, args.batch_size, args.max_in_flight)
    for (rid, skey), vector in zip(positions, vectors):
        nested[rid][skey]["vector"] = vector
    tmp_path = sentences_path.with_suffix(".json.tmp")
    try:
        tmp_path.write_bytes(orjson.dumps(nested))
        tmp_path.replace(sentences_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return len(positions), (len(vectors[0]) if vectors else 0)


def load_manifest(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"done": [], "reused": [], "embedded": []}


def save_manifest(path, manifest):
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences-dir", default=DEFAULT_SENTENCES_DIR, type=Path)
    parser.add_argument("--vectors-dir", default=DEFAULT_VECTORS_DIR, type=Path)
    parser.add_argument("--workers", default=DEFAULT_WORKERS, type=int,
                        help="Process-pool workers for the reuse (raw splice) path.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--concurrency", default=DEFAULT_CONCURRENCY, type=int)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--max-in-flight", default=None, type=int)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess files already in the manifest.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_in_flight is None:
        args.max_in_flight = args.concurrency
    sentences_dir = (SCRIPT_DIR / args.sentences_dir).resolve()
    vectors_dir = (SCRIPT_DIR / args.vectors_dir).resolve()
    manifest_path = SCRIPT_DIR / MANIFEST_NAME

    files = sorted(sentences_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON files found in {sentences_dir}")

    manifest = load_manifest(manifest_path)
    done = set() if args.overwrite else set(manifest["done"])
    todo = [f.name for f in files if f.name not in done]
    print(f"{len(files)} files, {len(todo)} to do, {len(done)} already done. workers={args.workers}")

    needs_embed = []
    total = len(files)
    completed = len(done)

    # Phase 1: parallel reuse across a process pool.
    tasks = [(name, str(sentences_dir), str(vectors_dir)) for name in todo]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(reuse_worker, task): task[0] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            kind, name, n = result[0], result[1], result[2]
            if kind == "reused":
                manifest["done"].append(name)
                manifest["reused"].append(name)
                save_manifest(manifest_path, manifest)
                completed += 1
                print(f"[{completed}/{total}] {name}: {n} sentences, dim {result[3]} -> reused")
            else:
                needs_embed.append(name)
                print(f"[--/{total}] {name}: {n} sentences -> needs embed (count mismatch)")

    # Phase 2: embed the few mismatched files via the cloud endpoint (if any).
    if needs_embed:
        token = load_token(args.token_file)
        client = EmbeddingClient(args.base_url, token, normalize=args.normalize)
        thread_executor = ThreadPoolExecutor(max_workers=args.concurrency)
        try:
            for name in needs_embed:
                started = time.time()
                n, dim = embed_file_in_main(name, sentences_dir, args, client, thread_executor)
                manifest["done"].append(name)
                manifest["embedded"].append(name)
                save_manifest(manifest_path, manifest)
                completed += 1
                print(f"[{completed}/{total}] {name}: {n} sentences, dim {dim} -> embedded in {time.time()-started:.1f}s")
        finally:
            thread_executor.shutdown(wait=True)

    print(
        f"Done. reused={len(manifest['reused'])} embedded={len(manifest['embedded'])} "
        f"total_done={len(manifest['done'])}/{total}"
    )


if __name__ == "__main__":
    main()
