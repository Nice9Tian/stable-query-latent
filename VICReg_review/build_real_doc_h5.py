"""Split + embed the verified wiki description texts into real_doc.h5.

Takes the games listed in wiki_descriptions/top100_verified.csv, strips the
collector header from each txt, splits sentences with the same splitter used
for the disturbance/variant texts (disturbtion_embed.split_text), embeds them
with the same Qwen3-Embedding backend as the corpus (local | cloud), and packs
everything into one H5 so eval code can treat the wiki docs as an independent
real-text query set.

Layout (schema real_doc_h5.v1, flat concat + offsets like text_h5):

    vectors                (sentences, dim)  float32
    texts                  (sentences,)      utf-8
    game_sentence_offsets  (games + 1,)      int64
    appids / game_names / source_titles / source_urls  (games,) utf-8

    python VICReg_review/build_real_doc_h5.py                    # local GPU
    python VICReg_review/build_real_doc_h5.py --backend cloud
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for extra in (str(ROOT), str(ROOT / "game_review_data")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from VICReg_review.disturbtion_embed import split_text  # noqa: E402
from game_review_data.embedding_data import (  # noqa: E402
    DEFAULT_LOCAL_MODEL,
    CloudEmbedder,
    LocalEmbedder,
)

DEFAULT_DESC_DIR = SCRIPT_DIR / "wiki_descriptions"
DEFAULT_CSV = DEFAULT_DESC_DIR / "top100_verified.csv"
DEFAULT_OUT = DEFAULT_DESC_DIR / "real_doc.h5"
HEADER_SEPARATOR = "-" * 8


def read_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    _, sep, body = text.partition("\n" + HEADER_SEPARATOR + "\n")
    return (body if sep else text).strip()


def build_embedder(args):
    if args.backend == "cloud":
        return CloudEmbedder(batch_size=args.batch_size), "cloud"
    return LocalEmbedder(args.local_model, device=args.device, batch_size=args.batch_size), args.local_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--desc-dir", type=Path, default=DEFAULT_DESC_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--backend", choices=("local", "cloud"), default="local")
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-doc-sentences", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.overwrite:
        print(f"real_doc h5 already exists (use --overwrite to rebuild): {args.out}", flush=True)
        return

    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    games = []
    for row in rows:
        path = args.desc_dir / row["filename"]
        body = read_body(path)
        sentences = split_text(body, args.max_doc_sentences)
        if not sentences:
            raise ValueError(f"no sentences extracted from {path}")
        games.append((row, sentences))
    total = sum(len(s) for _, s in games)
    print(f"games: {len(games)}, sentences to embed: {total}", flush=True)

    embedder, model_name = build_embedder(args)
    vectors = []
    offsets = [0]
    started = time.time()
    for index, (row, sentences) in enumerate(games, start=1):
        embedded = np.asarray(embedder.embed(sentences), dtype=np.float32)
        if embedded.shape[0] != len(sentences):
            raise RuntimeError(
                f"{row['game_name']}: embedded {embedded.shape[0]} of {len(sentences)} sentences"
            )
        vectors.append(embedded)
        offsets.append(offsets[-1] + embedded.shape[0])
        print(
            f"[{index}/{len(games)}] {row['game_name']}: {len(sentences)} sentences "
            f"({time.time() - started:.0f}s elapsed)",
            flush=True,
        )

    all_vectors = np.concatenate(vectors, axis=0)
    all_texts = [s for _, sentences in games for s in sentences]
    str_dtype = h5py.string_dtype(encoding="utf-8")

    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    try:
        with h5py.File(tmp, "w") as h5:
            h5.attrs["schema"] = "real_doc_h5.v1"
            h5.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            h5.attrs["embedding_model"] = model_name
            h5.attrs["backend"] = args.backend
            h5.attrs["input_dim"] = int(all_vectors.shape[1])
            h5.attrs["source_csv"] = str(args.csv)
            h5.attrs["games"] = len(games)
            h5.attrs["sentences"] = int(all_vectors.shape[0])
            h5.create_dataset("vectors", data=all_vectors, dtype=np.float32)
            h5.create_dataset("texts", data=all_texts, dtype=str_dtype)
            h5.create_dataset("game_sentence_offsets", data=np.asarray(offsets, dtype=np.int64))
            for column, name in (
                ("appid", "appids"),
                ("game_name", "game_names"),
                ("source_title", "source_titles"),
                ("source_url", "source_urls"),
            ):
                h5.create_dataset(name, data=[r.get(column, "") for r, _ in games], dtype=str_dtype)
        tmp.replace(args.out)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    print(
        f"wrote {args.out} ({len(games)} games, {all_vectors.shape[0]} sentences, "
        f"dim={all_vectors.shape[1]})",
        flush=True,
    )


if __name__ == "__main__":
    main()
