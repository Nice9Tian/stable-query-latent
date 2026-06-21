import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EMBEDDING_DIR = SCRIPT_DIR / "embedding"
if str(EMBEDDING_DIR) not in sys.path:
    sys.path.insert(0, str(EMBEDDING_DIR))

from embed_pseudo_text_sentences import (  # noqa: E402
    encode_with_sentence_transformers,
    encode_with_transformers,
    load_sentence_splitter,
    normalize_text,
)

DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_INPUT_DIR = "game_review_cleaned_3"
DEFAULT_OUTPUT_DIR = "game_review_cleaned_3_sentence_embeddings"
DEFAULT_SENTENCES_DIR = "game_review_cleaned_3_sentences"
DEFAULT_SENTENCE_MODEL_NAME = "sat-3l-sm"


def resolve_script_relative(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def load_reviews(input_path):
    """Each input file is a JSON list of review strings."""
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{input_path} does not contain a JSON list.")
    return data


def _clear_cuda_cache(device):
    if device and str(device).startswith("cuda"):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def split_reviews_into_sentences(reviews, splitter, chunk_size=2000, device=None):
    """Split every review (element) into sentences, preserving order.

    Reviews are fed to the splitter in fixed-size chunks so GPU memory stays
    bounded on files with very many reviews (the whole-file call OOMs on GPU).
    """
    normalized = [normalize_text(review) for review in reviews]

    sentences = []
    for start in range(0, len(normalized), chunk_size):
        chunk = normalized[start : start + chunk_size]
        for review_sentences in splitter.split(chunk):
            for sentence in review_sentences:
                cleaned = sentence.strip()
                if cleaned:
                    sentences.append(cleaned)
        _clear_cuda_cache(device)
    return sentences


def encode_sentences(sentences, args):
    if args.backend == "sentence-transformers":
        return encode_with_sentence_transformers(sentences, args)
    return encode_with_transformers(sentences, args)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split each review (every element) in the game_review_cleaned_3 JSON files "
            "into sentences and embed each sentence. Saves one big JSON list of sentence "
            "vectors per input file."
        )
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--sentences-dir", default=DEFAULT_SENTENCES_DIR, type=Path)
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Only run the sentence splitter and save the split text as JSON per file (no embedding).",
    )
    parser.add_argument(
        "--split-chunk-size",
        default=2000,
        type=int,
        help="Number of reviews fed to the splitter per chunk (bounds GPU memory).",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--sentence-model-name",
        default=DEFAULT_SENTENCE_MODEL_NAME,
        help="wtpsplit SaT model used for sentence segmentation.",
    )
    parser.add_argument(
        "--sentence-device",
        default=None,
        help="Optional device for wtpsplit sentence segmentation, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-length", default=8192, type=int)
    parser.add_argument(
        "--backend",
        choices=["transformers", "sentence-transformers"],
        default="transformers",
        help="Embedding backend to use.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="L2-normalize embeddings before saving.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-embed files even if an output JSON already exists.",
    )
    return parser.parse_args()


def split_only(input_files, splitter, sentences_dir, overwrite, chunk_size, device):
    """Run only the sentence splitter and save split text as one JSON list per file."""
    sentences_dir.mkdir(parents=True, exist_ok=True)
    print(f"[split-only] Saving sentence JSON files to {sentences_dir}")
    for file_index, input_path in enumerate(input_files, start=1):
        output_path = sentences_dir / input_path.name
        if output_path.exists() and not overwrite:
            print(f"[{file_index}/{len(input_files)}] Skipping {input_path.name} (already split)")
            continue

        reviews = load_reviews(input_path)
        sentences = split_reviews_into_sentences(reviews, splitter, chunk_size, device)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(sentences, file, ensure_ascii=False)

        print(
            f"[{file_index}/{len(input_files)}] {input_path.name}: "
            f"{len(reviews)} reviews -> {len(sentences)} sentences saved"
        )
    print(f"[split-only] Done. Sentence JSON files written to {sentences_dir}")


def main():
    args = parse_args()
    input_dir = resolve_script_relative(args.input_dir).resolve()
    output_dir = resolve_script_relative(args.output_dir).resolve()

    input_files = sorted(input_dir.glob("*.json"))
    if not input_files:
        raise ValueError(f"No JSON files were found in {input_dir}.")

    splitter = load_sentence_splitter(args)

    if args.split_only:
        sentences_dir = resolve_script_relative(args.sentences_dir).resolve()
        split_only(
            input_files,
            splitter,
            sentences_dir,
            args.overwrite,
            args.split_chunk_size,
            args.sentence_device,
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(input_files)} input files in {input_dir}")
    for file_index, input_path in enumerate(input_files, start=1):
        output_path = output_dir / input_path.name
        if output_path.exists() and not args.overwrite:
            print(f"[{file_index}/{len(input_files)}] Skipping {input_path.name} (already embedded)")
            continue

        reviews = load_reviews(input_path)
        sentences = split_reviews_into_sentences(
            reviews, splitter, args.split_chunk_size, args.sentence_device
        )

        if not sentences:
            print(f"[{file_index}/{len(input_files)}] {input_path.name}: no sentences, writing empty list")
            output_path.write_text("[]", encoding="utf-8")
            continue

        print(
            f"[{file_index}/{len(input_files)}] {input_path.name}: "
            f"{len(reviews)} reviews -> {len(sentences)} sentences"
        )
        embeddings = encode_sentences(sentences, args)

        # One big JSON list containing all sentence vectors for this file.
        vectors = embeddings.tolist()
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(vectors, file)

        print(
            f"[{file_index}/{len(input_files)}] Saved {len(vectors)} sentence vectors "
            f"(dim {embeddings.shape[1]}) to {output_path}"
        )

    print(f"Done. Sentence-embedding JSON files written to {output_dir}")


if __name__ == "__main__":
    main()
