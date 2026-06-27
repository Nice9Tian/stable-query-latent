"""Embed the unified game-review text H5 into a unified embedding H5.

Input:
    text_h5.h5

Output:
    embedding_h5.h5

The output keeps all text/review/game metadata from ``text_h5.h5`` and adds a
single streamable ``vectors`` dataset in the layout consumed directly by
``VICReg_review/train_vicreg_review_h5.py``.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

# cloud_embedding.py lives at the project root, one level up from this file.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent

try:
    from game_review_data.h5_corpus import DEFAULT_EMBEDDING_H5, DEFAULT_TEXT_H5, embed_text_h5
except ImportError:  # pragma: no cover - direct script execution
    from h5_corpus import DEFAULT_EMBEDDING_H5, DEFAULT_TEXT_H5, embed_text_h5

DEFAULT_INPUT_H5 = DEFAULT_TEXT_H5
DEFAULT_OUTPUT_H5 = DEFAULT_EMBEDDING_H5
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class LocalEmbedder:
    """Local Qwen3-Embedding via transformers with last-token pooling."""

    def __init__(self, model_name, device=None, max_length=2048, batch_size=32):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size

    @staticmethod
    def _last_token_pool(last_hidden, attention_mask):
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden[:, -1]
        lengths = attention_mask.sum(dim=1) - 1
        import torch

        return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), lengths]

    def embed(self, texts):
        torch = self.torch
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**tokens).last_hidden_state
                vectors = self._last_token_pool(hidden, tokens["attention_mask"])
            out.extend(vectors.cpu().float().tolist())
        return out


class CloudEmbedder:
    """Remote TEI endpoint via the robust client from cloud_embedding.py."""

    def __init__(
        self,
        base_url=None,
        token_file=None,
        concurrency=256,
        batch_size=32,
        max_in_flight=None,
        normalize=False,
    ):
        from cloud_embedding import DEFAULT_TOKEN_FILE, EmbeddingClient, chunked, load_credentials

        self.chunked = chunked
        creds = load_credentials(token_file or DEFAULT_TOKEN_FILE)
        self.client = EmbeddingClient(base_url or creds["url"], creds["token"], normalize=normalize)
        self.batch_size = batch_size
        self.max_in_flight = max_in_flight or concurrency
        self.executor = ThreadPoolExecutor(max_workers=concurrency)

    def embed(self, texts):
        batches = [batch for _, batch in self.chunked(texts, self.batch_size)]
        n = len(batches)
        results = [None] * n
        in_flight = {}
        next_submit = 0

        def fill():
            nonlocal next_submit
            while next_submit < n and len(in_flight) < self.max_in_flight:
                future = self.executor.submit(self.client.embed_batch, batches[next_submit])
                in_flight[future] = next_submit
                next_submit += 1

        fill()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                results[in_flight.pop(future)] = future.result()
            fill()

        vectors = []
        for batch_vectors in results:
            vectors.extend(batch_vectors)
        return vectors

    def close(self):
        self.executor.shutdown(wait=True)


def make_embedder(backend: str, **kwargs):
    if backend == "local":
        return LocalEmbedder(
            kwargs.get("local_model", DEFAULT_LOCAL_MODEL),
            device=kwargs.get("device"),
            batch_size=kwargs.get("batch_size", 32),
        ), kwargs.get("local_model", DEFAULT_LOCAL_MODEL)

    return CloudEmbedder(
        base_url=kwargs.get("base_url"),
        token_file=kwargs.get("token_file"),
        concurrency=kwargs.get("concurrency", 256),
        batch_size=kwargs.get("batch_size", 32),
        max_in_flight=kwargs.get("max_in_flight"),
        normalize=kwargs.get("normalize", False),
    ), kwargs.get("local_model", DEFAULT_LOCAL_MODEL)


def embed_data(input_h5, output_h5, backend, overwrite=False, **kwargs):
    """Compatibility wrapper used by build.py."""
    embedder, model_name = make_embedder(backend, **kwargs)
    closer = getattr(embedder, "close", None)
    try:
        return embed_text_h5(
            input_h5=input_h5,
            output_h5=output_h5,
            embedder=embedder,
            backend=backend,
            embedding_model=model_name,
            overwrite=overwrite,
            read_batch_size=kwargs.get("read_batch_size", 4096),
            dtype=kwargs.get("dtype", "float16"),
            chunk_rows=kwargs.get("chunk_rows", 2048),
            compression=kwargs.get("compression", "none"),
            gzip_level=kwargs.get("gzip_level", 1),
        )
    finally:
        if closer:
            closer()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, default=DEFAULT_INPUT_H5)
    parser.add_argument("--output-h5", type=Path, default=DEFAULT_OUTPUT_H5)
    parser.add_argument("--backend", choices=["local", "cloud"], default="local")
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--concurrency", default=256, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-in-flight", default=None, type=int)
    parser.add_argument("--read-batch-size", default=4096, type=int)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--chunk-rows", default=2048, type=int)
    parser.add_argument("--compression", choices=["none", "gzip", "lzf"], default="none")
    parser.add_argument("--gzip-level", default=1, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    embed_data(
        input_h5=args.input_h5,
        output_h5=args.output_h5,
        backend=args.backend,
        overwrite=args.overwrite,
        local_model=args.local_model,
        device=args.device,
        base_url=args.base_url,
        token_file=args.token_file,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        max_in_flight=args.max_in_flight,
        read_batch_size=args.read_batch_size,
        normalize=args.normalize,
        dtype=args.dtype,
        chunk_rows=args.chunk_rows,
        compression=args.compression,
        gzip_level=args.gzip_level,
    )


if __name__ == "__main__":
    main()
