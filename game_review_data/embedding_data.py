"""Stage 3 of the game-review embedding pipeline: embed every sentence and attach
its vector, keeping the review_id / sentence_id structure.

Reads the nested sentence files produced by split_data.py and writes one JSON object
per game where each sentence gains a ``vector`` field:

    { "<review_id>": { "sentence_1": {"sentence_text": ..., "vector": [...]}, ... }, ... }

Two backends (user choice):
    --backend local   local Qwen3-Embedding via transformers (GPU/CPU, last-token pool)
    --backend cloud   remote TEI endpoint (token from tokenAPI.txt, concurrent requests)

Vectors are inserted at their exact review_id/sentence_id position, so the
review <-> sentence <-> vector correspondence is preserved by construction.
"""

import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

DEFAULT_INPUT_DIR = "game_review_sentences"
DEFAULT_OUTPUT_DIR = "game_review_embedded"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def flatten_positions(nested):
    return [(rid, skey) for rid, sentences in nested.items() for skey in sentences]


# --------------------------------------------------------------------------- local
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
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**tokens).last_hidden_state
                vectors = self._last_token_pool(hidden, tokens["attention_mask"])
            out.extend(vectors.cpu().float().tolist())
        return out


# --------------------------------------------------------------------------- cloud
class CloudEmbedder:
    """Remote TEI endpoint via the robust client from cloud_embedding.py."""

    def __init__(self, base_url=None, token_file=None, concurrency=256, batch_size=32,
                 max_in_flight=None, normalize=False):
        from cloud_embedding import (
            DEFAULT_TOKEN_FILE,
            EmbeddingClient,
            chunked,
            load_credentials,
        )

        self.chunked = chunked
        creds = load_credentials(token_file or DEFAULT_TOKEN_FILE)
        # URL and token both come from the credentials file by default; ``base_url``
        # only overrides the URL key when explicitly passed.
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


def embed_data(input_dir, output_dir, backend, overwrite=False, **kwargs):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.json"))
    if not input_files:
        raise ValueError(f"No JSON files found in {input_dir}")

    if backend == "local":
        embedder = LocalEmbedder(
            kwargs.get("local_model", DEFAULT_LOCAL_MODEL),
            device=kwargs.get("device"),
            batch_size=kwargs.get("batch_size", 32),
        )
        closer = None
    else:
        embedder = CloudEmbedder(
            base_url=kwargs.get("base_url"),
            token_file=kwargs.get("token_file"),
            concurrency=kwargs.get("concurrency", 256),
            batch_size=kwargs.get("batch_size", 32),
            max_in_flight=kwargs.get("max_in_flight"),
            normalize=kwargs.get("normalize", False),
        )
        closer = embedder.close

    print(f"embed_data: {len(input_files)} files, backend={backend} -> {output_dir}")
    try:
        for file_index, input_path in enumerate(input_files, start=1):
            output_path = output_dir / input_path.name
            if output_path.exists() and not overwrite:
                print(f"[{file_index}/{len(input_files)}] {input_path.name}: skip (exists)")
                continue

            with input_path.open("r", encoding="utf-8") as file:
                nested = json.load(file)
            positions = flatten_positions(nested)
            if not positions:
                output_path.write_text("{}", encoding="utf-8")
                continue

            texts = [nested[rid][skey]["sentence_text"] for rid, skey in positions]
            started = time.time()
            vectors = embedder.embed(texts)
            for (rid, skey), vector in zip(positions, vectors):
                nested[rid][skey]["vector"] = vector

            tmp_path = output_path.with_suffix(".json.tmp")
            try:
                with tmp_path.open("w", encoding="utf-8") as file:
                    json.dump(nested, file, ensure_ascii=False)
                tmp_path.replace(output_path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise

            dim = len(vectors[0]) if vectors else 0
            print(
                f"[{file_index}/{len(input_files)}] {input_path.name}: "
                f"{len(texts)} sentences, dim {dim} in {time.time() - started:.1f}s"
            )
    finally:
        if closer:
            closer()
    print(f"Done. Embedded JSON written to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--concurrency", default=256, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-in-flight", default=None, type=int)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    embed_data(
        args.input_dir,
        args.output_dir,
        args.backend,
        overwrite=args.overwrite,
        local_model=args.local_model,
        device=args.device,
        base_url=args.base_url,
        token_file=args.token_file,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        max_in_flight=args.max_in_flight,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
