"""Retrieval pilot: can a generated article recall its OWN game from the gallery?

For each combo: encode the 258x4 generated articles through the combo's frozen
encoder (sentence embeddings come from the pilot cache built by
pilot_realtext_tag.py), then rank every article's cosine against ALL gallery
games' anchor features (eval_features npz, 2020 games). Reports hit@1/5/10 and
median rank per variant, writes <combo>/real_text_retrieval_pilot.json.

    python VICReg_review/pilot_realtext_retrieval.py --out-dir <local out_dir> \
        --combos "id1,id2,..."           (default: champions.json + nothing)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANTS = ("neutral", "positive", "negative", "noname")
OUT_NAME = "real_text_retrieval_pilot.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", default=str(ROOT / "game_review_data/embedding_h5.h5"))
    p.add_argument("--out-dir", default=str(ROOT / "VICReg_review/heads/cloud_full_sweep_a100"))
    p.add_argument("--cache", default=None,
                   help="pilot text embedding cache; default <out-dir>/text_variant_pilot_cache.npz")
    p.add_argument("--combos", default="", help="extra combo ids beyond champions.json")
    args = p.parse_args()

    import h5py
    import numpy as np
    import torch

    from VICReg_review import eval_battery_worker as worker
    from VICReg_review import run_data_view_sweep as sweep
    from VICReg_review import text_variant_eval
    from VICReg_review.identity_diagnostic import l2_normalize

    out_root = Path(args.out_dir)
    cache_path = Path(args.cache or out_root / "text_variant_pilot_cache.npz")
    if not cache_path.exists():
        raise SystemExit(f"pilot cache not found: {cache_path} -- run pilot_realtext_tag.py first")

    champions = [row["combo_id"] for row in json.loads(
        (out_root / "champions.json").read_text(encoding="utf-8"))["champions"]]
    champions += [c.strip() for c in args.combos.split(",")
                  if c.strip() and c.strip() not in champions]

    ev = worker.build_eval_args(argparse.Namespace(h5=args.h5, out_dir=str(out_root)))
    ev.out_dir = out_root
    ev.text_variant_cache = cache_path

    from VICReg_review import disturbtion_embed
    cache = disturbtion_embed.load_npz_payload(cache_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(ev.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])

    rows = []
    for cid in champions:
        cdir = out_root / cid
        ready, ckpt = worker.combo_ready(cdir)
        if not ready:
            print(f"  {cid}: NOT ready -- skipped")
            continue
        feats, names = sweep.build_vicreg_feature_cache(ev, ckpt, cdir)
        encoder, _, _, _ = sweep.load_frozen_encoder(ckpt, input_dim, device)
        variant_feats = text_variant_eval.encode_variant_features(ev, encoder, cache, device)

        # the eval feature cache is VIEW-level (each game contributes several
        # rows); retrieval must be game-level: per game take the MAX similarity
        # over its view rows, then rank among unique games
        gallery = l2_normalize(np.asarray(feats, dtype=np.float32))
        game_rows: dict[str, list[int]] = {}
        for i, n in enumerate(names):
            game_rows.setdefault(n, []).append(i)
        game_names = list(game_rows)
        row_groups = [np.asarray(game_rows[n]) for n in game_names]
        game_index = {n: i for i, n in enumerate(game_names)}

        per_variant = {v: [] for v in VARIANTS}          # ranks (1-based)
        for (name, variant), q in variant_feats.items():
            gi = game_index.get(name)
            if gi is None or variant not in per_variant:
                continue
            sims = gallery @ l2_normalize(np.asarray(q, dtype=np.float32).reshape(1, -1))[0]
            game_sims = np.asarray([sims[idx].max() for idx in row_groups])
            rank = int((game_sims > game_sims[gi]).sum()) + 1
            per_variant[variant].append(rank)

        report = {"combo_id": cid, "gallery_size": len(game_names), "variants": {}}
        for v, ranks in per_variant.items():
            if not ranks:
                continue
            r = np.asarray(ranks)
            report["variants"][v] = {
                "n_queries": int(r.size),
                "hit_at_1": float((r == 1).mean()),
                "hit_at_5": float((r <= 5).mean()),
                "hit_at_10": float((r <= 10).mean()),
                "median_rank": float(np.median(r)),
                "mean_rank": float(r.mean()),
            }
        text_variant_eval.atomic_json_write(report, cdir / OUT_NAME)
        rows.append(report)
        del feats, encoder, variant_feats
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {cid}: done")

    text_variant_eval.atomic_json_write(
        {"rows": rows}, out_root / "real_text_retrieval_summary.json")
    n_q = next((r["variants"].get("neutral", {}).get("n_queries") for r in rows), "?")
    n_gal = next((r.get("gallery_size") for r in rows), "?")
    print(f"\nARTICLE -> OWN-GAME RETRIEVAL  (gallery={n_gal} games, {n_q} queries/variant)")
    head = (f"{'combo_id':40} " + " ".join(
        f"{v[:3]+'@1':>7} {v[:3]+'med':>7}" for v in VARIANTS))
    print(head)
    print("-" * len(head))
    for r in rows:
        cells = []
        for v in VARIANTS:
            m = r["variants"].get(v, {})
            cells.append(f"{m.get('hit_at_1', float('nan')):7.3f} "
                         f"{m.get('median_rank', float('nan')):7.0f}")
        print(f"{r['combo_id']:40} " + " ".join(cells))


if __name__ == "__main__":
    main()
