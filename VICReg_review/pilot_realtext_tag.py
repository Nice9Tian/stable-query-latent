"""Pilot: real-text tag regression for the champions on the GENERATED variants.

Runs ONLY the text-variant part of the battery for every combo listed in
champions.json, pointing --text-variant-dir at the LLM-generated pilot set
(VICReg_review/text_variants_generated, checked into the repo) and using a
SEPARATE embedding cache + output file, so nothing from the original battery
(eval_report.json, the 2-game legacy cache) is touched.

Cheap: anchor features come from each champion's cached
eval_features_full_fv4.npz, so the only GPU work is embedding the pilot texts
(Qwen) and one encoder forward per text. Any single-GPU pod on the volume:

    python VICReg_review/pilot_realtext_tag.py

Writes <combo>/real_text_pilot.json per champion and prints a comparison
table (variant micro-F1 vs anchor-subset micro-F1 and the drop).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = ("neutral", "positive", "negative", "noname")
PILOT_JSON = "real_text_pilot.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", default=str(ROOT / "game_review_data/embedding_h5.h5"))
    p.add_argument("--out-dir", default=str(ROOT / "VICReg_review/heads/cloud_full_sweep_a100"))
    p.add_argument("--variant-dir", default=str(ROOT / "VICReg_review/text_variants_generated"))
    p.add_argument("--champions-json", default=None,
                   help="default: <out-dir>/champions.json")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="re-embed the pilot texts even if the pilot cache exists")
    args = p.parse_args()

    import h5py
    import torch

    from VICReg_review import eval_battery_worker as worker
    from VICReg_review import run_data_view_sweep as sweep
    from VICReg_review import text_variant_eval

    out_root = Path(args.out_dir)
    variant_dir = Path(args.variant_dir)
    if not variant_dir.exists():
        raise SystemExit(f"variant dir not found: {variant_dir} -- git pull first?")
    n_txt = len(list(variant_dir.glob("*/*.txt")))
    champs_path = Path(args.champions_json or out_root / "champions.json")
    champions = [row["combo_id"] for row in
                 json.loads(champs_path.read_text(encoding="utf-8"))["champions"]]
    print(f"{len(champions)} champions; {n_txt} pilot texts under {variant_dir}")

    ev = worker.build_eval_args(argparse.Namespace(h5=args.h5, out_dir=str(out_root)))
    ev.out_dir = out_root
    # pilot-only inputs/outputs -- the original battery artifacts stay untouched
    ev.text_variant_dir = variant_dir
    ev.text_variant_cache = out_root / "text_variant_pilot_cache.npz"
    ev.rebuild_text_variant_cache = bool(args.rebuild_cache)

    device = torch.device(ev.device if getattr(ev, "device", None)
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    with h5py.File(ev.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])

    rows = []
    for cid in champions:
        cdir = out_root / cid
        ready, ckpt = worker.combo_ready(cdir)
        if not ready:
            print(f"  {cid}: NOT ready (no done/checkpoint) -- skipped")
            continue
        feats, names = sweep.build_vicreg_feature_cache(ev, ckpt, cdir)
        encoder, _, _, _ = sweep.load_frozen_encoder(ckpt, input_dim, device)
        result = text_variant_eval.evaluate(ev, encoder, feats, names, cdir)
        text_variant_eval.atomic_json_write(result, cdir / PILOT_JSON)
        rt = result.get("real_text_tag") or {}
        anchor = (result.get("tag_generalization") or {}).get("anchor_test") or {}
        row = {"combo_id": cid, "anchor_test_f1": anchor.get("micro_f1")}
        drops = []
        for v in VARIANTS:
            d = rt.get(v) or {}
            row[v] = (d.get("variant") or {}).get("micro_f1")
            row[f"{v}_n"] = (d.get("variant") or {}).get("n_games")
            if d.get("drop_micro_f1") is not None:
                drops.append(d["drop_micro_f1"])
        row["anchor_subset_f1"] = ((rt.get("neutral") or {}).get("anchor_subset") or {}).get("micro_f1")
        row["mean_drop"] = sum(drops) / len(drops) if drops else None
        rows.append(row)
        del feats, encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {cid}: done -> {cdir / PILOT_JSON}")

    fmt = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "
    n_games = next((r.get("neutral_n") for r in rows if r.get("neutral_n")), "?")
    print(f"\nREAL-TEXT TAG PILOT  (LLM-generated variants, n_games={n_games})")
    header = f"{'combo_id':44} {'anchor':>7} {'a_sub':>7} " + \
        " ".join(f"{v:>8}" for v in VARIANTS) + f" {'drop':>7}"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: -(r.get("neutral") or 0)):
        print(f"{r['combo_id']:44} {fmt(r['anchor_test_f1']):>7} "
              f"{fmt(r['anchor_subset_f1']):>7} "
              + " ".join(f"{fmt(r[v]):>8}" for v in VARIANTS)
              + f" {fmt(r['mean_drop']):>7}")


if __name__ == "__main__":
    main()
