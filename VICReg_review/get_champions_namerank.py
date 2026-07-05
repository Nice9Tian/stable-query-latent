"""Champion re-selection with real-text evidence (name-recall rank included).

The original auto_champions selection used review-domain battery metrics only
(top-50 intersection of tag_f1 / identity / variant_drop / selectivity).
This script re-ranks every combo that has BOTH a battery row
(grid_metrics.json) and a real-text row (realtext_grid_metrics.json, from
realtext_grid_eval.py) using rank-sum over SEVEN metrics -- the original four
plus three real-text ones:

  rt_tag_f1     neutral-article micro-F1              (higher better)
  rt_drop       mean anchor->article F1 drop          (lower better)
  namerank      neutral article -> own game MEDIAN retrieval rank (lower)

Writes <out-dir>/champions_namerank.json (does NOT touch champions.json, so
the training/eval pipeline is unaffected) and prints the full ranking table
plus how the ORIGINAL champions moved.

    python VICReg_review/get_champions_namerank.py [--top-k 50]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METRICS = {
    # key -> (extractor, higher_is_better)
    "tag_f1": (lambda b, r: b.get("tag_f1"), True),
    "identity": (lambda b, r: b.get("identity_hit_at_1"), True),
    "variant_drop": (lambda b, r: b.get("variant_drop_mean"), False),
    "selectivity": (lambda b, r: b.get("selectivity"), True),
    "rt_tag_f1": (lambda b, r: (r.get("tag") or {}).get("neutral", {}).get("micro_f1"), True),
    "rt_drop": (lambda b, r: r.get("mean_drop"), False),
    "namerank": (lambda b, r: (r.get("retrieval") or {}).get("neutral", {}).get("median_rank"), False),
    # contrastive fine-tuned name recall (77 held-out games, mean-anchor gallery)
    "namerank_con": (lambda b, r: ((r.get("contrastive") or {}).get("con_linear") or {}).get("median_rank"), False),
    "hit1_con": (lambda b, r: ((r.get("contrastive") or {}).get("con_linear") or {}).get("hit_at_1"), True),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(ROOT / "VICReg_review/heads/cloud_full_sweep_a100"))
    p.add_argument("--top-k", type=int, default=50,
                   help="per-metric top-K for the intersection set (0 = skip)")
    p.add_argument("--n-champions", type=int, default=8,
                   help="how many top rank-sum combos to store as champions")
    p.add_argument("--by", default="rank_sum",
                   help="ordering: 'rank_sum' or a single metric key "
                        f"(one of {list(METRICS)}), e.g. --by namerank_con to "
                        "pick purely by fine-tuned name recall")
    args = p.parse_args()

    out_root = Path(args.out_dir)
    battery = {r["combo_id"]: r for r in json.loads(
        (out_root / "grid_metrics.json").read_text(encoding="utf-8"))["rows"]}
    realtext = {r["combo_id"]: r for r in json.loads(
        (out_root / "realtext_grid_metrics.json").read_text(encoding="utf-8"))["rows"]}
    old = []
    old_path = out_root / "champions.json"
    if old_path.exists():
        old = [r["combo_id"] for r in
               json.loads(old_path.read_text(encoding="utf-8"))["champions"]]

    pool = sorted(set(battery) & set(realtext))
    print(f"pool: {len(pool)} combos with battery + real-text rows "
          f"(battery {len(battery)}, realtext {len(realtext)})")

    values = {}
    for cid in pool:
        b, r = battery[cid], realtext[cid]
        vals = {}
        for key, (fn, _hib) in METRICS.items():
            v = fn(b, r)
            vals[key] = float(v) if isinstance(v, (int, float)) else None
        values[cid] = vals

    ranks: dict[str, dict[str, int]] = {cid: {} for cid in pool}
    top_sets = {}
    for key, (_fn, hib) in METRICS.items():
        scored = [(cid, values[cid][key]) for cid in pool if values[cid][key] is not None]
        scored.sort(key=lambda t: -t[1] if hib else t[1])
        for i, (cid, _v) in enumerate(scored, 1):
            ranks[cid][key] = i
        top_sets[key] = [cid for cid, _v in scored[: args.top_k]] if args.top_k else []

    rows = []
    for cid in pool:
        rk = ranks[cid]
        if len(rk) != len(METRICS):
            continue
        rows.append({"combo_id": cid, **values[cid], "ranks": rk,
                     "rank_sum": sum(rk.values())})
    if args.by == "rank_sum":
        rows.sort(key=lambda r: r["rank_sum"])
    else:
        if args.by not in METRICS:
            raise SystemExit(f"--by must be rank_sum or one of {list(METRICS)}")
        rows.sort(key=lambda r: r["ranks"][args.by])
    inter = set(pool)
    if args.top_k:
        for key in METRICS:
            inter &= set(top_sets[key])

    champions = rows[: args.n_champions]
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selection": f"get_champions_namerank: ordered by {args.by} over battery "
                     "+ real-text metrics (incl. zero-shot and contrastive "
                     "name-recall ranks)",
        "ordered_by": args.by,
        "metrics": list(METRICS),
        "top_k": args.top_k,
        "intersection_all_topk": sorted(inter),
        "old_champions": old,
        "champions": champions,
        "full_ranking": rows,
    }
    from VICReg_review.text_variant_eval import atomic_json_write
    atomic_json_write(payload, out_root / "champions_namerank.json")

    fmt = lambda x, w, d: (f"{x:{w}.{d}f}" if isinstance(x, (int, float)) else " " * (w - 1) + "-")
    print(f"\nORDERED BY {args.by}  over {list(METRICS)}")
    head = (f"{'#':>3} {'combo_id':40} {'sum':>5} {'tagF1':>6} {'rtF1':>6} "
            f"{'rtDrop':>7} {'nameRk':>7} {'nameRkC':>8} {'hit1C':>6} {'old?':>5}")
    print(head)
    print("-" * len(head))
    for i, r in enumerate(rows[:20], 1):
        print(f"{i:3d} {r['combo_id']:40} {r['rank_sum']:5d} "
              f"{fmt(r['tag_f1'], 6, 3)} {fmt(r['rt_tag_f1'], 6, 3)} "
              f"{fmt(r['rt_drop'], 7, 3)} {fmt(r['namerank'], 7, 0)} "
              f"{fmt(r['namerank_con'], 8, 0)} {fmt(r['hit1_con'], 6, 3)} "
              f"{'YES' if r['combo_id'] in old else '':>5}")
    print(f"\nintersection of all {len(METRICS)} top-{args.top_k} sets: "
          f"{len(inter)} combos")
    moved = [(cid, next((i + 1 for i, r in enumerate(rows) if r["combo_id"] == cid), None))
             for cid in old]
    print("original champions under the new ranking: "
          + ", ".join(f"{cid}=#{pos}" for cid, pos in moved))
    print(f"\nwritten -> {out_root / 'champions_namerank.json'} "
          "(champions.json untouched)")


if __name__ == "__main__":
    main()
