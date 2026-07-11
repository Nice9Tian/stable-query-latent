# -*- coding: utf-8 -*-
"""Aggregate every finished arm into one comparison table (markdown).

    python contrast_experiment/report.py [--out RESULTS/summary.md]

Reads ft4var_<arm>_best*.json (post-hoc-picked, 10-seed) from the results
dir; ranks by mean-of-4 hit@1. Also emits the CV table (mean +- std across
folds) when cv_* results are present.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from steam_reviews_framework.eval import VORDER
from dataset_builder.paths import RESULTS


def agg(runs):
    row = {}
    for var in VORDER:
        row[var] = (float(np.mean([r[var]["h1"] for r in runs])),
                    float(np.std([r[var]["h1"] for r in runs])),
                    float(np.mean([r[var]["tag"] for r in runs])))
    row["m4"] = float(np.mean([np.mean([r[v]["h1"] for r in runs])
                               for v in VORDER]))
    row["vsel"] = float(np.mean([r["vscore"] for r in runs]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_p = args.out or (args.results / "summary.md")

    fixed, cv = {}, defaultdict(dict)
    for p in sorted(args.results.glob("ft4var_*_best*.json")):
        d = json.loads(p.read_text())
        m = re.match(r"ft4var_(.+)_best(.*)$", p.stem)
        arm, hsuf = m.group(1), m.group(2)
        key = arm + hsuf
        row = agg(d["per_seed"])
        row["best_ep"] = d.get("best_ep")
        cvm = re.match(r"cv_(.+)_fold(\d)$", arm)
        if cvm:
            cv[cvm.group(1) + hsuf][int(cvm.group(2))] = row
        else:
            fixed[key] = row

    lines = ["# Experiment summary", "",
             "## Fixed split (test = 204 wiki games, 10 seeds, post-hoc pick)",
             "", "| arm | best ep | " +
             " | ".join(f"{v} h1 | {v} tag" for v in VORDER) +
             " | mean-of-4 | vsel |",
             "|---|---|" + "---|" * (2 * len(VORDER) + 2)]
    for k, r in sorted(fixed.items(), key=lambda kv: -kv[1]["m4"]):
        cells = " | ".join(f"{r[v][0]:.3f}±{r[v][1]:.3f} | {r[v][2]:.3f}"
                           for v in VORDER)
        lines.append(f"| {k} | {r['best_ep']} | {cells} | "
                     f"{r['m4']:.3f} | {r['vsel']:.3f} |")
    if cv:
        lines += ["", "## 5-fold CV (mean ± std across folds)", "",
                  "| recipe | folds | " +
                  " | ".join(f"{v} h1" for v in VORDER) + " | mean-of-4 |",
                  "|---|---|" + "---|" * (len(VORDER) + 1)]
        for k, folds in sorted(cv.items()):
            vs = {v: [folds[f][v][0] for f in folds] for v in VORDER}
            m4 = [folds[f]["m4"] for f in folds]
            cells = " | ".join(f"{np.mean(vs[v]):.3f}±{np.std(vs[v]):.3f}"
                               for v in VORDER)
            lines.append(f"| {k} | {len(folds)} | {cells} | "
                         f"{np.mean(m4):.3f}±{np.std(m4):.3f} |")
    out_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten -> {out_p}")


if __name__ == "__main__":
    main()
