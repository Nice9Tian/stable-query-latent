"""Adversary-weight sweep: does a stronger sentiment adversary beat pure VICReg?

For each weight W: train 39 epochs (seed 42, same config as the baselines, only
--adversary-weight differs; GRL schedule warmup 5 / ramp 10 / lambda 1 matches the
original adversary run), extract probe features, run the selectivity probe. Writes
one summary row per run to heads/sweep_adv/summary.tsv.

Baselines already on disk for the final table:
  pure VICReg (weight 0):  selectivity gap +0.335
  weight 1 (latest_3):     selectivity gap +0.162
"""

import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PY = sys.executable
OUT = SCRIPT_DIR / "heads" / "sweep_adv"
OUT.mkdir(parents=True, exist_ok=True)
WEIGHTS = [10, 20, 50]


def run(cmd):
    print("RUN:", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    rows = []
    for w in WEIGHTS:
        tag = f"adv{w}"
        best = OUT / f"vicreg_{tag}_best.pt"
        latest = OUT / f"vicreg_{tag}_latest.pt"
        print(f"\n===== adversary-weight {w} =====", flush=True)
        run([PY, SCRIPT_DIR / "train_vicreg_review_h5.py",
             "--device", "cuda", "--amp", "--epochs", 39, "--batch-size", 16,
             "--steps-per-epoch", 19, "--sample-fraction", 0.6, "--seed", 42,
             "--adversary-weight", w, "--grl-lambda", 1.0,
             "--grl-warmup-epochs", 5, "--grl-ramp-epochs", 10,
             "--checkpoint-out", latest, "--best-checkpoint-out", best,
             "--history-tsv", OUT / f"history_{tag}.tsv",
             "--manifest-json", OUT / f"manifest_{tag}.json"])

        run([PY, SCRIPT_DIR / "train_tag_probe.py",
             "--checkpoint", best, "--pool", "flatten", "--device", "cuda", "--amp",
             "--report-json", OUT / f"tag_probe_{tag}.json"])

        vic_cache = SCRIPT_DIR / "tags" / f"probe_feat_vicreg_{tag}_best_fv4_sf0.6.npz"
        report = OUT / f"selectivity_{tag}.json"
        run([PY, SCRIPT_DIR / "probe_selectivity.py", "--device", "cuda",
             "--vic-cache", vic_cache, "--report-json", report])

        r = json.loads(report.read_text(encoding="utf-8"))
        manifest = json.loads((OUT / f"manifest_{tag}.json").read_text(encoding="utf-8"))
        rows.append({
            "adversary_weight": w,
            "content_ret": r["retention"]["content"],
            "subjective_ret": r["retention"]["subjective"],
            "sentiment_ret": r["retention"]["sentiment"],
            "selectivity_gap": r["selectivity_gap"],
            "content_f1": r["results"]["vicreg"]["content"]["micro_f1"],
            "sentiment_r2": r["results"]["vicreg"]["sentiment"]["r2"],
            "final_loss": manifest["metrics"].get("loss"),
            "adv_entropy": manifest["metrics"].get("sentiment_entropy"),
        })
        # Persist after every run so partial progress survives.
        cols = list(rows[0].keys())
        lines = ["\t".join(cols)]
        for row in rows:
            lines.append("\t".join(f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols))
        (OUT / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"weight {w}: gap={r['selectivity_gap']:+.3f} content_ret={r['retention']['content']:.3f} "
              f"sentiment_ret={r['retention']['sentiment']:.3f}", flush=True)

    print("\nSWEEP DONE", time.strftime("%H:%M:%S"), flush=True)
    print((OUT / "summary.tsv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
