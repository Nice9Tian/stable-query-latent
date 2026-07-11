# -*- coding: utf-8 -*-
"""Path 1 entry point: reproduce the CHAMPION (cegate2 = CE-gated I-CE,
I x2, frozen tau 0.02, wiki_clean > sp_raw doc views, I-CE head).

    python steam_reviews_framework/train_champion.py [--epochs 1000] [--cv-fold K]

Prerequisite: data assets built (see dataset_builder/rebuild_data.py) or
linked via LARICE_* environment variables (dataset_builder/paths.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from steam_reviews_framework.data import load_bundle
from steam_reviews_framework.train import CHAMPION, run_arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--cv-fold", type=int, default=None,
                    help="0..4: run on a CV fold instead of the fixed split")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    spec = CHAMPION
    spec.epochs = args.epochs
    spec.ckpt_every = args.ckpt_every
    if args.cv_fold is not None:
        spec.name = f"{spec.name}_fold{args.cv_fold}"
    B = load_bundle(torch.device(args.device), cv_fold=args.cv_fold)
    print(f"bundle: {B.NG} games | test {len(B.test_g)} val {len(B.val_g)} "
          f"| train pool {len(B.train_pool_games)}", flush=True)
    run_arm(B, spec, log_cb=lambda *a: print(*a, flush=True))


if __name__ == "__main__":
    main()
