# -*- coding: utf-8 -*-
"""5-fold CV over the 814-game wiki universe (fold k = test, k+1 = val,
other three = train; permutation seed 20260711).

    python larice_experiment/run_cv.py                      # 6 recipes x 5 folds
    python larice_experiment/run_cv.py --recipes ce --folds 0 1
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from larice_framework.data import load_bundle
from larice_framework.train import CHAMPION, run_arm
from larice_experiment.contrast_models.roster import ARMS, CV_RECIPES
from larice_experiment.run_all import train_fn_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", nargs="*", default=None)
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    recipes = args.recipes or CV_RECIPES
    log = lambda *a: print(*a, flush=True)
    for fold in args.folds:
        B = load_bundle(torch.device(args.device), cv_fold=fold)
        log(f"fold {fold}: test {len(B.test_g)} val {len(B.val_g)} "
            f"train pool {len(B.train_pool_games)}")
        for r in recipes:
            base = CHAMPION if r == "champion_cegate2" else ARMS[r]
            spec = copy.copy(base)
            spec.name = f"cv_{r}_fold{fold}"
            spec.epochs = args.epochs
            spec.ckpt_every = args.ckpt_every
            run_arm(B, spec, train_fn=train_fn_for(spec), log_cb=log)
    log("run_cv done")


if __name__ == "__main__":
    main()
