# -*- coding: utf-8 -*-
"""ONE-CLICK full contrast suite (the OTHER run.py trains only the champion —
steam_reviews_framework/run.py; this one trains everything else).

    python contrast_experiment/run.py                    # 18 arms, fixed split
    python contrast_experiment/run.py --arms ce byol     # a subset
    python contrast_experiment/run.py --cv               # + 6 recipes x 5 folds
    python contrast_experiment/run.py --cv --folds 0 1   # some folds only

Data preparation (bundled corpora -> reviews h5 -> assets) is shared with
the champion entry and runs automatically. Everything is resume-safe:
finished towers/heads/folds are skipped on relaunch. Ends by writing the
comparison table (report.py).

API SETTINGS (embedding endpoint, LLM gateway, h5 download URLs): fill in
the block at the top of steam_reviews_framework/run.py — it is shared by
this entry through ensure_data(). Credential files under dataset_builder/
work too; in-code values win, mismatches are printed.
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from steam_reviews_framework.run import ensure_data
from steam_reviews_framework.data import load_bundle
from steam_reviews_framework.train import CHAMPION, run_arm, train_tower
from contrast_experiment.contrast_models.roster import ARMS, CV_RECIPES
from contrast_experiment.contrast_models.arcface import arc_loss_hook
from contrast_experiment.contrast_models.byol import train_byol


def train_fn_for(spec):
    if spec.tower == "byol":
        return train_byol
    if spec.tower == "arc":
        return lambda B, s, log_cb=print: train_tower(
            B, s, loss_hook=arc_loss_hook, log_cb=log_cb)
    return None                      # default champion-family objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None,
                    help=f"subset of: {', '.join(ARMS)}")
    ap.add_argument("--cv", action="store_true",
                    help="also run the 5-fold CV recipes")
    ap.add_argument("--recipes", nargs="*", default=None,
                    help=f"CV subset of: {', '.join(CV_RECIPES)}")
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--cv-epochs", type=int, default=600)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)

    ensure_data()

    # ---- fixed split: the contrast roster ----
    todo = args.arms or list(ARMS)
    unknown = [a for a in todo if a not in ARMS]
    assert not unknown, f"unknown arms: {unknown}"
    dev = torch.device(args.device)
    B = load_bundle(dev)
    log(f"bundle: {B.NG} games | test {len(B.test_g)} val {len(B.val_g)} "
        f"| train pool {len(B.train_pool_games)} | queue: {todo}")
    for a in todo:
        spec = copy.copy(ARMS[a])
        spec.epochs = args.epochs
        spec.ckpt_every = args.ckpt_every
        run_arm(B, spec, train_fn=train_fn_for(spec), log_cb=log)

    # ---- optional 5-fold CV ----
    if args.cv:
        recipes = args.recipes or CV_RECIPES
        for fold in args.folds:
            Bf = load_bundle(dev, cv_fold=fold)
            log(f"fold {fold}: test {len(Bf.test_g)} val {len(Bf.val_g)} "
                f"train pool {len(Bf.train_pool_games)}")
            for r in recipes:
                base = CHAMPION if r == "champion_cegate2" else ARMS[r]
                spec = copy.copy(base)
                spec.name = f"cv_{r}_fold{fold}"
                spec.epochs = args.cv_epochs
                spec.ckpt_every = args.ckpt_every
                run_arm(Bf, spec, train_fn=train_fn_for(spec), log_cb=log)

    # ---- comparison table ----
    import contrast_experiment.report as report
    sys.argv = [sys.argv[0]]
    report.main()


if __name__ == "__main__":
    main()
