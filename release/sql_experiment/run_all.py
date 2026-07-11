# -*- coding: utf-8 -*-
"""Path 2 entry point: train ALL contrast arms (current design) and compare.

    python sql_experiment/run_all.py                 # every arm
    python sql_experiment/run_all.py --arms ce byol  # a subset
    python sql_experiment/run_all.py --epochs 300    # quick pass

Resume-safe: finished towers/heads are skipped, so the run can be
interrupted and relaunched freely. The champion itself is path 1
(sql_framework/train_champion.py); run it too for the full comparison table.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from sql_framework.data import load_bundle
from sql_framework.train import run_arm, train_tower
from sql_experiment.contrast_models.roster import ARMS
from sql_experiment.contrast_models.arcface import arc_loss_hook
from sql_experiment.contrast_models.byol import train_byol


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
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    todo = args.arms or list(ARMS)
    unknown = [a for a in todo if a not in ARMS]
    assert not unknown, f"unknown arms: {unknown}"
    B = load_bundle(torch.device(args.device))
    log = lambda *a: print(*a, flush=True)
    log(f"bundle: {B.NG} games | test {len(B.test_g)} val {len(B.val_g)} "
        f"| train pool {len(B.train_pool_games)} | queue: {todo}")
    for a in todo:
        spec = ARMS[a]
        spec.epochs = args.epochs
        spec.ckpt_every = args.ckpt_every
        run_arm(B, spec, train_fn=train_fn_for(spec), log_cb=log)
    log("run_all done — build the table with: python sql_experiment/report.py")


if __name__ == "__main__":
    main()
