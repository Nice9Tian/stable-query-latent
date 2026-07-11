# -*- coding: utf-8 -*-
"""Protocol layer: the fixed wiki split, the fully-inductive exclusion rule,
and the vsel model-selection score.

FULLY INDUCTIVE (decree 2026-07-11): no text of a held-out game enters any
training stage — reviews, documents, pseudo-queries, or gallery-negative
gradients. Training samples AND the training-time gallery cover only the
train games; the full gallery is used with a frozen tower at eval only.
"""
import json
import math

import numpy as np

from dataset_builder.paths import SPLIT_JSON

SPLIT_SEED = 20260711


def load_split(appid2name, cv_fold=None, cv_seed=SPLIT_SEED):
    """Return (test_g, val_g, train_doc_g) as sets of full game names.

    Fixed protocol: the pinned wiki_eval_split.json (204/203/407).
    CV protocol (cv_fold 0..4): the same 814-game universe permuted by
    cv_seed; fold k = test, fold (k+1)%5 = val, the other three = train.
    """
    sp = json.loads(SPLIT_JSON.read_text())
    assert sp["seed"] == SPLIT_SEED, "authoritative split json mismatch"
    if cv_fold is None:
        return ({appid2name[a] for a in sp["test"]},
                {appid2name[a] for a in sp["val"]},
                {appid2name[a] for a in sp["train"]})
    universe = sorted(sp["test"] + sp["val"] + sp["train"])
    perm = np.random.default_rng(cv_seed).permutation(len(universe))
    folds = np.array_split(perm, 5)
    te = {universe[i] for i in folds[cv_fold]}
    va = {universe[i] for i in folds[(cv_fold + 1) % 5]}
    tr = set(universe) - te - va
    return ({appid2name[a] for a in te}, {appid2name[a] for a in va},
            {appid2name[a] for a in tr})


# ---- vsel: the model-selection score over val-game retrieval hits --------
# S(x) = -k1*(exp(a*(x0-x))-1) below target x0 (exponential penalty),
#        k2*(x-x0)^b at/above (polynomial bonus).
# vsel = max(S(non@1;0.45), S(non@5;0.65)) + S(neu@1;0.85):
# noname generalization is the OBJECTIVE (best of its two axes); neutral is
# an ADDITIVE sanity gate — a collapsed neutral pulls the total down
# exponentially and excludes the candidate.
K1, K2, S_A, S_B = 1.0, 1.0, 10.0, 1.0
VSEL_TARGETS = {"non1": 0.45, "non5": 0.65, "neu1": 0.85}


def S_fn(x: float, x0: float) -> float:
    return (-K1 * (math.exp(S_A * (x0 - x)) - 1) if x < x0
            else K2 * (x - x0) ** S_B)


def vsel_score(h_neu: float, h_non: float, h_non5: float) -> float:
    return (max(S_fn(h_non, VSEL_TARGETS["non1"]),
                S_fn(h_non5, VSEL_TARGETS["non5"]))
            + S_fn(h_neu, VSEL_TARGETS["neu1"]))
