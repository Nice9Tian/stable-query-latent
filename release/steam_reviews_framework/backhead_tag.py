# -*- coding: utf-8 -*-
"""BackHead-TAG: 23-tag readout via an anchor-trained Ridge probe.

The probe is trained on anchor (gallery) features of tag-split train games,
alpha + a global decision threshold selected on tag-split val games, then
applied unchanged to any query feature (wiki variants at eval). Micro-F1 is
the reported tag metric.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

VARIANTS = ("positive", "neutral", "negative", "noname")
DEFAULT_ALPHAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0,
                  1000.0)


def atomic_json_write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def make_or_load_split(path: Path, names: list[str], args) -> dict:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {k: [str(x) for x in payload.get(k, [])]
                for k in ("train", "val", "test")}
    train_frac = float(getattr(args, "tag_text_train_frac", 0.7))
    val_frac = float(getattr(args, "tag_text_val_frac", 0.15))
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("tag split fractions must leave a positive test split")
    rng = np.random.default_rng(int(getattr(args, "tag_text_split_seed", 42)))
    ordered = np.asarray(list(names), dtype=object)
    perm = rng.permutation(len(ordered))
    n_train = max(1, int(round(len(ordered) * train_frac)))
    n_val = max(1, int(round(len(ordered) * val_frac)))
    if n_train + n_val >= len(ordered):
        n_train = max(1, len(ordered) - 2)
        n_val = 1
    payload = {
        "train": [str(x) for x in ordered[perm[:n_train]]],
        "val": [str(x) for x in ordered[perm[n_train:n_train + n_val]]],
        "test": [str(x) for x in ordered[perm[n_train + n_val:]]],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": int(getattr(args, "tag_text_split_seed", 42)),
    }
    atomic_json_write(payload, path)
    return {k: payload[k] for k in ("train", "val", "test")}


def micro_prf(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= float(threshold)
    truth = y_true > 0
    tp = float((pred & truth).sum())
    fp = float((pred & ~truth).sum())
    fn = float((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"micro_f1": f1, "precision": precision, "recall": recall}


def threshold_grid(scores: np.ndarray, steps: int) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0 or float(finite.min()) == float(finite.max()):
        return np.asarray([0.5], dtype=np.float32)
    return np.linspace(float(finite.min()), float(finite.max()),
                       max(2, int(steps)), dtype=np.float32)


def train_anchor_ridge(args, X_anchor, y, name_to_index, split):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    train_idx = np.asarray([name_to_index[n] for n in split["train"]
                            if n in name_to_index], dtype=np.int64)
    val_idx = np.asarray([name_to_index[n] for n in split["val"]
                          if n in name_to_index], dtype=np.int64)
    if train_idx.size < 2 or val_idx.size < 1:
        raise ValueError("tag split has too few train/val games")

    scaler = StandardScaler().fit(X_anchor[train_idx])
    Xtr = scaler.transform(X_anchor[train_idx])
    Xva = scaler.transform(X_anchor[val_idx])
    ytr = y[train_idx].astype(np.float32)
    yva = y[val_idx]

    best = None
    for alpha in DEFAULT_ALPHAS:
        model = Ridge(alpha=float(alpha))
        model.fit(Xtr, ytr)
        val_scores = model.predict(Xva)
        steps = int(getattr(args, "tag_text_threshold_steps", 33))
        for threshold in threshold_grid(val_scores, steps):
            metrics = micro_prf(yva, val_scores, float(threshold))
            key = (metrics["micro_f1"], metrics["recall"], -float(alpha))
            if best is None or key > best["key"]:
                best = {"key": key, "alpha": float(alpha),
                        "threshold": float(threshold), "metrics": metrics,
                        "model": model}
    return scaler, best["model"], best["alpha"], best["threshold"], best["metrics"]
