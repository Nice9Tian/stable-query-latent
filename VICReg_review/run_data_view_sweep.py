"""Run the final data-size x view-fraction sweep for the Steam review encoder.

The sweep varies four axes:

* train-game-count: how many games are visible during self-supervised training.
* sample-fraction: the random review-view fraction used for the two VICReg views.
* output-dim: compact game-vector width after the hierarchical encoder.
* arm: GRL adversary enabled vs disabled.

Evaluation is always performed against the full 293-game H5 candidate pool. The
six long diagnostic texts are test-only here; the default description cache is
the train-only cache that excludes those texts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from VICReg_review.identity_diagnostic import (  # noqa: E402
    DEFAULT_LOCAL_MODEL,
    cases_from_defaults,
    encode_text_centroid,
    l2_normalize,
    participation_ratio,
    retrieval_rank,
    split_text,
    zscore_against_games,
)
from VICReg_review.train_tag_probe import (  # noqa: E402
    cross_validate as tag_cross_validate,
    extract_features,
    load_frozen_encoder,
    load_labels,
    pool_features,
    summarize as tag_summarize,
)

DEFAULT_H5 = SCRIPT_DIR / "h5" / "game_review_cleaned_3_sentences.h5"
DEFAULT_OUT_DIR = SCRIPT_DIR / "heads" / "data_view_sweep"
DEFAULT_DESCRIPTION_CACHE = SCRIPT_DIR / "heads" / "description_embedding_cache_train_only.npz"
DEFAULT_PYTHON = Path("C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe")


def atomic_text_write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_json_write(payload: dict, path: Path) -> None:
    atomic_text_write(json.dumps(payload, ensure_ascii=False, indent=2), path)


def decode_h5(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def run_command(cmd: list[str], cwd: Path) -> None:
    print("RUN " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def combo_name(output_dim: int, arm: str, train_games: int, view: float) -> str:
    return f"dim{output_dim:03d}_{arm_label(arm)}_n{train_games:03d}_view{int(round(view * 100)):02d}"


def arm_label(arm: str) -> str:
    return str(arm).strip().lower()


def arm_adversary_weight(arm: str) -> float:
    arm = arm_label(arm)
    if arm == "grl":
        return 10.0
    if arm in {"nogrl", "no_grl", "no-grl"}:
        return 0.0
    raise ValueError(f"Unknown arm: {arm}")


def combo_dir_for(args, output_dim: int, arm: str, train_games: int, view: float) -> Path:
    return args.out_dir / combo_name(output_dim, arm, train_games, view)


def build_train_command(args, output_dim: int, arm: str, train_games: int, view: float, combo_dir: Path) -> list[str]:
    return [
        str(args.python),
        str(SCRIPT_DIR / "train_vicreg_review_h5.py"),
        "--input-h5", str(args.h5),
        "--device", args.device,
        "--amp",
        "--epochs", str(args.epochs),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--batch-size", str(args.batch_size),
        "--sample-fraction", f"{view:g}",
        "--train-game-count", str(train_games),
        "--train-game-seed", str(args.train_game_seed),
        "--train-game-anchor-appids", args.train_game_anchor_appids,
        "--encoder-arch", "hierarchical",
        "--output-dim", str(output_dim),
        "--reduce-hidden", "128",
        "--vicreg-scope", "game",
        "--expander-dim", "512",
        "--expander-hidden", "256,512",
        "--compact-variance-weight", "25",
        "--compact-covariance-weight", "25",
        "--description-cache", str(args.description_cache),
        "--no-description-include-extra-cases",
        "--description-align-weight", "5",
        "--description-mse-weight", "10",
        "--recommendation-decorr-weight", "30",
        "--recommendation-target-transform", "logit",
        "--adversary-weight", f"{arm_adversary_weight(arm):g}",
        "--cache-mode", "queue",
        "--backward-mode", "recompute",
        "--probe-every", "0",
        "--checkpoint-out", str(combo_dir / "vicreg_review_h5_latest.pt"),
        "--best-checkpoint-out", str(combo_dir / "vicreg_review_h5_best.pt"),
        "--history-tsv", str(combo_dir / "vicreg_review_h5_history.tsv"),
        "--manifest-json", str(combo_dir / "vicreg_review_h5_manifest.json"),
        "--seed", str(args.seed),
    ]


def h5_game_metadata(h5_path: Path) -> tuple[list[str], list[str], list[str]]:
    with h5py.File(h5_path, "r") as h5:
        names = [decode_h5(x) for x in h5["game_names"][:]]
        appids = [decode_h5(x) for x in h5["appids"][:]]
        titles = [decode_h5(x) for x in h5["game_titles"][:]] if "game_titles" in h5 else names
    return names, appids, titles


def cache_raw_game_vectors(args, appids: list[str], names: list[str], titles: list[str]) -> dict:
    cache_path = args.out_dir / "raw_identity_cache_ms4000.npz"
    if cache_path.exists() and not args.rebuild_eval:
        data = np.load(cache_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    rng = np.random.default_rng(args.seed)
    X = []
    with h5py.File(args.h5, "r") as h5:
        game_offsets = h5["game_review_offsets"][:]
        review_offsets = h5["review_offsets"]
        vectors = h5["vectors"]
        for gi in range(len(appids)):
            review_start = int(game_offsets[gi])
            review_end = int(game_offsets[gi + 1])
            sentence_start = int(review_offsets[review_start])
            sentence_end = int(review_offsets[review_end])
            n_sent = sentence_end - sentence_start
            if n_sent <= args.max_game_sentences:
                block = vectors[sentence_start:sentence_end].astype(np.float32)
            else:
                selected = np.sort(rng.choice(n_sent, size=args.max_game_sentences, replace=False)) + sentence_start
                block = vectors[selected].astype(np.float32)
            X.append(block.mean(axis=0).astype(np.float32))
            if (gi + 1) % 50 == 0 or gi + 1 == len(appids):
                print(f"raw cache {gi + 1}/{len(appids)}", flush=True)
    payload = {
        "X": np.stack(X, axis=0).astype(np.float32),
        "appids": np.asarray(appids, dtype=object),
        "names": np.asarray(names, dtype=object),
        "titles": np.asarray(titles, dtype=object),
    }
    with cache_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return payload


def embed_test_cases(args) -> dict:
    cache_path = args.out_dir / "test_case_embeddings.npz"
    if cache_path.exists() and not args.rebuild_eval:
        data = np.load(cache_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    from game_review_data.embedding_data import LocalEmbedder

    embedder = LocalEmbedder(args.local_model, device=args.device, batch_size=args.embed_batch_size)
    vectors = []
    offsets = [0]
    games = []
    appids = []
    sentiments = []
    paths = []
    sentence_counts = []
    for case in cases_from_defaults():
        text = case["path"].read_text(encoding="utf-8")
        sentences = split_text(text, args.max_text_sentences)
        embedded = np.asarray(embedder.embed(sentences), dtype=np.float32)
        vectors.append(embedded)
        offsets.append(offsets[-1] + embedded.shape[0])
        games.append(case["game"])
        appids.append(case["appid"])
        sentiments.append(case["sentiment"])
        paths.append(str(case["path"]))
        sentence_counts.append(len(sentences))
        print(f"embedded test text {case['game']} {case['sentiment']} sentences={len(sentences)}", flush=True)
    payload = {
        "vectors": np.concatenate(vectors, axis=0).astype(np.float32),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "games": np.asarray(games, dtype=object),
        "appids": np.asarray(appids, dtype=object),
        "sentiments": np.asarray(sentiments, dtype=object),
        "paths": np.asarray(paths, dtype=object),
        "sentence_counts": np.asarray(sentence_counts, dtype=np.int32),
    }
    with cache_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return payload


def build_vicreg_feature_cache(args, checkpoint: Path, combo_dir: Path) -> tuple[np.ndarray, list[str]]:
    cache_path = combo_dir / "eval_features_full293_fv4.npz"
    if cache_path.exists() and not args.rebuild_eval:
        data = np.load(cache_path, allow_pickle=True)
        return data["feats"].astype(np.float32), [str(n) for n in data["names"]]

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    with h5py.File(args.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])
    encoder, _, _, _ = load_frozen_encoder(checkpoint, input_dim, device)
    with torch.no_grad():
        feats, names = extract_features(
            encoder,
            str(args.h5),
            args.eval_sample_fraction,
            args.eval_feature_views,
            args.seed,
            "float16",
            device,
            args.amp_eval and device.type == "cuda",
        )
    with cache_path.open("wb") as handle:
        np.savez_compressed(handle, feats=feats.astype(np.float32), names=np.asarray(names, dtype=object))
    return feats.astype(np.float32), list(names)


def tag_probe_metrics(args, feats: np.ndarray, feature_names: list[str]) -> dict:
    tags, label_names, labels = load_labels(None, str(args.h5))
    index = {n: i for i, n in enumerate(label_names)}
    y = np.zeros((len(feature_names), labels.shape[1]), dtype=np.int8)
    keep = np.zeros(len(feature_names), dtype=bool)
    for row, name in enumerate(feature_names):
        if name in index:
            y[row] = labels[index[name]]
            keep[row] = True
    X = pool_features(feats[keep], "flatten")
    y = y[keep]
    probe_args = SimpleNamespace(
        folds=args.probe_folds,
        seed=args.seed,
        min_train_pos=2,
        C=1.0,
        norm_eps=1e-8,
        freq_floors=[5, 10, 20, 30, 40, 60, 80],
    )
    per_tag_tp, per_tag_fp, per_tag_fn, scored, fold_f1s = tag_cross_validate(X, y, tags, probe_args)
    return tag_summarize(per_tag_tp, per_tag_fp, per_tag_fn, scored, fold_f1s, y, tags, probe_args)


def sentiment_r2(args, X: np.ndarray, names: list[str]) -> dict:
    cache = np.load(SCRIPT_DIR / "tags" / "game_sentiment.npz", allow_pickle=True)
    targets = {str(n): float(s) for n, s in zip(cache["names"], cache["sent"])}
    rows = [i for i, n in enumerate(names) if n in targets]
    y = np.asarray([targets[names[i]] for i in rows], dtype=np.float64)
    X = X[rows].astype(np.float32)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    pred = np.zeros(len(y), dtype=np.float64)
    for tr, va in kfold_indices(len(y), args.probe_folds, args.seed):
        scaler = StandardScaler().fit(X[tr])
        model = Ridge(alpha=10.0).fit(scaler.transform(X[tr]), y[tr])
        pred[va] = model.predict(scaler.transform(X[va]))
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 2 and np.std(pred) > 0 else float("nan")
    return {"r2": float(r2), "pearson": corr, "n": int(len(y))}


def kfold_indices(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, val


def recommendation_probe(args, X: np.ndarray, names: list[str]) -> dict:
    from backheads.train_recommendation_head import DEFAULT_REVIEWS_DIR, load_labels_for_h5
    from backheads.train_recommendation_vicreg_linear_probe import cross_validate as reco_cv, summarize_cv

    rows, keep_indices, _ = load_labels_for_h5(
        args.h5,
        DEFAULT_REVIEWS_DIR,
        label_min_length=0,
        min_label_count=10,
    )
    name_to_row = {name: i for i, name in enumerate(names)}
    with h5py.File(args.h5, "r") as h5:
        h5_names = [decode_h5(x) for x in h5["game_names"][:]]
    selected = []
    labels = []
    for row, game_index in zip(rows, keep_indices):
        game_name = h5_names[int(game_index)]
        if game_name not in name_to_row:
            continue
        selected.append(name_to_row[game_name])
        labels.append([row.positive_rate, row.negative_rate])
    reco_args = SimpleNamespace(
        folds=args.probe_folds,
        inner_folds=3,
        seed=args.seed,
        target_transform="logit",
        logit_eps=1e-4,
        alphas=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
    )
    folds = reco_cv(X[np.asarray(selected)].astype(np.float32), np.asarray(labels, dtype=np.float32), reco_args)
    return summarize_cv(folds)


def identity_metrics(args, checkpoint: Path, feats: np.ndarray, names: list[str], raw_cache: dict, text_cache: dict) -> dict:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    with h5py.File(args.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])
    encoder, _, _, _ = load_frozen_encoder(checkpoint, input_dim, device)
    encode_args = SimpleNamespace(
        feature_views=args.eval_feature_views,
        sample_fraction=args.eval_sample_fraction,
        amp=args.amp_eval,
        seed=args.seed,
    )
    appids = [str(x) for x in raw_cache["appids"]]
    titles = [str(x) for x in raw_cache["titles"]]
    X_raw = raw_cache["X"].astype(np.float32)
    X_vic = feats.mean(axis=1).astype(np.float32)
    rows = []
    text_features = {}
    offsets = text_cache["offsets"].astype(np.int64)
    for i, (game, appid, sentiment) in enumerate(zip(text_cache["games"], text_cache["appids"], text_cache["sentiments"])):
        vectors = text_cache["vectors"][int(offsets[i]): int(offsets[i + 1])].astype(np.float32)
        raw_query = vectors.mean(axis=0).astype(np.float32)
        vic_query = encode_text_centroid(encoder, vectors, encode_args, device)
        raw_rank, raw_sim, _, _ = retrieval_rank(X_raw, raw_query, appids, str(appid), 3)
        vic_rank, vic_sim, _, _ = retrieval_rank(X_vic, vic_query, appids, str(appid), 3)
        rows.append({
            "game": str(game),
            "appid": str(appid),
            "sentiment": str(sentiment),
            "raw_rank": int(raw_rank),
            "raw_similarity": float(raw_sim),
            "vicreg_rank": int(vic_rank),
            "vicreg_similarity": float(vic_sim),
        })
        text_features[(str(game), str(sentiment))] = {"raw": raw_query, "vicreg": vic_query}

    pair_rows = []
    for game in sorted({key[0] for key in text_features}):
        sentiments = sorted(key[1] for key in text_features if key[0] == game)
        for a_index in range(len(sentiments)):
            for b_index in range(a_index + 1, len(sentiments)):
                a = sentiments[a_index]
                b = sentiments[b_index]
                fa = text_features[(game, a)]
                fb = text_features[(game, b)]
                _, raw_a = zscore_against_games(X_raw, fa["raw"])
                _, raw_b = zscore_against_games(X_raw, fb["raw"])
                _, vic_a = zscore_against_games(X_vic, fa["vicreg"])
                _, vic_b = zscore_against_games(X_vic, fb["vicreg"])
                raw_cos = float((l2_normalize(raw_a[None, :]) @ l2_normalize(raw_b[None, :]).T)[0, 0])
                vic_cos = float((l2_normalize(vic_a[None, :]) @ l2_normalize(vic_b[None, :]).T)[0, 0])
                pair_rows.append({"game": game, "pair": f"{a} vs {b}", "raw_cosine": raw_cos, "vicreg_cosine": vic_cos})
    ranks = [row["vicreg_rank"] for row in rows]
    return {
        "participation_ratio": participation_ratio(X_vic)["pr"],
        "zscore_participation_ratio": participation_ratio(X_vic, zscore=True)["pr"],
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "hit_at_1": float(np.mean([rank <= 1 for rank in ranks])),
        "hit_at_5": float(np.mean([rank <= 5 for rank in ranks])),
        "hit_at_100": float(np.mean([rank <= 100 for rank in ranks])),
        "mean_vicreg_cosine": float(np.mean([row["vicreg_cosine"] for row in pair_rows])) if pair_rows else float("nan"),
        "retrieval_rows": rows,
        "pair_rows": pair_rows,
    }


def evaluate_combo(args, checkpoint: Path, combo_dir: Path) -> dict:
    report_path = combo_dir / "eval_report.json"
    if report_path.exists() and not args.rebuild_eval:
        return json.loads(report_path.read_text(encoding="utf-8"))

    names, appids, titles = h5_game_metadata(args.h5)
    raw_cache = cache_raw_game_vectors(args, appids, names, titles)
    text_cache = embed_test_cases(args)
    feats, feature_names = build_vicreg_feature_cache(args, checkpoint, combo_dir)
    X_stats = pool_features(feats, "stats").astype(np.float32)

    report = {
        "checkpoint": str(checkpoint.resolve()),
        "tag_probe": tag_probe_metrics(args, feats, feature_names),
        "sentiment_probe": sentiment_r2(args, X_stats, feature_names),
        "recommendation_probe": recommendation_probe(args, X_stats, feature_names),
        "identity": identity_metrics(args, checkpoint, feats, feature_names, raw_cache, text_cache),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_json_write(report, report_path)
    return report


def scalar_from_report(report: dict) -> dict:
    return {
        "tag_micro_f1": float(report["tag_probe"]["micro_f1"]),
        "tag_fold_std": float(report["tag_probe"]["fold_micro_f1_std"]),
        "sentiment_r2": float(report["sentiment_probe"]["r2"]),
        "recommendation_pearson": float(report["recommendation_probe"]["pearson_mean"]),
        "recommendation_mae": float(report["recommendation_probe"]["mae_mean"]),
        "pr": float(report["identity"]["participation_ratio"]),
        "mean_rank": float(report["identity"]["mean_rank"]),
        "median_rank": float(report["identity"]["median_rank"]),
        "hit_at_1": float(report["identity"]["hit_at_1"]),
        "hit_at_5": float(report["identity"]["hit_at_5"]),
        "hit_at_100": float(report["identity"]["hit_at_100"]),
        "mean_text_cosine": float(report["identity"]["mean_vicreg_cosine"]),
    }


def composite_score(row: dict) -> float:
    tag = row["tag_micro_f1"]
    identity = row["hit_at_5"]
    cos = (row["mean_text_cosine"] + 1.0) / 2.0
    sent_penalty = max(0.0, row["sentiment_r2"])
    reco_penalty = abs(row["recommendation_pearson"])
    pr_bonus = min(row["pr"] / 25.0, 1.0)
    return float(0.30 * tag + 0.30 * identity + 0.15 * cos + 0.15 * pr_bonus - 0.05 * sent_penalty - 0.05 * reco_penalty)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def render_report(rows: list[dict], args) -> str:
    complete = [row for row in rows if row["status"] == "done"]
    best = max(complete, key=lambda row: row["composite_score"]) if complete else None
    paired = []
    by_key = {(row.get("output_dim"), row.get("arm"), row["train_games"], row["view_fraction"]): row for row in complete}
    for output_dim in args.output_dims:
        for train_games in args.train_game_counts:
            for view in args.sample_fractions:
                grl = by_key.get((output_dim, "grl", train_games, view))
                nogrl = by_key.get((output_dim, "nogrl", train_games, view))
                if not grl or not nogrl:
                    continue
                paired.append({
                    "output_dim": output_dim,
                    "train_games": train_games,
                    "view_fraction": view,
                    "delta_score": grl["composite_score"] - nogrl["composite_score"],
                    "delta_tag": grl["tag_micro_f1"] - nogrl["tag_micro_f1"],
                    "delta_sentiment_r2": grl["sentiment_r2"] - nogrl["sentiment_r2"],
                    "delta_reco_pearson_abs": abs(grl["recommendation_pearson"]) - abs(nogrl["recommendation_pearson"]),
                    "delta_hit_at_5": grl["hit_at_5"] - nogrl["hit_at_5"],
                    "delta_pr": grl["pr"] - nogrl["pr"],
                })
    lines = [
        "# 数据量 x View Fraction 收尾实验报告",
        "",
        f"- 日期：{time.strftime('%Y-%m-%d')}",
        f"- 输出维度轴：{', '.join(str(x) for x in args.output_dims)}",
        f"- 总游戏数：293；实验数据量轴：{', '.join(str(x) for x in args.train_game_counts)}",
        f"- view fraction：{', '.join(f'{v:.1f}' for v in args.sample_fractions)}",
        f"- 对照 arm：{', '.join(args.arms)}（GRL=adversary_weight 10；no-GRL=adversary_weight 0）。",
        "- 评估候选池：始终使用全量 293 款游戏。",
        "- 测试文本：BG3、Cyberpunk、Across the Obelisk 的官方描述/长文本只在测试阶段使用；训练使用 train-only description cache。",
        f"- 每组合训练预算：epochs={args.epochs}, steps_per_epoch={args.steps_per_epoch}, batch_size={args.batch_size}。",
        "",
    ]
    if best:
        lines.extend([
            "## 结论",
            "",
            f"当前 sweep 的最佳综合窗口是 **dim={best['output_dim']}、arm={best['arm']}、N={best['train_games']}、view={best['view_fraction']:.1f}**。",
            f"综合分 {best['composite_score']:.3f}，TAG micro-F1 {best['tag_micro_f1']:.3f}，"
            f"身份 Hit@5 {best['hit_at_5']:.3f}，PR {best['pr']:.2f}，"
            f"情感 R² {best['sentiment_r2']:.3f}，好评率 Pearson {best['recommendation_pearson']:.3f}。",
            "",
            "综合分权重为：TAG 0.30、身份 Hit@5 0.30、同游戏情绪文本 cosine 0.15、PR 0.15，"
            "并对情感 R² 与好评率 Pearson 各扣 0.05。它不是论文指标，只用于窗口选择。",
            "",
        ])
    lines.extend([
        "## 数据量-性能曲线",
        "",
        "| dim | arm | N | view | score | TAG F1 | sentiment R² | reco Pearson | PR | mean rank | Hit@1 | Hit@5 | Hit@100 | text cosine |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(complete, key=lambda x: (x["output_dim"], x["arm"], x["train_games"], x["view_fraction"])):
        lines.append(
            f"| {row['output_dim']} | {row['arm']} | {row['train_games']} | {row['view_fraction']:.1f} | {row['composite_score']:.3f} | "
            f"{row['tag_micro_f1']:.3f} | {row['sentiment_r2']:.3f} | {row['recommendation_pearson']:.3f} | "
            f"{row['pr']:.2f} | {row['mean_rank']:.1f} | {row['hit_at_1']:.3f} | {row['hit_at_5']:.3f} | "
            f"{row['hit_at_100']:.3f} | {row['mean_text_cosine']:.3f} |"
        )

    lines.extend([
        "",
        "## GRL 对照差值",
        "",
        "| dim | N | view | Δscore | ΔTAG F1 | Δsentiment R² | Δabs(reco Pearson) | ΔHit@5 | ΔPR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in paired:
        lines.append(
            f"| {row['output_dim']} | {row['train_games']} | {row['view_fraction']:.1f} | {row['delta_score']:+.3f} | "
            f"{row['delta_tag']:+.3f} | {row['delta_sentiment_r2']:+.3f} | "
            f"{row['delta_reco_pearson_abs']:+.3f} | {row['delta_hit_at_5']:+.3f} | {row['delta_pr']:+.2f} |"
        )
    lines.extend([
        "",
        "说明：Δ = GRL - no-GRL。对情感 R² 和 abs(reco Pearson)，负数表示 GRL 更好；"
        "对 TAG、Hit@5、PR，正数表示 GRL 更好。",
        "",
        "## View 最佳窗口预测",
        "",
        "| dim | arm | view | 平均 score | 平均 TAG F1 | 平均 Hit@5 | 平均 PR | 平均 sentiment R² |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for output_dim in args.output_dims:
        for arm in args.arms:
            for view in args.sample_fractions:
                subset = [
                    row for row in complete
                    if row["output_dim"] == output_dim
                    and row["arm"] == arm
                    and abs(row["view_fraction"] - view) < 1e-8
                ]
                if not subset:
                    continue
                lines.append(
                    f"| {output_dim} | {arm} | {view:.1f} | {np.mean([r['composite_score'] for r in subset]):.3f} | "
                    f"{np.mean([r['tag_micro_f1'] for r in subset]):.3f} | "
                    f"{np.mean([r['hit_at_5'] for r in subset]):.3f} | "
                    f"{np.mean([r['pr'] for r in subset]):.2f} | "
                    f"{np.mean([r['sentiment_r2'] for r in subset]):.3f} |"
                )

    lines.extend([
        "",
        "## 备注",
        "",
        "- N 是训练阶段可见的游戏数量，不是每款游戏的评论条数。",
        "- BG3、Cyberpunk、Across the Obelisk 三个锚点在每个训练子集里固定保留，以免身份召回测试变成“目标未见过”的外推问题。",
        "- 若某组合 status 不是 done，它不会进入上面的曲线均值；原始 JSON 保存在各组合目录。",
    ])
    return "\n".join(lines) + "\n"


def summarize(args) -> list[dict]:
    rows = []
    for output_dim in args.output_dims:
        for arm in args.arms:
            arm = arm_label(arm)
            for train_games in args.train_game_counts:
                for view in args.sample_fractions:
                    name = combo_name(output_dim, arm, train_games, view)
                    combo_dir = combo_dir_for(args, output_dim, arm, train_games, view)
                    eval_path = combo_dir / "eval_report.json"
                    manifest_path = combo_dir / "vicreg_review_h5_manifest.json"
                    row = {
                        "output_dim": output_dim,
                        "arm": arm,
                        "combo": name,
                        "train_games": train_games,
                        "view_fraction": view,
                    }
                    if eval_path.exists():
                        report = json.loads(eval_path.read_text(encoding="utf-8"))
                        row.update(scalar_from_report(report))
                        row["composite_score"] = composite_score(row)
                        row["status"] = "done"
                    elif manifest_path.exists():
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        row["status"] = str(manifest.get("status", "missing_eval"))
                    else:
                        row["status"] = "missing"
                    rows.append(row)
    write_csv(rows, args.out_dir / "data_view_sweep_summary.csv")
    atomic_text_write(render_report(rows, args), args.out_dir / "DATA_VIEW_SWEEP_REPORT.md")
    atomic_json_write({"rows": rows, "args": vars_for_json(args)}, args.out_dir / "data_view_sweep_summary.json")
    return rows


def vars_for_json(args) -> dict:
    payload = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, list):
            payload[key] = [str(v) if isinstance(v, Path) else v for v in value]
        else:
            payload[key] = value
    return payload


def run(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for output_dim in args.output_dims:
        for train_games in args.train_game_counts:
            for view in args.sample_fractions:
                for arm in args.arms:
                    arm = arm_label(arm)
                    combo_dir = combo_dir_for(args, output_dim, arm, train_games, view)
                    combo_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint = combo_dir / "vicreg_review_h5_latest.pt"
                    manifest = combo_dir / "vicreg_review_h5_manifest.json"
                    if not args.skip_train:
                        needs_train = args.force_train or not checkpoint.exists()
                        if manifest.exists() and not args.force_train:
                            try:
                                needs_train = json.loads(manifest.read_text(encoding="utf-8")).get("status") != "done"
                            except json.JSONDecodeError:
                                needs_train = True
                        if needs_train:
                            run_command(build_train_command(args, output_dim, arm, train_games, view, combo_dir), ROOT)
                    if checkpoint.exists() and not args.skip_eval:
                        evaluate_combo(args, checkpoint, combo_dir)
                    summarize(args)
    rows = summarize(args)
    done = sum(1 for row in rows if row["status"] == "done")
    print(f"sweep summary: {done}/{len(rows)} combinations done -> {args.out_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default=DEFAULT_H5, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--python", default=DEFAULT_PYTHON, type=Path)
    parser.add_argument("--train-game-counts", type=int, nargs="+", default=[50, 100, 150, 200, 250, 293])
    parser.add_argument("--sample-fractions", type=float, nargs="+", default=[0.8, 0.6, 0.4, 0.2])
    parser.add_argument("--output-dims", type=int, nargs="+", default=[18, 36, 72])
    parser.add_argument("--arms", nargs="+", default=["grl", "nogrl"], choices=["grl", "nogrl"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-game-seed", type=int, default=20260626)
    parser.add_argument("--train-game-anchor-appids", default="1086940,1091500,1385380")
    parser.add_argument("--description-cache", default=DEFAULT_DESCRIPTION_CACHE, type=Path)
    parser.add_argument("--eval-feature-views", type=int, default=4)
    parser.add_argument("--eval-sample-fraction", type=float, default=0.6)
    parser.add_argument("--probe-folds", type=int, default=5)
    parser.add_argument("--max-game-sentences", type=int, default=4000)
    parser.add_argument("--max-text-sentences", type=int, default=4096)
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--amp-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-eval", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
