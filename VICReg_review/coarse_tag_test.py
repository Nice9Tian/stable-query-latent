"""Coarse-tag probe with L2-normalized features, tested on real descriptions.

Collapses the redundant fine Steam tags (Turn-Based / Turn-Based Strategy /
Combat / Tactics -> "Turn-Based"; Card Game / Card Battler / Deckbuilding ->
"Deckbuilder"; the 7 RPG variants -> "RPG"; etc.) into ~30 coarse families to
lower the task difficulty, and L2-normalizes features (kills the StandardScaler
saturation on out-of-domain text). Reports 5-fold CV on the 293 games plus a
prec@K / recall test on AO_text.txt and 2077_text.txt, for both the raw Qwen
embedding and the frozen VICReg code.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (str(ROOT), str(ROOT / "game_review_data")):
    if p not in sys.path:
        sys.path.insert(0, p)

from VICReg_review.tap_mapping import load_tap_mapping, map_tag_dict, keyword_scores  # noqa: E402
from VICReg_review import disturbtion_embed  # noqa: E402
from VICReg_review.train_tag_probe import load_frozen_encoder, pool_features  # noqa: E402

GAMES_JSON = ROOT / "game_review_data" / "games.json"
DEFAULT_H5 = ROOT / "game_review_data" / "embedding_h5.h5"
TESTS = [("AO_text.txt", "1385380", "Across the Obelisk"), ("2077_text.txt", "1091500", "Cyberpunk 2077")]


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def micro_prf(pred, true):
    tp = float((pred & true).sum()); fp = float((pred & ~true).sum()); fn = float((~pred & true).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def kfold(n, k, seed=42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        va = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        yield tr, va


def cv_micro_f1(F, Y, seed=42):
    from sklearn.linear_model import LogisticRegression
    tp = fp = fn = 0.0
    for tr, va in kfold(len(Y), 5, seed):
        for c in range(Y.shape[1]):
            if Y[tr, c].sum() < 2 or Y[va, c].sum() < 1:
                continue
            clf = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced").fit(F[tr], Y[tr, c])
            trp = clf.predict_proba(F[tr])[:, 1]
            best_thr, best = 0.5, -1.0
            for thr in np.linspace(0.1, 0.9, 17):
                f1, _, _ = micro_prf(trp >= thr, Y[tr, c].astype(bool))
                if f1 > best:
                    best, best_thr = f1, thr
            pred = clf.predict_proba(F[va])[:, 1] >= best_thr
            tru = Y[va, c].astype(bool)
            tp += float((pred & tru).sum()); fp += float((pred & ~tru).sum()); fn += float((~pred & tru).sum())
    return micro_prf(np.array([True]), np.array([True]))[0] if False else (
        (2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))))
        if (tp + fp) and (tp + fn) and (tp / (tp + fp) + tp / (tp + fn)) else 0.0)


def fit_full(F, Y):
    from sklearn.linear_model import LogisticRegression
    clfs = {}
    for c in range(Y.shape[1]):
        if Y[:, c].sum() < 2:
            continue
        clfs[c] = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced").fit(F, Y[:, c])
    return clfs


def predict_desc(clfs, x):
    probs = np.full(len(COARSE_NAMES), -1.0)
    for c, clf in clfs.items():
        probs[c] = clf.predict_proba(x[None, :])[0, 1]
    return probs


def blend_keywords(text, probs, weight=0.6):
    prior = keyword_scores(text, COARSE_NAMES)
    return (1.0 - weight) * probs + weight * prior


def score_desc(name, true_coarse, probs, K):
    order = np.argsort(-probs)[:K]
    pred = [COARSE_NAMES[i] for i in order]
    hits = [t for t in pred if t in true_coarse]
    prec = len(hits) / K
    rec = len([t for t in true_coarse if t in set(pred)]) / len(true_coarse) if true_coarse else 0.0
    print(f"  {name}: prec@{K}={prec:.2f} recall@{K}={rec:.2f}  (true coarse n={len(true_coarse)})")
    print(f"    top-{K}: {pred}")
    print(f"    hits ({len(hits)}): {hits}")
    return prec, rec


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    games = json.loads(GAMES_JSON.read_text(encoding="utf-8"))
    mapping = load_tap_mapping()

    # labels aligned to the cached feature game order
    with __import__("h5py").File(DEFAULT_H5, "r") as h5:
        global COARSE_NAMES
        COARSE_NAMES = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h5["tap_names"][:]]
        appids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h5["appids"][:]]
        Y = (h5["tap_labels"][:] > 0).astype(np.int8)
    print(f"coarse tags: {len(COARSE_NAMES)}  games: {len(Y)}  avg coarse/game: {Y.sum(1).mean():.1f}")

    # raw review features (cached) -> L2
    rawd = np.load(SCRIPT_DIR / "tags" / "raw_mean_features.npz", allow_pickle=True)
    raw_names = [str(n) for n in rawd["names"]]; raw_map = {n.split("_")[0]: rawd["feats"][i] for i, n in enumerate(raw_names)}
    F_raw = l2(np.stack([raw_map[a] for a in appids]))

    # vicreg code features (cached) -> stats pool -> L2
    vicd = np.load(SCRIPT_DIR / "tags" / "probe_feat_vicreg_adv10_best_fv4_sf0.6.npz", allow_pickle=True)
    vic_names = [str(n) for n in vicd["names"]]
    vic_map = {n.split("_")[0]: vicd["feats"][i] for i, n in enumerate(vic_names)}
    F_vic = l2(pool_features(np.stack([vic_map[a] for a in appids]), "stats"))

    print("\n5-fold CV micro-F1 (in-domain, coarse tags):")
    print(f"  raw-L2     : {cv_micro_f1(F_raw, Y):.3f}")
    print(f"  vicreg-L2  : {cv_micro_f1(F_vic, Y):.3f}")

    # Load pre-embedded descriptions; build raw + vicreg features from npz.
    from game_review_data.embedding_data import DEFAULT_LOCAL_MODEL

    cache_path = SCRIPT_DIR / "heads" / "data_view_sweep" / "test_case_embeddings.npz"
    disturbtion_embed.ensure_test_case_cache(
        cache_path,
        local_model=DEFAULT_LOCAL_MODEL,
        device=str(device),
        batch_size=32,
        max_text_sentences=4096,
    )
    text_cache = disturbtion_embed.load_npz_payload(cache_path)
    offsets = text_cache["offsets"].astype(np.int64)
    by_appid_sentiment = {
        (str(appid), str(sentiment)): text_cache["vectors"][int(offsets[i]): int(offsets[i + 1])].astype(np.float32)
        for i, (appid, sentiment) in enumerate(zip(text_cache["appids"], text_cache["sentiments"]))
    }
    text_by_appid_sentiment = {
        (str(appid), str(sentiment)): str(text)
        for appid, sentiment, text in zip(text_cache["appids"], text_cache["sentiments"], text_cache["texts"])
    } if "texts" in text_cache else {}
    probe = torch.load(SCRIPT_DIR / "heads" / "tag_probe_linear.pt", map_location="cpu", weights_only=False)
    with __import__("h5py").File(DEFAULT_H5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])
    encoder, _, _, _ = load_frozen_encoder(probe["encoder_checkpoint"], input_dim, device)

    def desc_features(appid):
        vecs = by_appid_sentiment[(str(appid), "neutral")]
        raw_feat = l2(vecs.mean(0))
        vt = torch.tensor(vecs, device=device)
        rng = np.random.default_rng(0); codes = []
        with torch.no_grad():
            for _ in range(4):
                if vt.shape[0] > 2:
                    idx = np.sort(rng.choice(vt.shape[0], max(1, int(np.ceil(vt.shape[0] * 0.6))), replace=False))
                    sub = vt[idx]
                else:
                    sub = vt
                codes.append(encoder(sub.unsqueeze(0), key_padding_mask=None).squeeze(0).float())
        code = torch.stack(codes, 0).mean(0).cpu().numpy()
        vic_feat = l2(pool_features(code[None, ...], "stats")[0])
        return raw_feat, vic_feat

    clfs_raw, clfs_vic = fit_full(F_raw, Y), fit_full(F_vic, Y)
    for path, appid, name in TESTS:
        true_coarse = set(map_tag_dict(games[appid].get("tags", {}), mapping))
        text = text_by_appid_sentiment.get((str(appid), "neutral"), "")
        raw_feat, vic_feat = desc_features(appid)
        K = len(true_coarse)
        print(f"\n=== {name} ===  true coarse: {sorted(true_coarse)}")
        print(" raw-L2 probe:")
        score_desc("raw", true_coarse, predict_desc(clfs_raw, raw_feat), K)
        print(" raw-L2 + keyword probe:")
        score_desc("raw+kw", true_coarse, blend_keywords(text, predict_desc(clfs_raw, raw_feat)), K)
        print(" vicreg-L2 probe:")
        score_desc("vicreg", true_coarse, predict_desc(clfs_vic, vic_feat), K)
        print(" vicreg-L2 + keyword probe:")
        score_desc("vicreg+kw", true_coarse, blend_keywords(text, predict_desc(clfs_vic, vic_feat)), K)


if __name__ == "__main__":
    main()
