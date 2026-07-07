# -*- coding: utf-8 -*-
"""5-fold CV worker: one (recipe, fold) cell = tower + 10 userft-MRR heads + full metrics.

Runs on a single GPU (launch one process per GPU via fivefold_cv.ipynb; pin the card
with CUDA_VISIBLE_DEVICES). Protocol is the champion recipe from the fusion pilot
(MEMO_fusion_pilot.md rounds 9-11), with the 258 article games re-partitioned into
5 folds instead of the fixed 77-game test split:

  tower   : SetPool N latents, W=16 fresh masks/step from the 2048-sent pool,
            single-view InfoNCE vs the full 2020-anchor gallery (+ optional 2nd
            view with cosine pull), AMP, val = Hit@1 on 216 pseudo queries.
  heads   : Linear->128 (x10 seeds). Phase 1 self-sup InfoNCE on pseudo queries
            (fold-val games excluded); phase 2 = user objective on the fold's
            train neutral articles (CE vs [gallery ; in-batch reals] + cosine
            pull to own anchor), early-stopped by val-MRR on fold-val neutrals.
  metrics : Hit@1/Hit@5 x {neutral, noname} on the fold's test games + anchor-ridge
            tag micro-F1 (protocol split seed 42, unchanged across folds).

Resume: skips work if the output JSON already exists (pass --overwrite to redo).
Output: <out-dir>/fivefold_<recipe>_fold<k>.json  (per-seed metrics + fold config).
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPES = {
    "n4pull300": dict(N=4, W=16, pull=1.0, epochs=300, patience=48),
    "n4pull150": dict(N=4, W=16, pull=1.0, epochs=150, patience=24),
    "n4fresh300": dict(N=4, W=16, pull=0.0, epochs=300, patience=9999),
}
SPLIT_SEED = 20260705   # same rng as the fixed 77/27/154 protocol split
VAL_FRAC = 0.15         # of the non-test games, in permutation order
DM, HEADS = 128, 4


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="fusion_cache dir (npz/npy corpus)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", required=True, help="repo root (for VICReg_review import)")
    ap.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--head-epochs", type=int, default=600)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


class SetPoolN(nn.Module):
    def __init__(s, N):
        super().__init__()
        s.q0 = nn.Parameter(torch.randn(1, N, DM) * 0.02)
        s.attn = nn.MultiheadAttention(DM, HEADS, kdim=1024, vdim=1024, batch_first=True)
        s.head = nn.Sequential(nn.Linear(DM, 256), nn.GELU(), nn.Linear(256, DM))

    def forward(s, S, m=None):
        a, _ = s.attn(s.q0.expand(S.shape[0], -1, -1), S.float(), S.float(),
                      key_padding_mask=m, need_weights=False)
        return F.normalize(s.head(a.mean(1)), dim=-1)


def rown(x, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    s = x.std(-1, keepdims=True)
    return (x - m) / (s + eps)


def main():
    args = parse_args()
    import sys
    sys.path.insert(0, args.repo)
    from VICReg_review.text_variant_eval import train_anchor_ridge, make_or_load_split, micro_prf

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fivefold_{args.recipe}_fold{args.fold}.json"
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path.name} exists")
        return

    dev = torch.device("cuda")
    C = Path(args.data_dir)
    G = np.load(C / "games.npz", allow_pickle=True)
    A = np.load(C / "articles.npz", allow_pickle=True)
    Qs = np.load(C / "ss_queries.npz", allow_pickle=True)
    GAL = np.load(C / "wscan_gal.npz")
    VALQ = np.load(C / "wscan_val.npz")
    pool = np.load(C / "wscan_pool.npy", mmap_mode="r")
    y = np.load(C / "tag_labels.npz", allow_pickle=True)["y"]

    names = [str(x) for x in G["names"]]
    NG = len(names)
    n2i = {n: i for i, n in enumerate(names)}
    SGal = torch.tensor(GAL["gal"]).to(dev)
    SA = rown(torch.tensor(A["S"]).to(dev).float()).half()
    mA = torch.arange(SA.shape[1], device=dev)[None, :] >= torch.tensor(A["S_len"]).to(dev)[:, None]
    variants = [str(x) for x in A["variants"]]
    art_games = [str(x) for x in A["names"]]
    SV = torch.tensor(VALQ["S"]).to(dev)
    mV = torch.arange(SV.shape[1], device=dev)[None, :] >= torch.tensor(VALQ["S_len"]).to(dev)[:, None]
    gV = torch.tensor(VALQ["gidx"]).to(dev)
    gA_t = torch.tensor(A["gidx"]).to(dev)
    gQ_t = torch.tensor(Qs["gidx"]).to(dev)

    # ---- fold split over the 258 article games (each game tested exactly once) ----
    ug = sorted(set(art_games))
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(ug))
    folds = np.array_split(perm, args.n_folds)
    test_g = {ug[i] for i in folds[args.fold]}
    rest = [ug[i] for i in perm if ug[i] not in test_g]   # permutation order, protocol-style
    n_val = max(1, int(round(len(rest) * VAL_FRAC)))
    val_g = set(rest[:n_val])
    train_g = set(rest[n_val:])
    train_pool_games = np.array([i for i in range(NG) if names[i] not in val_g])
    print(f"[fold {args.fold}/{args.n_folds}] test {len(test_g)} / val {len(val_g)} / "
          f"train {len(train_g)} article games", flush=True)

    # tag-ridge protocol objects (seed-42 split over the 2020 games; fold-independent)
    targs = SimpleNamespace(tag_text_train_frac=0.7, tag_text_val_frac=0.15,
                            tag_text_split_seed=42, seed=42, tag_text_threshold_steps=33)
    # per-job file: content is deterministic (seed 42) but concurrent workers must
    # not race on one path
    tag_split_path = out_dir / f"_tag_split_{args.recipe}_fold{args.fold}.json"
    tag_split = make_or_load_split(tag_split_path, names, targs)

    def gallery(model, chunk=256, grad=False):
        outs = []
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            for i in range(0, NG, chunk):
                outs.append(model(SGal[i:i+chunk]))
        return torch.cat(outs)

    def sample_views(gids, W, rng):
        sel = np.argsort(rng.random((len(gids), pool.shape[1])), axis=1)[:, :W]
        out = np.empty((len(gids), W, 1024), np.float16)
        for k, g in enumerate(gids):
            out[k] = pool[g, np.sort(sel[k])]
        return torch.from_numpy(out).to(dev, non_blocking=True)

    @torch.no_grad()
    def val_hit(model):
        Zg = gallery(model)
        Za = model(SV, mV)
        sim = Za.float() @ Zg.float().T
        return float(((sim > sim.gather(1, gV[:, None])).sum(1) + 1 == 1).float().mean())

    try:
        amp_cls = lambda: torch.amp.GradScaler("cuda"); amp_cls()
    except Exception:
        amp_cls = lambda: torch.cuda.amp.GradScaler()

    def train_tower(N, W, pull, epochs, patience, seed=0, bs=192, per_epoch=3072):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = SetPoolN(N).to(dev)
        logt = nn.Parameter(torch.tensor(np.log(1/0.07), dtype=torch.float32, device=dev))
        opt = torch.optim.AdamW(list(model.parameters()) + [logt], lr=5e-4, weight_decay=1e-4)
        amp = amp_cls()
        best, bs_, pat = -1.0, None, 0
        for ep in range(epochs):
            model.train()
            for _ in range(per_epoch // bs):
                gids = rng.choice(train_pool_games, bs, replace=False)
                S1 = sample_views(gids, W, rng)
                tgt = torch.tensor(gids).to(dev)
                with torch.amp.autocast("cuda"):
                    Z1 = model(S1)
                    Zg = gallery(model, grad=True)
                    loss = F.cross_entropy(Z1 @ Zg.T * logt.exp(), tgt)
                    if pull > 0:
                        Z2 = model(sample_views(gids, W, rng))
                        loss = loss + pull * (1 - (Z1 * Z2).sum(-1)).mean() \
                                    + F.cross_entropy(Z2 @ Zg.T * logt.exp(), tgt)
                opt.zero_grad(); amp.scale(loss).backward()
                amp.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                amp.step(opt); amp.update()
            if ep % 2 == 1:
                model.eval()
                h1 = val_hit(model)
                if h1 > best:
                    best, pat = h1, 0
                    bs_ = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    pat += 1
                if pat >= patience:
                    break
        model.load_state_dict(bs_); model.eval()
        return model, ep + 1

    # ---- tower (cached per fold so head-only reruns are cheap) ----
    feat_path = out_dir / f"tower_{args.recipe}_fold{args.fold}.npz"
    if feat_path.exists() and not args.overwrite:
        T = np.load(feat_path)
        SPg, SPa, SPq = T["SPg"], T["SPa"], T["SPq"]
        tower_ep = int(T["ep"])
        print(f"[tower] reusing {feat_path.name} (ep{tower_ep})", flush=True)
    else:
        t0 = time.time()
        model, tower_ep = train_tower(seed=0, **RECIPES[args.recipe])
        SQs = rown(torch.tensor(Qs["S"]).to(dev).float()).half()
        mQs = torch.arange(SQs.shape[1], device=dev)[None, :] >= torch.tensor(Qs["S_len"]).to(dev)[:, None]
        with torch.no_grad():
            SPg = gallery(model).float().cpu().numpy()
            SPa = model(SA, mA).float().cpu().numpy()
            SPq = torch.cat([model(SQs[i:i+2048], mQs[i:i+2048]).float()
                             for i in range(0, SQs.shape[0], 2048)]).cpu().numpy()
        del SQs, mQs
        tmp = feat_path.with_suffix(".tmp.npz")
        np.savez(tmp, SPg=SPg, SPa=SPa, SPq=SPq, ep=tower_ep)
        tmp.replace(feat_path)
        print(f"[tower] trained ep{tower_ep} in {time.time()-t0:.0f}s", flush=True)

    # ---- heads: userft objective + val-MRR selection ----
    va_neu = [i for i, g in enumerate(art_games) if g in val_g and variants[i] == "neutral"]
    tr_neu = [i for i, g in enumerate(art_games) if g in train_g and variants[i] == "neutral"]
    vgi = {n2i[g] for g in val_g}
    q_train = np.where(~np.isin(Qs["gidx"], list(vgi)))[0]

    mu, sd = SPg.mean(0, keepdims=True), SPg.std(0, keepdims=True) + 1e-6
    tt = lambda x: torch.tensor((x - mu) / sd, dtype=torch.float32).to(dev)
    Xg, Xa, Xq = tt(SPg), tt(SPa), tt(SPq)

    def full_metrics(gal, art):
        gz = gal / (np.linalg.norm(gal, axis=1, keepdims=True) + 1e-8)
        az = art / (np.linalg.norm(art, axis=1, keepdims=True) + 1e-8)
        out = {}
        for var in ("neutral", "noname"):
            ii = [i for i, g in enumerate(art_games) if g in test_g and variants[i] == var]
            sim = az[ii] @ gz.T
            tgt = A["gidx"][ii]
            rk = (sim > sim[np.arange(len(ii)), tgt][:, None]).sum(1) + 1
            out["h1_" + var] = float((rk == 1).mean())
            out["h5_" + var] = float((rk <= 5).mean())
        sc, rg, al, th, _ = train_anchor_ridge(targs, gal, y, n2i, tag_split)
        for var in ("neutral", "noname"):
            idx = [i for i in range(len(art_games)) if variants[i] == var and art_games[i] in test_g]
            s = rg.predict(sc.transform(np.stack([art[i] for i in idx]).astype(np.float32)))
            labs = np.stack([y[n2i[art_games[i]]] for i in idx])
            out["tag_" + var] = micro_prf(labs, s, th)["micro_f1"]
        return out

    def train_head(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        head = nn.Linear(Xg.shape[1], 128).to(dev)
        fwd = lambda x: F.normalize(head(x), dim=-1)
        logt = nn.Parameter(torch.tensor(np.log(1/0.07), dtype=torch.float32, device=dev))
        opt = torch.optim.AdamW(list(head.parameters()) + [logt], lr=1e-3, weight_decay=1e-4)

        def vmrr():
            with torch.no_grad():
                sim = fwd(Xa[va_neu]) @ fwd(Xg).T
                rk = (sim > sim.gather(1, gA_t[va_neu][:, None])).sum(1) + 1
                return float((1.0 / rk.float()).mean())

        best, bs_, pat = -1.0, None, 0
        for ep in range(80):  # phase 1: self-sup InfoNCE on pseudo queries
            head.train()
            order = np.random.choice(q_train, 6144, replace=False)
            for k in range(0, 6144, 256):
                b = torch.tensor(order[k:k+256]).to(dev)
                loss = F.cross_entropy(fwd(Xq[b]) @ fwd(Xg).T * logt.exp(), gQ_t[b])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0); opt.step()
                with torch.no_grad(): logt.clamp_(max=float(np.log(100.0)))
            if ep % 5 == 4:
                m = vmrr()
                if m > best:
                    best, pat = m, 0
                    bs_ = {k2: v.detach().clone() for k2, v in head.state_dict().items()}
                else:
                    pat += 1
                if pat >= 8:
                    break
        head.load_state_dict(bs_)

        opt = torch.optim.AdamW(list(head.parameters()) + [logt], lr=1e-4, weight_decay=1e-4)
        best, pat = vmrr(), 0
        bs_ = {k2: v.detach().clone() for k2, v in head.state_dict().items()}
        for ep in range(args.head_epochs):  # phase 2: user objective on real neutrals
            head.train()
            order = np.random.permutation(tr_neu)
            for k in range(0, len(order), 128):
                b = torch.tensor(order[k:k+128]).to(dev)
                z = fwd(Xa[b]); Zg = fwd(Xg); tgt = gA_t[b]
                la = z @ Zg.T
                lt = z @ z.T - torch.eye(z.shape[0], device=dev) * 1e4
                loss = F.cross_entropy(torch.cat([la, lt], 1) * logt.exp(), tgt)
                loss = loss + (1 - (z * Zg[tgt]).sum(-1)).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0); opt.step()
                with torch.no_grad(): logt.clamp_(max=float(np.log(100.0)))
            if ep % 5 == 4:
                m = vmrr()
                if m > best:
                    best, pat = m, 0
                    bs_ = {k2: v.detach().clone() for k2, v in head.state_dict().items()}
                else:
                    pat += 1
                if pat >= 24:
                    break
        head.load_state_dict(bs_); head.eval()
        with torch.no_grad():
            return fwd(Xg).cpu().numpy(), fwd(Xa).cpu().numpy()

    runs = []
    for seed in range(args.seeds):
        t0 = time.time()
        gal, art = train_head(seed)
        m = full_metrics(gal, art)
        runs.append(m)
        print(f"[head] fold{args.fold} {args.recipe} seed{seed}: "
              f"h1={m['h1_neutral']:.3f} h5={m['h5_neutral']:.3f} "
              f"nn={m['h1_noname']:.3f}/{m['h5_noname']:.3f} "
              f"tag={m['tag_neutral']:.3f}/{m['tag_noname']:.3f} [{time.time()-t0:.0f}s]",
              flush=True)

    result = {
        "recipe": args.recipe, "fold": args.fold, "n_folds": args.n_folds,
        "tower_ep": tower_ep, "split_seed": SPLIT_SEED,
        "test_games": sorted(test_g), "n_val": len(val_g), "n_train": len(train_g),
        "seeds": runs,
        "agg": {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
                for k in runs[0]},
    }
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out_path)
    print(f"[done] {out_path.name} agg:",
          {k: f"{v[0]:.3f}+-{v[1]:.3f}" for k, v in result["agg"].items()}, flush=True)


if __name__ == "__main__":
    main()
