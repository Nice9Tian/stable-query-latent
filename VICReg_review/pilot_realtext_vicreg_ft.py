"""VICReg fine-tune on the four rewrites: label-free tone/domain invariance.

The four same-game articles are treated as VIEWS of each other; a small head
on the frozen features is trained with the VICReg objective (invariance +
variance + covariance, 25/25/1). Two head shapes (per user spec): a residual
LINEAR map, and a residual 2-layer MLP bottleneck (shrink width, expand back).
Two view sets: the 4 articles only (pure tone invariance), or 4 articles + the
game's review anchor as a 5th view (label-free DOMAIN alignment). Evaluation
mirrors the contrastive protocol exactly: 5-fold CV over games, neutral
held-out queries vs the g-transformed mean-anchor gallery, plus a con_linear
(InfoNCE) reference trained under the same harness.

    python VICReg_review/pilot_realtext_vicreg_ft.py \
        --combos dim036_nogrl_n2000_view20 --out-dir <local out_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANTS = ("neutral", "positive", "negative", "noname")
OUT_NAME = "real_text_vicreg_ft.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", default=str(ROOT / "game_review_data/embedding_h5.h5"))
    p.add_argument("--out-dir", default=str(ROOT / "VICReg_review/heads/cloud_full_sweep_a100"))
    p.add_argument("--cache", default=None)
    p.add_argument("--combos", default="dim036_nogrl_n2000_view20")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260705)
    args = p.parse_args()

    import h5py
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from VICReg_review import disturbtion_embed
    from VICReg_review import eval_battery_worker as worker
    from VICReg_review import run_data_view_sweep as sweep
    from VICReg_review import text_variant_eval as tve

    torch.manual_seed(args.seed)
    out_root = Path(args.out_dir)
    cache = disturbtion_embed.load_npz_payload(
        Path(args.cache or out_root / "text_variant_pilot_cache.npz"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = worker.build_eval_args(argparse.Namespace(h5=args.h5, out_dir=str(out_root)))
    ev.out_dir = out_root
    with h5py.File(ev.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])

    class Residual(nn.Module):
        def __init__(self, dim, hidden=0):
            super().__init__()
            self.f = (nn.Linear(dim, dim, bias=False) if not hidden else
                      nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, dim)))
            self.scale = nn.Parameter(torch.zeros(()))

        def forward(self, q):
            return q + self.scale * self.f(q)

    def vicreg_loss(z1, z2, w_inv=25.0, w_var=25.0, w_cov=1.0):
        inv = F.mse_loss(z1, z2)
        def var_t(z):
            return F.relu(1.0 - torch.sqrt(z.var(dim=0) + 1e-4)).mean()
        def cov_t(z):
            zc = z - z.mean(dim=0)
            c = (zc.T @ zc) / max(1, z.shape[0] - 1)
            off = c - torch.diag(torch.diag(c))
            return (off ** 2).sum() / z.shape[1]
        return (w_inv * inv + w_var * (var_t(z1) + var_t(z2)) / 2
                + w_cov * (cov_t(z1) + cov_t(z2)) / 2)

    def hits(sims, labels):
        rk = (sims > sims.gather(1, labels[:, None])).sum(1) + 1
        return rk

    def agg(all_ranks):
        r = torch.cat(all_ranks).float()
        return {"hit_at_1": float((r == 1).float().mean()),
                "hit_at_10": float((r <= 10).float().mean()),
                "median_rank": float(r.median())}

    for cid in [c.strip() for c in args.combos.split(",") if c.strip()]:
        cdir = out_root / cid
        ready, ckpt = worker.combo_ready(cdir)
        if not ready:
            print(f"{cid}: NOT ready -- skipped")
            continue
        feats, names = sweep.build_vicreg_feature_cache(ev, ckpt, cdir)
        anchor = np.asarray(feats).mean(axis=1).astype(np.float32)
        n2i = {n: i for i, n in enumerate(names)}
        encoder, _, _, _ = sweep.load_frozen_encoder(ckpt, input_dim, device)
        vfeat = tve.encode_variant_features(ev, encoder, cache, device)
        del feats, encoder

        games = sorted({n for (n, v) in vfeat
                        if n in n2i and all((n, w) in vfeat for w in VARIANTS)})
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(games))
        folds = np.array_split(order, 5)

        # standardize inputs on the whole anchor set (fixed, label-free)
        mu = anchor.mean(axis=0, keepdims=True)
        sd = anchor.std(axis=0, keepdims=True) + 1e-6
        A_std = torch.tensor((anchor - mu) / sd, device=device)

        def art(g, v):
            return (np.asarray(vfeat[(g, v)], dtype=np.float32) - mu[0]) / sd[0]

        def eval_head(head, te_games):
            with torch.no_grad():
                gal = head(A_std) if head is not None else A_std
                gal = gal / gal.norm(dim=1, keepdim=True)
                q = torch.tensor(np.stack([art(g, "neutral") for g in te_games]),
                                 device=device)
                q = head(q) if head is not None else q
                q = q / q.norm(dim=1, keepdim=True)
                lab = torch.tensor([n2i[g] for g in te_games], device=device)
                return hits(q @ gal.T, lab)

        def train_vicreg(fit_g, val_g, hidden, with_anchor):
            views = list(VARIANTS) + (["__anchor__"] if with_anchor else [])
            X = {v: torch.tensor(np.stack(
                    [art(g, v) for g in fit_g] if v != "__anchor__" else
                    [(anchor[n2i[g]] - mu[0]) / sd[0] for g in fit_g]),
                    device=device) for v in views}
            head = Residual(A_std.shape[1], hidden).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
            best_state, best_val, patience = None, -1.0, 0
            g_rng = np.random.default_rng(args.seed + 7)
            for _ep in range(args.epochs):
                head.train()
                for _step in range(4):
                    v1, v2 = g_rng.choice(len(views), size=2, replace=False)
                    z1 = head(X[views[v1]])
                    z2 = head(X[views[v2]])
                    loss = vicreg_loss(z1, z2)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                head.eval()
                rk = eval_head(head, val_g)
                v1s = float((rk == 1).float().mean())
                if v1s > best_val:
                    best_val, patience = v1s, 0
                    best_state = {k: v.clone() for k, v in head.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 30:
                        break
            head.load_state_dict(best_state)
            head.eval()
            return head

        def train_con(fit_g, val_g):
            A_n = A_std / A_std.norm(dim=1, keepdim=True)
            Xf = torch.tensor(np.stack([art(g, v) for g in fit_g for v in VARIANTS]),
                              device=device)
            yf = torch.tensor([n2i[g] for g in fit_g for _ in VARIANTS], device=device)
            head = Residual(A_std.shape[1], 0).to(device)
            log_tau = nn.Parameter(torch.tensor(-2.65, device=device))
            opt = torch.optim.AdamW(list(head.parameters()) + [log_tau],
                                    lr=1e-3, weight_decay=1e-4)
            best_state, best_val, patience = None, -1.0, 0
            for _ep in range(args.epochs):
                head.train()
                perm = torch.randperm(Xf.shape[0], device=device)
                for i in range(0, len(perm), 256):
                    idx = perm[i:i + 256]
                    q = head(Xf[idx])
                    q = q / q.norm(dim=1, keepdim=True)
                    loss = F.cross_entropy((q @ A_n.T) / log_tau.exp(), yf[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                head.eval()
                rk = eval_head(head, val_g)
                v1s = float((rk == 1).float().mean())
                if v1s > best_val:
                    best_val, patience = v1s, 0
                    best_state = {k: v.clone() for k, v in head.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 30:
                        break
            head.load_state_dict(best_state)
            head.eval()
            return head

        dim = A_std.shape[1]
        CONFIGS = {
            "zero_shot": None,
            "con_linear": ("con", 0, False),
            "vicreg_linear_art": ("vic", 0, False),
            "vicreg_mlp_art": ("vic", dim // 4, False),
            "vicreg_linear_anchor": ("vic", 0, True),
            "vicreg_mlp_anchor": ("vic", dim // 4, True),
        }
        results = {k: [] for k in CONFIGS}
        for k in range(5):
            te = [games[i] for i in folds[k]]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != k])
            tr = [games[i] for i in tr_idx]
            n_fit = int(round(len(tr) * 0.85))
            fit_g, val_g = tr[:n_fit], tr[n_fit:]
            for name, cfg in CONFIGS.items():
                if cfg is None:
                    results[name].append(eval_head(None, te))
                    continue
                kind, hidden, with_anchor = cfg
                head = (train_con(fit_g, val_g) if kind == "con"
                        else train_vicreg(fit_g, val_g, hidden, with_anchor))
                results[name].append(eval_head(head, te))
                del head
            print(f"  fold {k + 1}/5 done", flush=True)

        report = {"combo_id": cid, "hidden_mlp": dim // 4,
                  "results": {name: agg(rks) for name, rks in results.items()}}
        tve.atomic_json_write(report, cdir / OUT_NAME)
        print(f"\n{cid}  (5-fold CV, neutral queries, g-transformed gallery)")
        print(f"{'config':22} {'hit@1':>7} {'hit@10':>7} {'median':>7}")
        for name, m in report["results"].items():
            print(f"{name:22} {m['hit_at_1']:7.3f} {m['hit_at_10']:7.3f} "
                  f"{m['median_rank']:7.1f}")


if __name__ == "__main__":
    main()
