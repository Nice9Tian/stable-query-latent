"""Contrastive retrieval head on frozen features: can articles hit rank 1?

The MSE adapter optimizes coordinate closeness; retrieval needs ranking. This
trains a query-side head with InfoNCE (own game = positive, all other gallery
games = in-batch-free FULL softmax over 2020) and compares three heads on the
same fixed 77-game test split:

  mse_linear : the ridge adapter from pilot_realtext_finetune (baseline)
  con_linear : residual linear head, identity-initialized, InfoNCE
  con_mlp    : residual 2-layer MLP head, identity-initialized, InfoNCE

Both contrastive heads start exactly at the zero-shot solution (residual scale
0), so training can only move away from it if that helps val hit@1 (early
stop). Also prints the top-1 confusions so the series-sibling ceiling is
visible (a WWE 2K24 article ranking WWE 2K23 first is not a fixable error).

    python VICReg_review/pilot_realtext_contrastive.py --out-dir <local out_dir> \
        [--combos "cid1,cid2"] [--epochs 300]
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
OUT_NAME = "real_text_contrastive_pilot.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", default=str(ROOT / "game_review_data/embedding_h5.h5"))
    p.add_argument("--out-dir", default=str(ROOT / "VICReg_review/heads/cloud_full_sweep_a100"))
    p.add_argument("--cache", default=None)
    p.add_argument("--combos",
                   default="dim064_grl_n2000_view40_lat512x2,dim064_grl_n2000_view60_lat512x2")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=20260705)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--hidden", type=int, default=1024)
    args = p.parse_args()

    import h5py
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

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

    class ResidualHead(nn.Module):
        """q + scale*f(q): scale starts at 0 -> exact zero-shot at init."""
        def __init__(self, dim, hidden=0):
            super().__init__()
            self.f = (nn.Linear(dim, dim, bias=False) if not hidden else
                      nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, dim)))
            self.scale = nn.Parameter(torch.zeros(()))
            self.log_tau = nn.Parameter(torch.tensor(-2.65))   # tau ~ 0.07

        def forward(self, q):
            return q + self.scale * self.f(q)

    def hits(sims, labels):
        ranks = (sims > sims.gather(1, labels[:, None])).sum(1) + 1
        return {"hit_at_1": float((ranks == 1).float().mean()),
                "hit_at_5": float((ranks <= 5).float().mean()),
                "hit_at_10": float((ranks <= 10).float().mean()),
                "median_rank": float(ranks.float().median())}

    for cid in [c.strip() for c in args.combos.split(",") if c.strip()]:
        cdir = out_root / cid
        ready, ckpt = worker.combo_ready(cdir)
        if not ready:
            print(f"{cid}: NOT ready -- skipped")
            continue
        feats, names = sweep.build_vicreg_feature_cache(ev, ckpt, cdir)
        anchor = np.asarray(feats).mean(axis=1).astype(np.float32)
        name_to_index = {n: i for i, n in enumerate(names)}
        encoder, _, _, _ = sweep.load_frozen_encoder(ckpt, input_dim, device)
        vfeat = tve.encode_variant_features(ev, encoder, cache, device)
        del feats, encoder

        art_games = sorted({n for (n, v) in vfeat
                            if n in name_to_index and all((n, w) in vfeat for w in VARIANTS)})
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(art_games))
        n_tr = int(round(len(art_games) * args.train_frac))
        tr_games = [art_games[i] for i in order[:n_tr]]
        te_games = [art_games[i] for i in order[n_tr:]]
        n_fit = int(round(len(tr_games) * 0.85))
        fit_g, val_g = tr_games[:n_fit], tr_games[n_fit:]

        A = torch.tensor(anchor, device=device)
        A = A / A.norm(dim=1, keepdim=True)

        def qtensor(games, variants=VARIANTS):
            X = np.stack([vfeat[(g, v)] for g in games for v in variants]).astype(np.float32)
            lab = torch.tensor([name_to_index[g] for g in games for _ in variants],
                               device=device)
            return torch.tensor(X, device=device), lab

        Xf, yf = qtensor(fit_g)
        Xv, yv = qtensor(val_g)
        Xt, yt = qtensor(te_games, ("neutral",))
        Xt_all, yt_all = qtensor(te_games)

        def evaluate(head):
            with torch.no_grad():
                out = {}
                for tag, X, lab in (("neutral", Xt, yt), ("all_variants", Xt_all, yt_all)):
                    q = head(X) if head is not None else X
                    q = q / q.norm(dim=1, keepdim=True)
                    out[tag] = hits(q @ A.T, lab)
                return out

        report = {"combo_id": cid, "n_train": len(tr_games), "n_test": len(te_games),
                  "heads": {"zero_shot": evaluate(None)}}

        # -- baseline: MSE ridge adapter (same protocol as finetune pilot)
        sc = StandardScaler().fit(Xf.cpu().numpy())
        tgt = anchor[yf.cpu().numpy()]
        best = None
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            ad = Ridge(alpha=alpha).fit(sc.transform(Xf.cpu().numpy()), tgt)
            qv = torch.tensor(ad.predict(sc.transform(Xv.cpu().numpy())),
                              dtype=torch.float32, device=device)
            h1 = hits((qv / qv.norm(dim=1, keepdim=True)) @ A.T, yv)["hit_at_1"]
            if best is None or h1 > best[0]:
                best = (h1, ad)
        ad = best[1]
        qt = lambda X: torch.tensor(ad.predict(sc.transform(X.cpu().numpy())),
                                    dtype=torch.float32, device=device)
        report["heads"]["mse_linear"] = {
            tag: hits((q / q.norm(dim=1, keepdim=True)) @ A.T, lab)
            for tag, q, lab in (("neutral", qt(Xt), yt), ("all_variants", qt(Xt_all), yt_all))}

        # -- contrastive heads
        for tag_h, hidden in (("con_linear", 0), ("con_mlp", args.hidden)):
            head = ResidualHead(A.shape[1], hidden).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
            best_state, best_val, patience = None, -1.0, 0
            for epoch in range(args.epochs):
                head.train()
                perm = torch.randperm(Xf.shape[0], device=device)
                for i in range(0, len(perm), 256):
                    idx = perm[i:i + 256]
                    q = head(Xf[idx])
                    q = q / q.norm(dim=1, keepdim=True)
                    logits = (q @ A.T) / head.log_tau.exp()
                    loss = nn.functional.cross_entropy(logits, yf[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                head.eval()
                with torch.no_grad():
                    qv = head(Xv)
                    qv = qv / qv.norm(dim=1, keepdim=True)
                    v1 = hits(qv @ A.T, yv)["hit_at_1"]
                if v1 > best_val:
                    best_val, patience = v1, 0
                    best_state = {k: v.clone() for k, v in head.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 30:
                        break
            head.load_state_dict(best_state)
            head.eval()
            report["heads"][tag_h] = evaluate(head)

        tve.atomic_json_write(report, cdir / OUT_NAME)
        print(f"\n{cid}  (train {len(tr_games)} games, test {len(te_games)})")
        print(f"{'head':12} {'neu@1':>7} {'neu@5':>7} {'neu@10':>7} {'neu_med':>8} "
              f"{'all@1':>7} {'all_med':>8}")
        for h, m in report["heads"].items():
            print(f"{h:12} {m['neutral']['hit_at_1']:7.3f} {m['neutral']['hit_at_5']:7.3f} "
                  f"{m['neutral']['hit_at_10']:7.3f} {m['neutral']['median_rank']:8.1f} "
                  f"{m['all_variants']['hit_at_1']:7.3f} "
                  f"{m['all_variants']['median_rank']:8.1f}")

        # -- what does top-1 confuse? (neutral, best contrastive head)
        with torch.no_grad():
            q = head(Xt)
            q = q / q.norm(dim=1, keepdim=True)
            sims = q @ A.T
            top1 = sims.argmax(1)
        wrong = [(te_games[i], names[int(top1[i])]) for i in range(len(te_games))
                 if int(top1[i]) != int(yt[i])]
        print(f"  top-1 misses (con_mlp, neutral): {len(wrong)}/{len(te_games)}; first 8:")
        for truth, got in wrong[:8]:
            print(f"    wanted {truth:28} got {got}")


if __name__ == "__main__":
    main()
