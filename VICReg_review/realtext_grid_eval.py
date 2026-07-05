"""Real-text tag + retrieval over the FULL FULL-N pool (all done combos).

Champion selection ran on review-domain battery metrics only; this evaluates
EVERY done FULL-N combo on the 258-game generated article set so selection can
be revisited with real-text evidence (see get_champions_namerank.py):

  per combo -> <combo>/real_text_grid.json
      tag  : zero-shot article micro-F1 per variant + mean drop vs anchor
      name : article -> own-game retrieval (game-level max-sim over view rows),
             hit@1/5/10 + median rank, neutral and noname (258 queries)
      con  : contrastive fine-tune (frozen backbone, residual linear head,
             full-softmax InfoNCE, identity init, early stop on val hit@1;
             181/77 game split, seed 20260705) -- zero-shot vs fine-tuned
             retrieval on the SAME held-out 77 games / mean-anchor gallery
  after every combo the streaming summary <out_dir>/realtext_grid_metrics.json
  is rebuilt from all per-combo files (resume = skip existing output).

Multi-GPU on one machine = manual shards (multi-VM claims deliberately absent):

    python VICReg_review/realtext_grid_eval.py                  # single GPU
    CUDA_VISIBLE_DEVICES=1 python ... --shard 1/4               # per-GPU shards
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANTS = ("neutral", "positive", "negative", "noname")
OUT_NAME = "real_text_grid.json"
SUMMARY = "realtext_grid_metrics.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", default=str(ROOT / "game_review_data/embedding_h5.h5"))
    p.add_argument("--out-dir", default=str(ROOT / "VICReg_review/heads/cloud_full_sweep_a100"))
    p.add_argument("--variant-dir", default=str(ROOT / "VICReg_review/text_variants_generated"))
    p.add_argument("--cache", default=None,
                   help="article sentence-embedding cache npz; default "
                        "<out-dir>/text_variant_pilot_cache.npz (built on first run)")
    p.add_argument("--shard", default="0/1", help="i/n: evaluate every n-th combo")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-contrastive", action="store_true",
                   help="zero-shot metrics only (faster)")
    p.add_argument("--con-epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260705)
    p.add_argument("--selfstop", action="store_true",
                   help="stop this RunPod pod once EVERY ready combo has output "
                        "(the last shard to finish triggers it; same ladder as "
                        "the auto notebooks)")
    p.add_argument("--runpod-api-key", default="",
                   help="override for pod_selfstop (else env RUNPOD_API_KEY)")
    p.add_argument("--local-data-dir", default="/root/data",
                   help="machine-local dir holding the resident vectors .dat "
                        "(same as training/battery); used only if a combo's "
                        "eval feature cache is missing and features must be "
                        "re-extracted from bulk vectors")
    args = p.parse_args()

    if args.selfstop:
        from VICReg_review import pod_selfstop
        pod_id, api_key, ctl = pod_selfstop.preflight(args.runpod_api_key)
        if not pod_id or not api_key:
            raise SystemExit("--selfstop requested but the pod cannot stop itself; "
                             "fix credentials first (see preflight output).")

    import h5py
    import numpy as np
    import torch

    from VICReg_review import disturbtion_embed
    from VICReg_review import eval_battery_worker as worker
    from VICReg_review import run_data_view_sweep as sweep
    from VICReg_review import text_variant_eval as tve
    from VICReg_review.identity_diagnostic import l2_normalize

    out_root = Path(args.out_dir)
    variant_dir = Path(args.variant_dir)
    if not variant_dir.exists():
        raise SystemExit(f"variant dir missing: {variant_dir}")
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    ev = worker.build_eval_args(argparse.Namespace(h5=args.h5, out_dir=str(out_root)))
    ev.out_dir = out_root
    ev.text_variant_dir = variant_dir
    ev.text_variant_cache = Path(args.cache or out_root / "text_variant_pilot_cache.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(ev.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])

    # I/O ladder for the rare feature-cache miss: point the trainer module at
    # the machine-local resident .dat (H5 -> local NVMe .dat -> OS page cache)
    # exactly like the battery worker, so any extract_features fallback never
    # streams bulk vectors over the shared FS with the GPU idle.
    dat = worker.enable_resident_vectors(ev.h5, args.local_data_dir)
    if dat:
        print(f"resident .dat ON: {dat} (fallback feature extraction reads local NVMe)",
              flush=True)
    else:
        print(f"resident .dat not staged under {args.local_data_dir}; fine as long "
              "as every combo has its eval feature cache (a cache MISS would "
              "stream bulk vectors over the shared FS -- slow)", flush=True)

    # sentence-embedding cache is combo-independent: build/load once
    names_all, appids = tve.load_h5_names(Path(ev.h5))
    records = tve.discover_variant_records(variant_dir, names_all, appids)
    cache = tve.load_or_embed_variant_texts(ev, records, ev.text_variant_cache)

    def rebuild_summary():
        pool, _live, full_n = worker.full_pool(Path(ROOT / "VICReg_review/sweep/sweep.yaml"))
        rows = []
        for c in pool:
            rp = out_root / c.combo_id / OUT_NAME
            if rp.exists():
                try:
                    rows.append(json.loads(rp.read_text(encoding="utf-8")))
                except Exception:
                    pass
        tve.atomic_json_write(
            {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "full_n": full_n,
             "n_rows": len(rows), "rows": rows}, out_root / SUMMARY)
        return len(rows)

    pool, _live, _fn = worker.full_pool(Path(ROOT / "VICReg_review/sweep/sweep.yaml"))
    todo = [c for i, c in enumerate(pool) if i % shard_n == shard_i]
    print(f"shard {args.shard}: {len(todo)}/{len(pool)} combos in scope", flush=True)

    done = skip = fail = 0
    for combo in todo:
        cdir = out_root / combo.combo_id
        outp = cdir / OUT_NAME
        ready, ckpt = worker.combo_ready(cdir)
        if not ready:
            continue
        if outp.exists() and not args.overwrite:
            skip += 1
            continue
        try:
            feats, names = sweep.build_vicreg_feature_cache(ev, ckpt, cdir)
            feats = np.asarray(feats)
            anchor = feats.mean(axis=1).astype(np.float32)
            n2i = {n: i for i, n in enumerate(names)}
            y, _tags = tve.align_labels(Path(ev.h5), names)
            split = tve.make_or_load_split(
                Path(getattr(ev, "tag_text_split_json", "") or out_root / "tag_text_eval_split.json"),
                names, ev)
            a_scaler, a_ridge, _al, a_th, _vm = tve.train_anchor_ridge(ev, anchor, y, n2i, split)
            encoder, _, _, _ = sweep.load_frozen_encoder(ckpt, input_dim, device)
            vfeat = tve.encode_variant_features(ev, encoder, cache, device)
            games = sorted({n for (n, v) in vfeat
                            if n in n2i and all((n, w) in vfeat for w in VARIANTS)})

            # tag: zero-shot per variant + anchor subset
            ya = np.stack([y[n2i[g]] for g in games])
            Xa = np.stack([anchor[n2i[g]] for g in games])
            anchor_sub = tve.micro_prf(ya, a_ridge.predict(a_scaler.transform(Xa)), a_th)
            tag = {}
            drops = []
            for v in VARIANTS:
                Xv = np.stack([vfeat[(g, v)] for g in games]).astype(np.float32)
                m = tve.micro_prf(ya, a_ridge.predict(a_scaler.transform(Xv)), a_th)
                tag[v] = m
                drops.append(anchor_sub["micro_f1"] - m["micro_f1"])

            # retrieval: game-level max-sim over VIEW rows (2020-game gallery)
            flat = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
            gal = l2_normalize(flat)
            n_views = feats.shape[1]
            def ret(variant):
                ranks = []
                for g in games:
                    q = np.asarray(vfeat[(g, variant)], dtype=np.float32)
                    q = q / (np.linalg.norm(q) or 1.0)
                    sims = (gal @ q).reshape(-1, n_views).max(axis=1)
                    ranks.append(int((sims > sims[n2i[g]]).sum()) + 1)
                r = np.asarray(ranks)
                return {"hit_at_1": float((r == 1).mean()),
                        "hit_at_5": float((r <= 5).mean()),
                        "hit_at_10": float((r <= 10).mean()),
                        "median_rank": float(np.median(r))}

            payload = {
                "combo_id": combo.combo_id, "arm": combo.arm,
                "output_dim": combo.output_dim, "num_latents": combo.num_latents,
                "view": combo.view, "n_games": len(games),
                "anchor_subset_f1": anchor_sub["micro_f1"],
                "tag": tag, "mean_drop": float(np.mean(drops)),
                "retrieval": {v: ret(v) for v in ("neutral", "noname")},
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            # ---- contrastive fine-tune on the frozen backbone (linear head)
            if not args.skip_contrastive:
                import torch.nn as nn
                torch.manual_seed(args.seed)
                rng = np.random.default_rng(args.seed)
                order = rng.permutation(len(games))
                n_tr = int(round(len(games) * 0.7))
                tr_games = [games[i] for i in order[:n_tr]]
                te_games = [games[i] for i in order[n_tr:]]
                n_fit = int(round(len(tr_games) * 0.85))
                fit_g, val_g = tr_games[:n_fit], tr_games[n_fit:]

                A = torch.tensor(anchor, device=device)
                A = A / A.norm(dim=1, keepdim=True)

                def qt(gs, vs=VARIANTS):
                    X = np.stack([vfeat[(g, v)] for g in gs for v in vs]).astype(np.float32)
                    lab = torch.tensor([n2i[g] for g in gs for _ in vs], device=device)
                    return torch.tensor(X, device=device), lab

                def hits_t(sims, labels):
                    rk = (sims > sims.gather(1, labels[:, None])).sum(1) + 1
                    return {"hit_at_1": float((rk == 1).float().mean()),
                            "hit_at_10": float((rk <= 10).float().mean()),
                            "median_rank": float(rk.float().median())}

                Xf, yf = qt(fit_g)
                Xv, yv = qt(val_g)
                Xt, yt = qt(te_games, ("neutral",))

                class Head(nn.Module):
                    def __init__(self, dim):
                        super().__init__()
                        self.f = nn.Linear(dim, dim, bias=False)
                        self.scale = nn.Parameter(torch.zeros(()))
                        self.log_tau = nn.Parameter(torch.tensor(-2.65))

                    def forward(self, q):
                        return q + self.scale * self.f(q)

                head = Head(A.shape[1]).to(device)
                opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
                best_state, best_val, patience = None, -1.0, 0
                for _ep in range(args.con_epochs):
                    head.train()
                    perm = torch.randperm(Xf.shape[0], device=device)
                    for i in range(0, len(perm), 256):
                        idx = perm[i:i + 256]
                        q = head(Xf[idx])
                        q = q / q.norm(dim=1, keepdim=True)
                        loss = nn.functional.cross_entropy(
                            (q @ A.T) / head.log_tau.exp(), yf[idx])
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    head.eval()
                    with torch.no_grad():
                        qv = head(Xv)
                        qv = qv / qv.norm(dim=1, keepdim=True)
                        v1 = hits_t(qv @ A.T, yv)["hit_at_1"]
                    if v1 > best_val:
                        best_val, patience = v1, 0
                        best_state = {k: v.clone() for k, v in head.state_dict().items()}
                    else:
                        patience += 1
                        if patience >= 30:
                            break
                head.load_state_dict(best_state)
                head.eval()
                with torch.no_grad():
                    q0 = Xt / Xt.norm(dim=1, keepdim=True)
                    qc = head(Xt)
                    qc = qc / qc.norm(dim=1, keepdim=True)
                    payload["contrastive"] = {
                        "n_train_games": len(tr_games), "n_test_games": len(te_games),
                        "zero_shot": hits_t(q0 @ A.T, yt),
                        "con_linear": hits_t(qc @ A.T, yt),
                    }
                del head, A, Xf, Xv, Xt
            tve.atomic_json_write(payload, outp)
            done += 1
            n = rebuild_summary()
            print(f"done {combo.combo_id} (done={done} skip={skip} fail={fail}; "
                  f"summary={n} rows)", flush=True)
            del feats, encoder, vfeat
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            fail += 1
            print(f"FAIL {combo.combo_id}: {type(e).__name__}: {e}", flush=True)
    rebuild_summary()
    print(f"shard {args.shard} exit: done={done} skip={skip} fail={fail}", flush=True)

    if args.selfstop:
        # complete = every READY combo of the FULL pool has an output file;
        # only the last shard to finish sees this true, so N shards need no
        # coordination -- earlier finishers just exit.
        missing = [c.combo_id for c in pool
                   if worker.combo_ready(out_root / c.combo_id)[0]
                   and not (out_root / c.combo_id / OUT_NAME).exists()]
        if missing:
            print(f"selfstop: {len(missing)} ready combos still lack output "
                  f"(other shards running?) -- NOT stopping. First: {missing[:3]}",
                  flush=True)
        else:
            print("selfstop: grid complete -- stopping this pod.", flush=True)
            from VICReg_review import pod_selfstop
            pod_selfstop.stop_pod(pod_id, api_key, ctl)


if __name__ == "__main__":
    main()
