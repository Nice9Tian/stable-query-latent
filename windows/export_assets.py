# -*- coding: utf-8 -*-
"""export_assets — turn a trained champion run into the GUI's asset pack.

Run this in the TRAINING environment (cuda_Vit conda env), with the release
data assets built (dataset_builder/rebuild_data.py) and a champion tower
checkpoint present in data/results (produced by
steam_reviews_framework/train_champion.py — arm name champion_cegate2).

    C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe windows/export_assets.py \
        [--ckpt data/results/ckpt_champion_cegate2_ep600.pt] [--seeds 3]

Writes into windows/assets/:
    tower.pt               champion tower state_dict (as trained)
    champion_assets.npz    head W/b/logt, gallery feats (tower + head space),
                           feature mu/sd, ridge tag probe, names, tag names
    meta.json              provenance + the vsel/test metrics of the export
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WINDOWS_DIR = Path(__file__).resolve().parent
RELEASE = WINDOWS_DIR.parent / "release"
sys.path.insert(0, str(RELEASE))

from dataset_builder.paths import ASSETS, RESULTS                    # noqa: E402
from main_model import LariceConfig, LariceTower                     # noqa: E402
from steam_reviews_framework.backhead_name import train_backhead_name  # noqa: E402
from steam_reviews_framework.backhead_tag import train_anchor_ridge  # noqa: E402
from steam_reviews_framework.data import load_bundle                 # noqa: E402
from steam_reviews_framework.eval import metrics4                    # noqa: E402
from steam_reviews_framework.train import CHAMPION, make_doc_tiers, project_cache  # noqa: E402

ARM = "champion_cegate2"


def pick_checkpoint(results: Path):
    """Prefer the post-hoc vsel-picked epoch, else the newest checkpoint."""
    best = results / f"ft4var_{ARM}_best.json"
    if best.exists():
        ek = json.loads(best.read_text())["best_ep"]
        p = results / f"ckpt_{ARM}_ep{ek}.pt"
        if p.exists():
            return p
    cands = sorted(results.glob(f"ckpt_{ARM}_ep*.pt"),
                   key=lambda p: int(p.stem.split("_ep")[-1]))
    if not cands:
        raise SystemExit(f"no {ARM} checkpoint in {results}; "
                         "run steam_reviews_framework/train_champion.py first")
    return cands[-1]


class _HeadCapture:
    """Wrap train_backhead_name and capture the linear head weights + logt.

    The release function keeps the head internal; rather than fork its logic
    we re-run it while intercepting nn.Linear construction (the head) and
    AdamW construction (whose param list carries the lone scalar logt).
    Patching nn.Parameter itself is NOT safe: torch lazy-imports _dynamo,
    which subclasses nn.Parameter (metaclass conflict with a function).
    """

    def __init__(self):
        self.head = None
        self.logt_param = None

    def run(self, B, Xg, Xg_nd, Xa, Xq, Xd, d_pos, seed):
        # prewarm torch's lazy _dynamo import OUTSIDE the patch window
        torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=1.0)

        orig_linear = nn.Linear
        orig_adamw = torch.optim.AdamW
        cap = self

        class SpyLinear(orig_linear):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                cap.head = self

        class SpyAdamW(orig_adamw):
            def __init__(self, params, *a, **k):
                params = list(params)
                head_ids = ({id(p) for p in cap.head.parameters()}
                            if cap.head is not None else set())
                for p in params:
                    if torch.is_tensor(p) and p.numel() == 1 \
                            and id(p) not in head_ids:
                        cap.logt_param = p
                super().__init__(params, *a, **k)

        nn.Linear = SpyLinear
        torch.optim.AdamW = SpyAdamW
        try:
            gal, art, vs = train_backhead_name(
                B, Xg, Xg_nd, Xa, Xq, Xd, d_pos, seed,
                p1="ice", p2="ice", ls=0.0, iw=CHAMPION.hiw)
        finally:
            nn.Linear = orig_linear
            torch.optim.AdamW = orig_adamw
        assert cap.head is not None and cap.logt_param is not None, \
            "head capture failed"
        return gal, art, vs

    @property
    def logt(self):
        return float(self.logt_param.detach().cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="tower checkpoint .pt "
                    "(default: vsel-picked / newest champion ckpt in results)")
    ap.add_argument("--proj", default="", help="existing tower projection npz "
                    "(SPg/SPg_nd/SPa/SPq/SPd/SPd_gidx); skips recompute")
    ap.add_argument("--seeds", type=int, default=3,
                    help="head seeds to try; the best-vsel one is exported")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", default=str(WINDOWS_DIR / "assets"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = Path(RESULTS)
    ckpt = Path(args.ckpt) if args.ckpt else pick_checkpoint(results)
    print(f"checkpoint : {ckpt}")

    dev = torch.device(args.device)
    B = load_bundle(dev)
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # ---- tower projections (reuse the training run's cache if present) ----
    ek = ckpt.stem.split("_ep")[-1]
    proj = Path(args.proj) if args.proj else results / f"tower_{ARM}_ep{ek}.npz"
    if not proj.exists():
        print("projection cache missing -- computing ...")
        model = LariceTower(LariceConfig(readout="pool")).to(dev)
        model.load_state_dict({k: v.to(dev) for k, v in sd.items()})
        model.eval()
        tiers, g2wiki, _ = make_doc_tiers(B, CHAMPION)
        project_cache(model, B, (g2wiki,) + tuple(tiers[0][1:]),
                      out / f"_proj_ep{ek}.npz")
        proj = out / f"_proj_ep{ek}.npz"
    T = np.load(proj)

    g0 = T["SPg"]
    mu = g0.mean(0, keepdims=True)
    sdv = g0.std(0, keepdims=True) + 1e-6
    tt = lambda x: torch.tensor((x - mu) / sdv, dtype=torch.float32).to(dev)
    Xg, Xa, Xq, Xd = tt(g0), tt(T["SPa"]), tt(T["SPq"]), tt(T["SPd"])
    Xg_nd = tt(T["SPg_nd"])
    d_pos = B.pos_of_g_t[T["SPd_gidx"]].to(dev)

    # ---- head: try seeds, keep the best vsel ----
    best = None
    for seed in range(args.seeds):
        capture = _HeadCapture()
        gal, art, vs = capture.run(B, Xg, Xg_nd, Xa, Xq, Xd, d_pos, seed)
        print(f"seed {seed}: vsel {vs['vscore']:.4f} "
              f"(neu {vs['v_neu']:.3f} non {vs['v_non']:.3f})")
        if best is None or vs["vscore"] > best["vs"]["vscore"]:
            best = {"seed": seed, "gal": gal, "art": art, "vs": vs,
                    "W": capture.head.weight.detach().cpu().numpy(),
                    "b": capture.head.bias.detach().cpu().numpy(),
                    "logt": capture.logt}
    print(f"exporting seed {best['seed']}")

    # ---- test metrics of the exported head (provenance) ----
    m4 = metrics4(B, best["gal"], best["art"])

    # ---- tag ridge probe on the exported head's gallery feats ----
    sc, rg, alpha, thr, val_m = train_anchor_ridge(
        B.targs, best["gal"], B.y, B.n2i, B.tag_split)
    tagf = ASSETS / "tag_labels.npz"
    tl = np.load(tagf, allow_pickle=True)
    tag_names = None
    for key in ("tags", "tag_names", "columns", "names"):
        if key in tl and len(tl[key]) == B.y.shape[1]:
            tag_names = [str(t) for t in tl[key]]
            break
    if tag_names is None:
        tag_names = [f"tag_{i}" for i in range(B.y.shape[1])]

    # ---- human-readable titles (text_h5 game_titles), fallback = name ----
    import h5py
    from dataset_builder.paths import TEXT_H5
    titles = {}
    try:
        with h5py.File(TEXT_H5, "r") as h:
            hn = [g.decode() if isinstance(g, bytes) else str(g)
                  for g in h["game_names"][:]]
            ht = [t.decode() if isinstance(t, bytes) else str(t)
                  for t in h["game_titles"][:]]
            titles = dict(zip(hn, ht))
    except Exception as e:
        print(f"NOTE: no titles from text_h5 ({e}); falling back to names")
    display = [titles.get(n) or n for n in B.names]

    # ---- write the pack ----
    torch.save(sd, out / "tower.pt")
    np.savez(out / "champion_assets.npz",
             head_W=best["W"], head_b=best["b"],
             head_logt=np.float32(best["logt"]),
             feat_mu=mu[0], feat_sd=sdv[0] - 1e-6,
             gallery_head=best["gal"], gallery_tower=g0,
             names=np.array(B.names, dtype=object),
             display_names=np.array(display, dtype=object),
             tag_coef=rg.coef_, tag_intercept=rg.intercept_,
             tag_scaler_mean=sc.mean_, tag_scaler_scale=sc.scale_,
             tag_threshold=np.float32(thr),
             tag_names=np.array(tag_names, dtype=object))
    meta = {"arm": ARM, "checkpoint": str(ckpt), "epoch": int(ek),
            "head_seed": best["seed"], "vsel": best["vs"],
            "test_metrics4": m4, "tag_alpha": alpha,
            "tag_threshold": thr, "tag_val": val_m,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2,
                                              ensure_ascii=False),
                                   encoding="utf-8")
    print(f"assets written to {out}")
    print(json.dumps({k: m4[k]["h1"] for k in m4}, indent=2))


if __name__ == "__main__":
    main()
