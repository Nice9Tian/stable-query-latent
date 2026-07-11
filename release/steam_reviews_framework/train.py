# -*- coding: utf-8 -*-
"""Unified trainer: tower (full budget, checkpoint every 50 ep, NO online
early stopping) + per-checkpoint BackHead-NAME grid + post-hoc vsel pick.

The per-checkpoint fine-tune results ARE the learning curve and the optimum
finder: pass 1 fine-tunes every checkpoint with 3 seeds, pass 2 picks the
best checkpoint by mean vsel (never test) and tops it up to 10 seeds.
"""
import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from main_model import LariceConfig, LariceTower, invariance_loss

from .anchors import gallery, gallery_nodoc, gallery_train
from .backhead_name import train_backhead_name
from .data import load_views
from .eval import metrics4, zs_metrics, VORDER
from dataset_builder.paths import RESULTS
from .protocol import SPLIT_SEED
from .sampler import pad_flat, sample_views


@dataclass
class ArmSpec:
    """One experiment arm = tower recipe + gate + head grid."""
    name: str
    tower: str = "cegate"        # cegate | igate | rgate | plain | arc | byol
    inv_weight: float = 2.0      # I dose (0 = pure CE)
    gate_scope: str = "doc"      # gated towers: "doc" (any doc) | "wiki"
    wiki_src: str = "clean"      # doc family: wiki_clean | wiki_llm
    use_sp_view: bool = True     # sp_raw tier under the wiki tier
    use_doc_view: bool = True    # False = nodoc (pure review views)
    tau: float = 0.02            # frozen tower temperature
    num_views: int = 4           # NV (last view = doc view when available)
    epochs: int = 1000
    ckpt_every: int = 50
    # head grid rows: (suffix, phase1, phase2, label_smoothing)
    head_cfgs: tuple = (("", "ice", "ice", 0.0),)
    head_iw: float | None = None          # defaults to max(inv_weight, 1)

    @property
    def hiw(self) -> float:
        return self.head_iw if self.head_iw is not None else max(self.inv_weight, 1.0)


CHAMPION = ArmSpec(name="champion_cegate2", tower="cegate", inv_weight=2.0)


# ------------------------- doc-view machinery ------------------------------

def make_doc_tiers(B, spec: ArmSpec):
    """Priority-tiered doc views: wiki family first, sp_raw fills the rest.
    Held-out games are excluded from every tier (fully inductive)."""
    if not spec.use_doc_view:
        return [], {}, set()
    WK, SW, mW = load_views(f"wiki_{spec.wiki_src}_views.npz", B.dev)
    g2wiki = {int(WK["gidx"][i]): i for i in range(len(WK["gidx"]))
              if str(WK["names"][i]) not in B.excl}
    tiers = [(g2wiki, SW, mW)]
    g2store = {}
    if spec.use_sp_view:
        ST, SS, mS = load_views("sp_raw_views.npz", B.dev)
        g2store = {int(ST["gidx"][i]): i for i in range(len(ST["gidx"]))
                   if str(ST["names"][i]) not in B.excl
                   and int(ST["gidx"][i]) not in g2wiki}
        tiers.append((g2store, SS, mS))
    doc_games = set(g2wiki) | set(g2store)
    return tiers, g2wiki, doc_games


def gate_set(B, spec: ArmSpec, g2wiki, doc_games):
    """Games on which the gated loss term fires."""
    if spec.tower == "rgate":       # random coverage-matched control gate
        n = sum(1 for g in B.train_pool_games if int(g) in doc_games)
        rngG = np.random.default_rng(SPLIT_SEED + 7)
        return set(rngG.choice(B.train_pool_games, n, replace=False).tolist())
    return set(g2wiki) if spec.gate_scope == "wiki" else doc_games


def assemble_doc_view(model, B, tiers, gids, W, rng, bs, out_dim):
    """The NV-th view: the game's highest-tier doc view, falling back to a
    fresh review view for games with no document."""
    Z = torch.empty(bs, out_dim, device=B.dev, dtype=torch.float16)
    assigned = np.zeros(bs, bool)
    for g2x, Sx, mx in tiers:
        msk = np.array([(not a) and (g in g2x) for a, g in zip(assigned, gids)])
        if msk.any():
            rows = [g2x[g] for g in gids[msk]]
            Z[torch.tensor(msk).to(B.dev)] = model(Sx[rows], mx[rows]).half()
            assigned |= msk
    rest = ~assigned
    if rest.any():
        S, m = sample_views(B.pool, B.rev_tab, gids[rest], W, rng, B.dev)
        Z[torch.tensor(rest).to(B.dev)] = model(S, m).half()
    return Z


def _amp_scaler():
    try:
        s = torch.amp.GradScaler("cuda")
        return s
    except Exception:
        return torch.cuda.amp.GradScaler()


# --------------------------- tower training --------------------------------

def train_tower(B, spec: ArmSpec, seed=0, W=16, bs=192, per_epoch=3072,
                loss_hook=None, log_cb=print):
    """Full-budget tower training; returns (model, checkpoints dict).

    loss_hook(Zs, Zg, tgt, gate_idx) -> loss overrides the default champion
    objective (used by contrast arms like ArcFace)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = LariceConfig(readout="pool", num_views=spec.num_views, tau=spec.tau,
                      inv_weight=spec.inv_weight)
    model = LariceTower(cfg).to(B.dev)
    inv_t = 1.0 / spec.tau
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    amp = _amp_scaler()
    tiers, g2wiki, doc_games = make_doc_tiers(B, spec)
    gated = spec.tower in ("cegate", "igate", "rgate")
    gate_games = gate_set(B, spec, g2wiki, doc_games) if gated else set()
    NV = spec.num_views
    ckpts = {}
    t0 = time.time()
    for ep in range(spec.epochs):
        model.train()
        for _ in range(per_epoch // bs):
            gids = rng.choice(B.train_pool_games, bs, replace=False)
            tgt = B.pos_of_g_t[gids].to(B.dev)
            with torch.amp.autocast("cuda"):
                Zg = gallery_train(model, B)
                Zs = [model(*sample_views(B.pool, B.rev_tab, gids, W, rng, B.dev))
                      for _ in range(NV - 1)]
                Zs.append(assemble_doc_view(model, B, tiers, gids, W, rng, bs,
                                            cfg.out_dim))
                if loss_hook is not None:
                    hd = (torch.tensor(np.array([g in gate_games for g in gids])
                                       ).to(B.dev).nonzero(as_tuple=True)[0]
                          if gated else None)
                    loss = loss_hook(Zs, Zg, tgt, hd)
                elif spec.tower == "cegate" or spec.tower == "rgate":
                    hd = torch.tensor(np.array([g in gate_games for g in gids])
                                      ).to(B.dev).nonzero(as_tuple=True)[0]
                    loss = (sum(F.cross_entropy(
                        Z.float()[hd] @ Zg.T.float() * inv_t, tgt[hd])
                        for Z in Zs)
                        if len(hd) else torch.zeros((), device=B.dev))
                else:
                    loss = sum(F.cross_entropy(Z.float() @ Zg.T.float() * inv_t,
                                               tgt) for Z in Zs)
                if spec.inv_weight > 0:
                    if spec.tower == "igate":
                        hd = torch.tensor(np.array(
                            [g in gate_games for g in gids], dtype=np.float32)
                        ).to(B.dev)
                        from itertools import combinations
                        pairs = list(combinations(range(NV), 2))
                        loss = loss + spec.inv_weight * sum(
                            ((1 - (Zs[i].float() * Zs[j].float()).sum(-1)) * hd
                             ).sum() / hd.sum().clamp(min=1)
                            for i, j in pairs) / len(pairs)
                    else:
                        loss = loss + spec.inv_weight * invariance_loss(Zs)
            opt.zero_grad()
            amp.scale(loss).backward()
            amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            amp.step(opt)
            amp.update()
        if (ep + 1) % spec.ckpt_every == 0:
            ckpts[ep + 1] = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}
        if ep % 100 == 99:
            log_cb(f"  [{spec.name} ep{ep + 1}] {time.time() - t0:.0f}s")
    model.eval()
    return model, ckpts


# ----------------- projection + head grid + post-hoc pick ------------------

def project_cache(model, B, tiers_wiki, path):
    """Freeze the tower; project galleries, eval queries, pseudo-queries and
    the family docs into tower space for the head stage."""
    g2wiki, SW, mW = tiers_wiki
    d_gidx = np.array(sorted(g2wiki), np.int64)
    d_rows = [g2wiki[g] for g in sorted(g2wiki)]
    NQ = len(B.q_gidx)
    with torch.no_grad():
        SPg = gallery(model, B).float().cpu().numpy()
        SPg_nd = gallery_nodoc(model, B).float().cpu().numpy()
        SPa = torch.cat([model(B.SA[i:i + 256], B.mA[i:i + 256])
                         for i in range(0, B.SA.shape[0], 256)]
                        ).float().cpu().numpy()
        SPq = torch.cat([model(*pad_flat(B.QS_S, B.q_off,
                                         range(i, min(i + 64, NQ)), B.dev)
                               ).float()
                         for i in range(0, NQ, 64)]).cpu().numpy()
        SPd = torch.cat([model(SW[d_rows[i:i + 256]], mW[d_rows[i:i + 256]])
                         for i in range(0, len(d_rows), 256)]
                        ).float().cpu().numpy()
    np.savez(path, SPg=SPg, SPg_nd=SPg_nd, SPa=SPa, SPq=SPq,
             SPd=SPd, SPd_gidx=d_gidx)


def head_runs(B, path, seeds, p1, p2, ls, hiw, tag_, log_cb=print):
    T = np.load(path)
    g0 = T["SPg"]
    d_pos = B.pos_of_g_t[T["SPd_gidx"]].to(B.dev)
    assert int(d_pos.min()) >= 0
    mu, sd = g0.mean(0, keepdims=True), g0.std(0, keepdims=True) + 1e-6
    tt = lambda x: torch.tensor((x - mu) / sd, dtype=torch.float32).to(B.dev)
    Xg, Xa, Xq, Xd = tt(g0), tt(T["SPa"]), tt(T["SPq"]), tt(T["SPd"])
    Xg_nd = tt(T["SPg_nd"])
    runs = []
    for seed in seeds:
        gal, art, vs = train_backhead_name(B, Xg, Xg_nd, Xa, Xq, Xd, d_pos,
                                           seed, p1=p1, p2=p2, ls=ls, iw=hiw)
        m = metrics4(B, gal, art)
        m.update(vs)
        runs.append(m)
        log_cb(f"{tag_} seed{seed}: "
               f"neu {m['neutral']['h1']:.3f}/{m['neutral']['h5']:.3f} "
               f"non {m['noname']['h1']:.3f}/{m['noname']['h5']:.3f} "
               f"vsel {vs['vscore']:.3f}")
    return runs


def agg_print(tag_, runs, log_cb=print):
    for var in VORDER:
        h1 = [r[var]["h1"] for r in runs]
        h5 = [r[var]["h5"] for r in runs]
        tg = [r[var]["tag"] for r in runs]
        log_cb(f"AGG {tag_} {var:9s} h1={np.mean(h1):.3f}+-{np.std(h1):.3f} "
               f"h5={np.mean(h5):.3f}+-{np.std(h5):.3f} "
               f"tag={np.mean(tg):.3f}+-{np.std(tg):.3f}")
    m4 = np.mean([np.mean([r[v]["h1"] for r in runs]) for v in VORDER])
    log_cb(f"AGG {tag_} mean-of-4 = {m4:.3f} | "
           f"val-score = {np.mean([r['vscore'] for r in runs]):.3f}")


def run_arm(B, spec: ArmSpec, out_dir=None, seeds_pass1=3, seeds_final=10,
            train_fn=None, log_cb=print):
    """Full pipeline for one arm. Resume-safe: existing tower projections and
    head jsons are skipped, so an interrupted run continues where it stopped."""
    out = out_dir or RESULTS
    out.mkdir(parents=True, exist_ok=True)
    name = spec.name
    final_npz = out / f"tower_{name}_ep{spec.epochs}.npz"
    if not final_npz.exists():
        t0 = time.time()
        if train_fn is not None:
            model, ckpts = train_fn(B, spec, log_cb=log_cb)
        else:
            model, ckpts = train_tower(B, spec, log_cb=log_cb)
        log_cb(f"tower {name}: {spec.epochs} ep in {time.time() - t0:.0f}s")
        tiers, g2wiki, _ = make_doc_tiers(B, spec)
        assert g2wiki, "head phase-2 requires a wiki doc family"
        wiki_tier = (g2wiki,) + tuple(tiers[0][1:])
        zs_traj = {}
        for ek in sorted(ckpts):
            torch.save(ckpts[ek], out / f"ckpt_{name}_ep{ek}.pt")
            m2 = LariceTower(LariceConfig(readout="pool")).to(B.dev)
            m2.load_state_dict({k: v.to(B.dev) for k, v in ckpts[ek].items()})
            m2.eval()
            zk = zs_metrics(m2, B)
            zs_traj[f"ep{ek}"] = zk
            log_cb(f"tower {name} ZS(ep{ek}): "
                   f"{dict((k, round(v, 3)) for k, v in zk.items())}")
            project_cache(m2, B, wiki_tier, out / f"tower_{name}_ep{ek}.npz")
            del m2
        json.dump(zs_traj, open(out / f"zs_traj_{name}.json", "w"), indent=2)

    ck_paths = sorted(out.glob(f"tower_{name}_ep*.npz"),
                      key=lambda p: int(p.stem.split("_ep")[-1]))
    for hsuf, p1, p2, ls in spec.head_cfgs:
        for path in ck_paths:                    # pass 1: every ckpt, 3 seeds
            ek = path.stem.split("_ep")[-1]
            tag_ = f"{name}_ep{ek}{hsuf}"
            outj = out / f"ft4var_{tag_}.json"
            if outj.exists():
                continue
            runs = head_runs(B, path, range(seeds_pass1), p1, p2, ls,
                             spec.hiw, tag_, log_cb)
            agg_print(tag_, runs, log_cb)
            json.dump({"per_seed": runs}, open(outj, "w"), indent=2)
        outb = out / f"ft4var_{name}_best{hsuf}.json"   # pass 2: post-hoc pick
        if outb.exists():
            continue
        vms = {}
        for path in ck_paths:
            ek = path.stem.split("_ep")[-1]
            rs = json.loads((out / f"ft4var_{name}_ep{ek}{hsuf}.json"
                             ).read_text())["per_seed"]
            vms[ek] = float(np.mean([r["vscore"] for r in rs]))
        bek = max(vms, key=vms.get)
        tag_ = f"{name}_best{hsuf}(ep{bek})"
        log_cb(f"POST-HOC pick {name}{hsuf}: ep{bek} "
               f"(val-score {vms[bek]:.3f}); topping up to {seeds_final} seeds")
        prev = json.loads((out / f"ft4var_{name}_ep{bek}{hsuf}.json"
                           ).read_text())["per_seed"]
        runs = prev + head_runs(B, out / f"tower_{name}_ep{bek}.npz",
                                range(seeds_pass1, seeds_final), p1, p2, ls,
                                spec.hiw, tag_, log_cb)
        agg_print(tag_, runs, log_cb)
        json.dump({"best_ep": int(bek), "vsel_by_ep": vms, "per_seed": runs},
                  open(outb, "w"), indent=2)
    log_cb(f"arm {name} done")
