"""Capability-vs-data curve: real-text tag F1 (mean of 4 styles) for the two
best families at n in {200,500,1000,2000}.

Self-consistent light protocol for ALL 8 points: game-side feature = centroid
of the FIRST 2000 review sentences through the frozen encoder
(encode_text_centroid, the same function the article side uses); ridge trained
on the standard split; articles scored per style; y = mean over the 4 styles.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, r"C:\Users\admin\Documents\studable query latent")
from VICReg_review import disturbtion_embed
from VICReg_review import eval_battery_worker as worker
from VICReg_review import run_data_view_sweep as sweep
from VICReg_review import text_variant_eval as tve
from VICReg_review.identity_diagnostic import encode_text_centroid

root = Path(r"C:\runpod_data\stable-query-latent\VICReg_review\heads\cloud_full_sweep_a100")
H5 = r"C:\Users\admin\Documents\studable query latent\game_review_data\embedding_h5.h5"
VAR = ("neutral", "positive", "negative", "noname")
CAP = 2000

ev = worker.build_eval_args(argparse.Namespace(h5=H5, out_dir=str(root)))
ev.out_dir = root
device = torch.device("cuda")
with h5py.File(H5, "r") as h5:
    input_dim = int(h5.attrs["input_dim"])
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["game_names"][:]]
    gro = h5["game_review_offsets"][:]
    ro = h5["review_offsets"][:]
n2i = {n: i for i, n in enumerate(names)}

cache = disturbtion_embed.load_npz_payload(root / "text_variant_pilot_cache.npz")
c_names = [str(x) for x in cache["names"]]
c_variants = [str(x) for x in cache["variants"]]
c_vec = np.asarray(cache["vectors"], dtype=np.float32)
c_off = np.asarray(cache["offsets"], dtype=np.int64)
rec = {}
for i, (n, v) in enumerate(zip(c_names, c_variants)):
    rec[(n, v)] = c_vec[c_off[i]:c_off[i + 1]]
art_games = sorted({n for (n, v) in rec if n in n2i and all((n, w) in rec for w in VAR)})

class EncArgs:
    eval_feature_views = 1
    feature_views = 1
    sample_fraction = 1.0
    eval_sample_fraction = 1.0
    text_variant_feature_views = 1
    text_variant_sample_fraction = 1.0
    seed = 20260705
    amp = True
    max_batch_sentences = 4096

FAMS = {
    "no-GRL d=36 rho=0.2": "dim036_nogrl_n{n}_view20",
    "GRL d=64 rho=0.6": "dim064_grl_n{n}_view60_lat512x2",
}
NS = [200, 500, 1000, 2000]
OUT_JSON = root / "capability_vs_data.json"
out = (json.loads(OUT_JSON.read_text(encoding="utf-8"))
       if OUT_JSON.exists() else {})
y_lab, _tags = tve.align_labels(Path(H5), names)
split = tve.make_or_load_split(root / "tag_text_eval_split.json", names, ev)

with h5py.File(H5, "r") as h5:
    vec = h5["vectors"]
    for label, pat in FAMS.items():
        out.setdefault(label, {})
        for nval in NS:
            if str(nval) in out[label]:
                print(f"{pat.format(n=nval)}: cached, skip", flush=True)
                continue
            cid = pat.format(n=nval)
            cdir = root / cid
            ready, ckpt = worker.combo_ready(cdir)
            if not ready:
                print(f"{cid}: not ready, skip")
                continue
            encoder, _, _, _ = sweep.load_frozen_encoder(ckpt, input_dim, device)
            # game-side centroid features (first CAP review sentences)
            game_vecs = []
            for i in range(len(names)):
                s = int(ro[int(gro[i])])
                e = min(int(ro[int(gro[i + 1])]), s + CAP)
                block = vec[s:e].astype(np.float32)
                game_vecs.append(encode_text_centroid(encoder, block, EncArgs, device))
                if i % 500 == 0:
                    print(f"  {cid} game {i}/{len(names)}", flush=True)
            X = np.stack(game_vecs).astype(np.float32)
            a_scaler, a_ridge, _al, a_th, _vm = tve.train_anchor_ridge(ev, X, y_lab, n2i, split)
            ya = np.stack([y_lab[n2i[g]] for g in art_games])
            f1s = []
            for v in VAR:
                Xv = np.stack([encode_text_centroid(encoder, rec[(g, v)], EncArgs, device)
                               for g in art_games]).astype(np.float32)
                m = tve.micro_prf(ya, a_ridge.predict(a_scaler.transform(Xv)), a_th)
                f1s.append(m["micro_f1"])
            out[label][str(nval)] = float(np.mean(f1s))
            OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                encoding="utf-8")
            print(f"{cid}: mean4 article F1 = {out[label][str(nval)]:.3f}", flush=True)
            del encoder
            torch.cuda.empty_cache()

print(json.dumps(out, indent=1))
