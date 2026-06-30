"""19:6 alluvial / Sankey banner -- TIME on the horizontal axis (SYNTHETIC).

Games -> N topic LANES (k-means on the pooled latent space, fixed across time).
Per frame: lane thickness = that lane's review-heat (each time-column sums to 1).
Between frames: game-level OT plan aggregated to a lane->lane transport matrix;
each lane->lane flow is drawn as a ribbon (width = heat moved, color = source lane).

Usage: python heat_flow_alluvial.py [N_LANES]
"""
from __future__ import annotations
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2

rng = np.random.default_rng(7)
BASE = r"C:\Users\admin\AppData\Local\Temp\claude\C--Users-admin-Documents-eval-array-latent\2169e11b-e93d-4912-ae86-dbb6eeb74343\scratchpad"
NL = int(sys.argv[1]) if len(sys.argv) > 1 else 6

GENRES = {"Shooter/MOBA": (2.0, 8.0), "Narrative/RPG": (3.0, 3.0),
          "Survival/Craft": (5.0, 5.2), "Roguelike/Deckbuild": (8.0, 3.0),
          "Cozy/Sim": (8.0, 8.0)}
GC = np.array(list(GENRES.values())); GN = list(GENRES)
FRAMES = ["Early\n(<=2019)", "2020", "2021", "2022", "2023", "2024"]
MASS = np.array([
    [0.45, 0.30, 0.18, 0.04, 0.03], [0.34, 0.22, 0.22, 0.12, 0.10],
    [0.26, 0.17, 0.22, 0.20, 0.15], [0.18, 0.12, 0.21, 0.27, 0.22],
    [0.12, 0.08, 0.18, 0.32, 0.30], [0.08, 0.05, 0.15, 0.37, 0.35]])
N_GAMES = [45, 180, 200, 210, 200, 170]

def make(fi):
    pts, ws = [], []
    for gi in range(len(GC)):
        n = max(1, int(round(N_GAMES[fi] * MASS[fi, gi])))
        pts.append(rng.normal(GC[gi], 0.85, size=(n, 2)))
        w = rng.lognormal(0, 0.8, size=n); ws.append(w / w.sum() * MASS[fi, gi])
    return np.vstack(pts), np.concatenate(ws)

GAMES = [make(fi) for fi in range(6)]
NT = len(GAMES)

# ---- lanes: k-means on pooled coords (fixed across time) ----
allp = np.vstack([g[0] for g in GAMES]); allw = np.concatenate([g[1] for g in GAMES])
samp = allp[rng.choice(len(allw), size=8000, p=allw/allw.sum())]
LC, _ = kmeans2(samp, NL, seed=3, minit="++", missing="raise")
def lane_of(P): return np.argmin(((P[:, None]-LC[None])**2).sum(-1), axis=1)
LAB = [lane_of(g[0]) for g in GAMES]

# order lanes by PC1 of their centroid so spatial neighbours are adjacent (fewer crossings)
mu = np.average(allp, axis=0, weights=allw)
cov = np.cov((allp-mu).T, aweights=allw); ev, evec = np.linalg.eigh(cov)
pc1 = evec[:, np.argmax(ev)]
ORDER = np.argsort(LC @ pc1)                 # lane id in stacking order (bottom->top)
POS = {lane: o for o, lane in enumerate(ORDER)}
# nearest genre name per lane (for labels)
LANE_NAME = [GN[np.argmin(((LC[l]-GC)**2).sum(1))] for l in range(NL)]

# heat per lane per frame (each frame sums to 1)
heat = np.zeros((NL, NT))
for t in range(NT):
    for l in range(NL):
        heat[l, t] = GAMES[t][1][LAB[t] == l].sum()

# ---- OT lane->lane transport matrices ----
def sinkhorn(a, b, C, eps, it=400):
    K = np.exp(-C/eps); u = np.ones_like(a); v = np.ones_like(b)
    for _ in range(it):
        u = a/(K@v+1e-300); v = b/(K.T@u+1e-300)
    return (u[:, None]*K)*v[None, :]

def onehot(lab, n):
    M = np.zeros((len(lab), n)); M[np.arange(len(lab)), lab] = 1; return M

Mt = []                                       # transport matrix per transition (NL x NL)
for t in range(NT-1):
    X1, a = GAMES[t]; X2, b = GAMES[t+1]
    C = ((X1[:, None]-X2[None])**2).sum(-1); eps = 0.3*np.median(C)
    P = sinkhorn(a/a.sum(), b/b.sum(), C, eps)
    Mt.append(onehot(LAB[t], NL).T @ P @ onehot(LAB[t+1], NL))

# ---- band geometry: stacked, centered streamgraph (col sum = 1) ----
GAP = 0.012                                   # small visual gap between lanes
def bands(t):
    th = heat[:, t]
    out = {}
    cum = -(th.sum() + GAP*(NL-1))/2
    for o in range(NL):
        lane = ORDER[o]
        out[lane] = (cum, cum + th[lane]); cum += th[lane] + GAP
    return out
BANDS = [bands(t) for t in range(NT)]

# ---- render ----
fig, ax = plt.subplots(figsize=(19, 6))
cmap = plt.get_cmap("turbo")
lane_color = {lane: cmap(0.05 + 0.9*POS[lane]/max(1, NL-1)) for lane in range(NL)}
NODE_W = 0.045

# ribbons
for t in range(NT-1):
    xa = t + NODE_W; xb = (t+1) - NODE_W
    src_cum = {l: BANDS[t][l][0] for l in range(NL)}      # outgoing stack (by target order)
    dst_cum = {l: BANDS[t+1][l][0] for l in range(NL)}    # incoming stack (by source order)
    for i in ORDER:                                       # source lanes bottom->top
        for j in ORDER:                                   # target lanes bottom->top
            m = Mt[t][i, j]
            if m < 1e-4: continue
            s0 = src_cum[i]; src_cum[i] += m; s1 = src_cum[i]
            d0 = dst_cum[j]; dst_cum[j] += m; d1 = dst_cum[j]
            xx = np.linspace(xa, xb, 50); s = (xx-xa)/(xb-xa); s = s*s*(3-2*s)
            bot = s0 + (d0-s0)*s; top = s1 + (d1-s1)*s
            ax.fill_between(xx, bot, top, color=lane_color[i], alpha=0.45,
                            linewidth=0, zorder=1)

# nodes (lane blocks at each time)
for t in range(NT):
    for l in range(NL):
        y0, y1 = BANDS[t][l]
        if y1 - y0 < 1e-5: continue
        ax.add_patch(plt.Rectangle((t-NODE_W, y0), 2*NODE_W, y1-y0,
                     color=lane_color[l], ec="white", lw=0.5, zorder=3))

# lane labels (left at Early, right at 2024)
for l in range(NL):
    y0, y1 = BANDS[0][l]
    if y1 - y0 > 0.015:
        ax.text(-0.08, (y0+y1)/2, LANE_NAME[l], ha="right", va="center", fontsize=9,
                color=lane_color[l])
    y0, y1 = BANDS[NT-1][l]
    if y1 - y0 > 0.015:
        ax.text(NT-1+0.08, (y0+y1)/2, LANE_NAME[l], ha="left", va="center", fontsize=9,
                color=lane_color[l])

ax.set_xticks(range(NT)); ax.set_xticklabels([f.replace("\n", " ") for f in FRAMES], fontsize=11)
ax.set_yticks([]); ax.set_xlim(-0.7, NT-1+0.7); ax.set_ylim(-0.62, 0.62)
ax.set_xlabel("time  ->", fontsize=12)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.set_title(f"Game-review heat flow over time  (SYNTHETIC alluvial, 19:6)  -  {NL} topic lanes,"
             f"  band width ~ heat,  ribbons = OT transfer", fontsize=13)
fig.tight_layout()
out = BASE + rf"\heat_flow_alluvial_n{NL}.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)
