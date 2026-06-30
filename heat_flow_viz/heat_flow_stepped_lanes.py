"""Stepped swimlanes (SYNTHETIC): equal-width time steps + FIXED lane tracks.

Time axis is cut into equal steps (the "staircase"). Each lane is a FIXED
horizontal track (its vertical position never moves -> spatial consistency).
Within each step, the lane's band thickness = its review-heat that step, drawn
as a flat block -> the band is a step/staircase profile. Cross-lane OT transfer
is shown as arrows at the step boundaries.

Usage: python heat_flow_stepped_lanes.py [N_LANES]
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
NL = int(sys.argv[1]) if len(sys.argv) > 1 else 5

GENRES = {"Shooter/MOBA": (2.0, 8.0), "Narrative/RPG": (3.0, 3.0),
          "Survival/Craft": (5.0, 5.2), "Roguelike/Deckbuild": (8.0, 3.0),
          "Cozy/Sim": (8.0, 8.0)}
GC = np.array(list(GENRES.values())); GN = list(GENRES)
FRAMES = ["Early (<=2019)", "2020", "2021", "2022", "2023", "2024"]
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

# ---- fixed lanes (k-means, ordered by PC1; positions NEVER change) ----
allp = np.vstack([g[0] for g in GAMES]); allw = np.concatenate([g[1] for g in GAMES])
samp = allp[rng.choice(len(allw), size=8000, p=allw/allw.sum())]
LC, _ = kmeans2(samp, NL, seed=3, minit="++", missing="raise")
def lane_of(P): return np.argmin(((P[:, None]-LC[None])**2).sum(-1), axis=1)
LAB = [lane_of(g[0]) for g in GAMES]
mu = np.average(allp, axis=0, weights=allw)
cov = np.cov((allp-mu).T, aweights=allw); ev, evec = np.linalg.eigh(cov)
ORDER = np.argsort(LC @ evec[:, np.argmax(ev)])     # bottom -> top
LANE_NAME = [GN[np.argmin(((LC[l]-GC)**2).sum(1))] for l in range(NL)]

heat = np.zeros((NL, NT))
for t in range(NT):
    for l in range(NL):
        heat[l, t] = GAMES[t][1][LAB[t] == l].sum()
hmax = heat.max()

# ---- OT lane->lane transport per step boundary ----
def sinkhorn(a, b, C, eps, it=400):
    K = np.exp(-C/eps); u = np.ones_like(a); v = np.ones_like(b)
    for _ in range(it):
        u = a/(K@v+1e-300); v = b/(K.T@u+1e-300)
    return (u[:, None]*K)*v[None, :]
def onehot(lab, n):
    M = np.zeros((len(lab), n)); M[np.arange(len(lab)), lab] = 1; return M
Mt = []
for t in range(NT-1):
    X1, a = GAMES[t]; X2, b = GAMES[t+1]
    C = ((X1[:, None]-X2[None])**2).sum(-1); eps = 0.3*np.median(C)
    P = sinkhorn(a/a.sum(), b/b.sum(), C, eps)
    Mt.append(onehot(LAB[t], NL).T @ P @ onehot(LAB[t+1], NL))

# ---- render: fixed tracks, stepped blocks ----
fig, ax = plt.subplots(figsize=(18, 6))
cmap = plt.get_cmap("turbo")
SLOT = 1.0                                   # fixed vertical slot per lane
ypos = {ORDER[o]: o * SLOT for o in range(NL)}   # FIXED lane y-centers
lane_color = {l: cmap(0.06 + 0.9*o/max(1, NL-1)) for o, l in enumerate(ORDER)}
GAP = 0.06                                    # gap between steps (shows the staircase edges)

for l in range(NL):
    yc = ypos[l]
    for t in range(NT):
        hh = 0.46 * SLOT * (heat[l, t] / hmax)    # block half-height = heat
        x0, x1 = t + GAP, t + 1 - GAP
        ax.fill_between([x0, x1], yc - hh, yc + hh, color=lane_color[l], zorder=2)
        # thin baseline marking the fixed track
    ax.hlines(yc, -0.2, NT - 0.8, color="lightgrey", lw=0.5, zorder=1)
    ax.text(-0.25, yc, LANE_NAME[l], ha="right", va="center", fontsize=10, color=lane_color[l])

# OT transfer arrows at step boundaries (cross-lane only)
for t in range(NT-1):
    M = Mt[t]; mmax = M.max()
    xa, xb = t + 1 - GAP, t + 1 + GAP
    for i in range(NL):
        for j in range(NL):
            if i == j: continue
            frac = M[i, j] / mmax
            if frac < 0.06: continue
            ax.annotate("", xy=(xb, ypos[j]), xytext=(xa, ypos[i]),
                        arrowprops=dict(arrowstyle="-|>", color=lane_color[i],
                                        alpha=min(0.85, 0.25 + frac), lw=0.6 + 4*frac,
                                        shrinkA=1, shrinkB=1, connectionstyle="arc3,rad=0.15"),
                        zorder=3)

ax.set_xticks([t + 0.5 for t in range(NT)]); ax.set_xticklabels(FRAMES, fontsize=11)
ax.set_yticks([]); ax.set_xlim(-1.0, NT - 0.3); ax.set_ylim(-0.7, (NL-1)*SLOT + 0.7)
ax.set_xlabel("time (equal steps)  ->", fontsize=12)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.set_title(f"Stepped swimlanes (SYNTHETIC): equal time steps, FIXED lane tracks  -  "
             f"{NL} lanes, block height ~ heat, arrows = OT transfer", fontsize=13)
fig.tight_layout()
out = BASE + rf"\heat_flow_stepped_lanes_n{NL}.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)
