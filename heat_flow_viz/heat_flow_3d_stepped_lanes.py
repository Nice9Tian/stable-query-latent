"""3D stepped swimlanes (SYNTHETIC): lanes live at their TRUE latent (x,y).

Each topic lane is a column standing at its real 2D-latent centroid (spatial
consistency = genuine embedding coordinates). Time = z axis, cut into equal
steps; per step a box (the "stair") whose footprint ~ that step's review-heat.
OT transfer = arrows between columns across step boundaries.

Usage: python heat_flow_3d_stepped_lanes.py [N_LANES]
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

# fixed lanes at TRUE latent centroids
allp = np.vstack([g[0] for g in GAMES]); allw = np.concatenate([g[1] for g in GAMES])
samp = allp[rng.choice(len(allw), size=8000, p=allw/allw.sum())]
LC, _ = kmeans2(samp, NL, seed=3, minit="++", missing="raise")
def lane_of(P): return np.argmin(((P[:, None]-LC[None])**2).sum(-1), axis=1)
LAB = [lane_of(g[0]) for g in GAMES]
LANE_NAME = [GN[np.argmin(((LC[l]-GC)**2).sum(1))] for l in range(NL)]
heat = np.array([[GAMES[t][1][LAB[t] == l].sum() for t in range(NT)] for l in range(NL)])
hmax = heat.max()

# OT transport per step boundary
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
    Mt.append(onehot(LAB[t], NL).T @ sinkhorn(a/a.sum(), b/b.sum(), C, eps) @ onehot(LAB[t+1], NL))

cmap = plt.get_cmap("turbo")
lane_color = {l: cmap(0.06 + 0.9*l/max(1, NL-1)) for l in range(NL)}
ZSTEP, DZ, SMAX = 1.0, 0.72, 2.6           # z spacing, block thickness, max footprint

def render(view, fname):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection="3d")
    for l in range(NL):
        cx, cy = LC[l]
        ax.plot([cx, cx], [cy, cy], [0, (NT-1)*ZSTEP+DZ], color="lightgrey", lw=0.6, alpha=0.5)
        for t in range(NT):
            s = SMAX * np.sqrt(heat[l, t] / hmax)        # footprint ~ sqrt(heat)
            if s < 1e-3: continue
            ax.bar3d(cx - s/2, cy - s/2, t*ZSTEP, s, s, DZ,
                     color=lane_color[l], alpha=0.92, shade=True)
        ax.text(cx, cy, (NT-1)*ZSTEP + DZ + 0.3, LANE_NAME[l], fontsize=8,
                ha="center", color="black")
    # OT transfer arrows (cross-lane, significant)
    for t in range(NT-1):
        M = Mt[t]; mmax = M.max()
        for i in range(NL):
            for j in range(NL):
                if i == j: continue
                frac = M[i, j] / mmax
                if frac < 0.12: continue
                xi, yi = LC[i]; xj, yj = LC[j]
                ax.quiver(xi, yi, t*ZSTEP + DZ, xj-xi, yj-yi, ZSTEP-DZ,
                          color=lane_color[i], alpha=min(0.9, 0.3+frac),
                          lw=0.6+3*frac, arrow_length_ratio=0.18)
    ax.set_xlabel("latent dim 1"); ax.set_ylabel("latent dim 2"); ax.set_zlabel("time")
    ax.set_zticks([t*ZSTEP for t in range(NT)]); ax.set_zticklabels(FRAMES, fontsize=7)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_zlim(0, (NT-1)*ZSTEP+DZ+0.6)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_title(f"3D stepped swimlanes (SYNTHETIC): columns at true latent positions, "
                 f"z=time steps, footprint~heat - {NL} lanes", fontsize=11)
    fig.savefig(fname, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", fname)

render((18, -60), BASE + rf"\heat_flow_3d_stepped_lanes_n{NL}_a.png")
render((30, 35),  BASE + rf"\heat_flow_3d_stepped_lanes_n{NL}_b.png")
