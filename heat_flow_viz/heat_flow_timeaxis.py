"""19:6 long banner -- TIME on the horizontal axis (SYNTHETIC).

x = time (Early -> 2024).  y = latent space collapsed to 1D (PC1 of VICReg-2D).
Background = heat "river": per-time 1D density of review-heat (sum=1 per column).
Foreground = N OT-driven worldlines flowing left->right, width ~ corridor heat.

Usage: python heat_flow_timeaxis.py [N_LINES]
"""
from __future__ import annotations
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.cluster.vq import kmeans2

rng = np.random.default_rng(7)
BASE = r"C:\Users\admin\AppData\Local\Temp\claude\C--Users-admin-Documents-eval-array-latent\2169e11b-e93d-4912-ae86-dbb6eeb74343\scratchpad"
N_LINES = int(sys.argv[1]) if len(sys.argv) > 1 else 8

GENRES = {"Shooter/MOBA": (2.0, 8.0), "Narrative/RPG": (3.0, 3.0),
          "Survival/Craft": (5.0, 5.2), "Roguelike/Deckbuild": (8.0, 3.0),
          "Cozy/Sim": (8.0, 8.0)}
CENTERS = np.array(list(GENRES.values()))
FRAMES = ["Early (<=2019)", "2020", "2021", "2022", "2023", "2024"]
MASS = np.array([
    [0.45, 0.30, 0.18, 0.04, 0.03], [0.34, 0.22, 0.22, 0.12, 0.10],
    [0.26, 0.17, 0.22, 0.20, 0.15], [0.18, 0.12, 0.21, 0.27, 0.22],
    [0.12, 0.08, 0.18, 0.32, 0.30], [0.08, 0.05, 0.15, 0.37, 0.35]])
N_GAMES = [45, 180, 200, 210, 200, 170]

def make(fi):
    pts, ws = [], []
    for gi in range(len(CENTERS)):
        n = max(1, int(round(N_GAMES[fi] * MASS[fi, gi])))
        pts.append(rng.normal(CENTERS[gi], 0.85, size=(n, 2)))
        w = rng.lognormal(0, 0.8, size=n); ws.append(w / w.sum() * MASS[fi, gi])
    return np.vstack(pts), np.concatenate(ws)

GAMES = [make(fi) for fi in range(6)]
NT = len(GAMES)

# ---- 1D collapse: PC1 of pooled (weighted) coords ----
allp = np.vstack([g[0] for g in GAMES]); allw = np.concatenate([g[1] for g in GAMES])
mu = np.average(allp, axis=0, weights=allw)
cov = np.cov((allp - mu).T, aweights=allw)
evals, evecs = np.linalg.eigh(cov)
pc1 = evecs[:, np.argmax(evals)]
if pc1[0] < 0: pc1 = -pc1                     # orient consistently
def proj(xy): return (xy - mu) @ pc1
SMIN, SMAX = proj(allp).min() - 0.5, proj(allp).max() + 0.5

# ---- 2D OT velocity fields (for advection) ----
G, (XMIN, XMAX, YMIN, YMAX) = 160, (-1.0, 11.0, -1.0, 11.0)
xs = np.linspace(XMIN, XMAX, G); CELL = xs[1] - xs[0]

def sinkhorn(a, b, C, eps, it=400):
    K = np.exp(-C/eps); u = np.ones_like(a); v = np.ones_like(b)
    for _ in range(it):
        u = a/(K@v+1e-300); v = b/(K.T@u+1e-300)
    return (u[:, None]*K)*v[None, :]

def ot_disp(X1, a, X2, b):
    C = ((X1[:, None]-X2[None])**2).sum(-1); eps = 0.3*np.median(C)
    P = sinkhorn(a/a.sum(), b/b.sum(), C, eps)
    return (P@X2)/(P.sum(1, keepdims=True)+1e-300) - X1

def splat(pts, vecs, ws):
    aU = np.zeros((G, G)); aV = np.zeros((G, G)); aW = np.zeros((G, G))
    ix = np.clip(((pts[:, 0]-XMIN)/CELL).astype(int), 0, G-1)
    iy = np.clip(((pts[:, 1]-YMIN)/CELL).astype(int), 0, G-1)
    np.add.at(aU, (iy, ix), ws*vecs[:, 0]); np.add.at(aV, (iy, ix), ws*vecs[:, 1])
    np.add.at(aW, (iy, ix), ws)
    aU = gaussian_filter(aU, 8.0); aV = gaussian_filter(aV, 8.0); aW = gaussian_filter(aW, 8.0)
    return aU/(aW+1e-9), aV/(aW+1e-9)

DISP = [ot_disp(GAMES[t][0], GAMES[t][1], GAMES[t+1][0], GAMES[t+1][1]) for t in range(NT-1)]
flows = [splat(GAMES[t][0], DISP[t], GAMES[t][1]) for t in range(NT-1)]

def sample(F, x, y):
    fx = np.clip((x-XMIN)/CELL, 0, G-1.001); fy = np.clip((y-YMIN)/CELL, 0, G-1.001)
    x0 = int(fx); y0 = int(fy); dx = fx-x0; dy = fy-y0
    return (F[y0, x0]*(1-dx)*(1-dy)+F[y0, x0+1]*dx*(1-dy)
            + F[y0+1, x0]*(1-dx)*dy+F[y0+1, x0+1]*dx*dy)

# ---- background heat river: displacement-interp frames -> 1D density columns ----
SUB = 16; NB = 240
sbins = np.linspace(SMIN, SMAX, NB)
cols, tcol = [], []
for t in range(NT-1):
    X1, a = GAMES[t]
    for k in range(SUB):
        P = X1 + (k/SUB)*DISP[t]
        h, _ = np.histogram(proj(P), bins=NB, range=(SMIN, SMAX), weights=a)
        h = gaussian_filter1d(h, 4.0); cols.append(h/(h.sum()+1e-12)); tcol.append(t + k/SUB)
h, _ = np.histogram(proj(GAMES[-1][0]), bins=NB, range=(SMIN, SMAX), weights=GAMES[-1][1])
h = gaussian_filter1d(h, 4.0); cols.append(h/(h.sum()+1e-12)); tcol.append(NT-1.0)
river = np.array(cols).T                                  # (NB latent, time)

# ---- N worldlines (weighted k-means seeds on Early), advect, project to 1D ----
ep, ew = GAMES[0]
samp = ep[rng.choice(len(ew), size=5000, p=ew/ew.sum())]
cent, lab = kmeans2(samp, N_LINES, seed=1, minit="++", missing="raise")
seed_w = np.bincount(lab, minlength=N_LINES).astype(float); seed_w /= seed_w.sum()
K = 16
paths = []
for s in cent:
    pos = s.copy(); rec = [(0.0, proj(pos))]
    for ti, (U, V) in enumerate(flows):
        for k in range(K):
            step = np.array([sample(U, pos[0], pos[1]), sample(V, pos[0], pos[1])])/K
            n = np.hypot(*step)
            if n > 0.6: step *= 0.6/n
            pos = pos + step
            rec.append((ti + (k+1)/K, proj(pos)))
    paths.append(np.array(rec))

# ---- render 19:6 ----
fig, ax = plt.subplots(figsize=(19, 6))
ax.imshow(river, origin="lower", extent=[0, NT-1, SMIN, SMAX], aspect="auto",
          cmap="magma", vmax=np.percentile(river, 99.5))
wmax = seed_w.max()
cmap = plt.get_cmap("cool")
for p, w in zip(paths, seed_w):
    ax.plot(p[:, 0], p[:, 1], color=cmap(0.2 + 0.8*w/wmax),
            lw=0.8 + 6.0*(w/wmax), alpha=0.9, solid_capstyle="round")
# genre labels at left (Early) and right (2024)
for nm, c in GENRES.items():
    ax.text(-0.06*(NT-1), proj(np.array(c)), nm, ha="right", va="center", fontsize=9, color="navy")
    ax.text((NT-1)*1.005, proj(np.array(c)), nm, ha="left", va="center", fontsize=9, color="darkred")
ax.set_xticks(range(NT)); ax.set_xticklabels(FRAMES, fontsize=11)
ax.set_xlabel("time  ->", fontsize=12); ax.set_ylabel("latent axis (PC1)", fontsize=12)
ax.set_xlim(0, NT-1); ax.set_ylim(SMIN, SMAX)
ax.set_title(f"Game-review heat flow over time  (SYNTHETIC, 19:6)  -  {N_LINES} OT corridors,"
             f"  width ~ heat", fontsize=13)
fig.tight_layout()
out = BASE + rf"\heat_flow_timeaxis_n{N_LINES}.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)
