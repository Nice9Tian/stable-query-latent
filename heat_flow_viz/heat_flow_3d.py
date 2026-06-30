"""3D heat-transfer worldlines (SYNTHETIC).

Plane (x,y) = VICReg-2D latent space.  Vertical axis z = time.
Each release-frame is a horizontal heat slice; optimal-transport velocity
fields connect consecutive slices, and particles seeded on the Early heat are
advected through them and lifted to z=time -> 3D worldlines of heat transfer.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(7)
BASE = r"C:\Users\admin\AppData\Local\Temp\claude\C--Users-admin-Documents-eval-array-latent\2169e11b-e93d-4912-ae86-dbb6eeb74343\scratchpad"

GENRES = {"Shooter/MOBA": (2.0, 8.0), "Narrative/RPG": (3.0, 3.0),
          "Survival/Craft": (5.0, 5.2), "Roguelike/Deckbuild": (8.0, 3.0),
          "Cozy/Sim": (8.0, 8.0)}
CENTERS = np.array(list(GENRES.values()))
FRAMES = ["Early\n(<=2019)", "2020", "2021", "2022", "2023", "2024"]
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

G, (XMIN, XMAX, YMIN, YMAX) = 160, (-1.0, 11.0, -1.0, 11.0)
xs = np.linspace(XMIN, XMAX, G); ys = np.linspace(YMIN, YMAX, G)
CELL = xs[1] - xs[0]; XX, YY = np.meshgrid(xs, ys)

def field(pts, ws):
    h = np.zeros((G, G))
    ix = np.clip(((pts[:, 0]-XMIN)/CELL).astype(int), 0, G-1)
    iy = np.clip(((pts[:, 1]-YMIN)/CELL).astype(int), 0, G-1)
    np.add.at(h, (iy, ix), ws)
    h = gaussian_filter(h, 6.0, mode="constant")
    return h / h.sum()

H = [field(*g) for g in GAMES]

# ---- optimal transport -> grid velocity fields ----
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

flows = []
for t in range(NT-1):
    X1, a = GAMES[t]; X2, b = GAMES[t+1]
    flows.append(splat(X1, ot_disp(X1, a, X2, b), a))

def sample(F, x, y):
    fx = np.clip((x-XMIN)/CELL, 0, G-1.001); fy = np.clip((y-YMIN)/CELL, 0, G-1.001)
    x0 = int(fx); y0 = int(fy); dx = fx-x0; dy = fy-y0
    return (F[y0, x0]*(1-dx)*(1-dy)+F[y0, x0+1]*dx*(1-dy)
            + F[y0+1, x0]*(1-dx)*dy+F[y0+1, x0+1]*dx*dy)

# ---- seed lines by PRUNING (greedy heat + redundancy suppression) ----
# Candidate = every Early-frame game (weight = its review heat). Walk heaviest
# first; KEEP a candidate only if it is >= MIN_SEP from every already-kept line,
# else PRUNE it (redundant). Stop at N_LINES. Knobs: N_LINES (count), MIN_SEP.
import sys
N_LINES = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MIN_SEP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
ep, ew = GAMES[0]
order = np.argsort(ew)[::-1]              # heaviest candidates first
kept = []
for i in order:
    p = ep[i]
    if all(np.hypot(*(p - ep[j])) >= MIN_SEP for j in kept):
        kept.append(i)
    if len(kept) >= N_LINES:
        break
seeds = ep[kept]
seed_w = ew[kept]; seed_w = seed_w / seed_w.sum()
print(f"pruned: kept {len(kept)} / {len(ew)} candidate lines  (N={N_LINES}, sep={MIN_SEP})")

# ---- advect each seed, lift to z = time ----
K = 14
paths, path_w = [], []
for s, w in zip(seeds, seed_w):
    pos = s.copy(); pts = [(pos[0], pos[1], 0.0)]
    for ti, (U, V) in enumerate(flows):
        for k in range(K):
            step = np.array([sample(U, pos[0], pos[1]), sample(V, pos[0], pos[1])]) / K
            n = np.hypot(*step)
            if n > 0.6: step *= 0.6/n
            pos = pos + step
            pts.append((pos[0], pos[1], ti + (k+1)/K))
    paths.append(np.array(pts)); path_w.append(w)

# segments + per-segment time color + per-segment width (~corridor heat)
segs, cols, lws = [], [], []
wmax = max(path_w)
for p, w in zip(paths, path_w):
    lw = 0.6 + 6.0 * (w / wmax)
    for i in range(len(p)-1):
        segs.append([p[i], p[i+1]]); cols.append(p[i, 2] / (NT-1)); lws.append(lw)

def render(view, fname):
    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(projection="3d")
    # heat slices
    for t in range(NT):
        ax.contourf(XX, YY, H[t], zdir="z", offset=t, levels=8, cmap="magma",
                    alpha=0.32, antialiased=True)
        ax.text(XMIN, YMAX, t, " " + FRAMES[t].replace("\n", " "), fontsize=8, color="k")
    # worldlines
    lc = Line3DCollection(segs, array=np.array(cols), cmap="plasma", linewidths=lws, alpha=0.9)
    ax.add_collection3d(lc)
    # seed markers (corridor starts), size ~ heat
    ax.scatter(seeds[:, 0], seeds[:, 1], np.zeros(len(seeds)), s=20+400*np.array(path_w),
               c="k", alpha=0.5, depthshade=False)
    # genre pillars (faint vertical guides)
    for nm, (gx, gy) in GENRES.items():
        ax.plot([gx, gx], [gy, gy], [0, NT-1], color="grey", lw=0.6, alpha=0.35)
        ax.text(gx, gy, NT-1+0.15, nm, fontsize=7, ha="center", color="navy")
    ax.set_xlabel("latent dim 1"); ax.set_ylabel("latent dim 2"); ax.set_zlabel("time")
    ax.set_zticks(range(NT)); ax.set_zticklabels([f.replace("\n", " ") for f in FRAMES], fontsize=7)
    ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX); ax.set_zlim(0, NT-1)
    ax.view_init(elev=view[0], azim=view[1])
    cb = fig.colorbar(lc, ax=ax, fraction=0.022, pad=0.02); cb.set_label("time (Early->2024)")
    ax.set_title(f"3D heat-transfer worldlines (SYNTHETIC): z=time, OT-driven, {N_LINES} corridors",
                 fontsize=12)
    fig.savefig(fname, dpi=115, bbox_inches="tight"); plt.close(fig)
    print("wrote", fname)

render((12, 30), BASE + rf"\heat_flow_3d_n{N_LINES}.png")
