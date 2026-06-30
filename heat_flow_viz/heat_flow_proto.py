"""Prototype: heat-transfer streamlines over a (fake) game latent space.

Pipeline mirrors the real plan:
  games -> weighted points in a fixed 2D space
  per time-frame: review-weighted KDE -> heat field -> normalize to sum=1 (conservation)
  consecutive frames -> Horn-Schunck optical flow -> velocity field v
  draw: (a) mean-field streamlines (speed-colored)
        (b) particles advected through v(t), colored by time = "heat transfer"

Frames (impulse model): Early(<=2019 bundled), 2020, 2021, 2022, 2023, 2024.
Data here is SYNTHETIC, only to test the rendering.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter, convolve

rng = np.random.default_rng(7)
OUT = r"C:\Users\admin\AppData\Local\Temp\claude\C--Users-admin-Documents-eval-array-latent\2169e11b-e93d-4912-ae86-dbb6eeb74343\scratchpad\heat_flow_proto.png"

# ---------------------------------------------------------------- 1. fake space
# 5 genre clusters in a pretend VICReg-2D space.
GENRES = {
    "Shooter/MOBA":        (2.0, 8.0),   # old hot (top-left)
    "Narrative/RPG":       (3.0, 3.0),   # old (bottom-left)
    "Survival/Craft":      (5.0, 5.2),   # center, steady
    "Roguelike/Deckbuild": (8.0, 3.0),   # new hot (bottom-right)
    "Cozy/Sim":            (8.0, 8.0),   # new hot (top-right)
}
CENTERS = np.array(list(GENRES.values()))
NAMES = list(GENRES)
FRAMES = ["Early (<=2019)", "2020", "2021", "2022", "2023", "2024"]

# review-weight mass share per genre per frame (rows ~sum to 1): heat drifts L->R
MASS = np.array([
    # Shoot  Narr  Surv  Rogue  Cozy
    [0.45,  0.30, 0.18, 0.04, 0.03],   # Early: old genres dominate
    [0.34,  0.22, 0.22, 0.12, 0.10],   # 2020
    [0.26,  0.17, 0.22, 0.20, 0.15],   # 2021
    [0.18,  0.12, 0.21, 0.27, 0.22],   # 2022
    [0.12,  0.08, 0.18, 0.32, 0.30],   # 2023
    [0.08,  0.05, 0.15, 0.37, 0.35],   # 2024: new genres dominate
])
# few-but-heavy early games vs many later games (the survivorship/concentration note)
N_GAMES = [45, 180, 200, 210, 200, 170]
SIGMA_POS = 0.85   # genre blob spread in data units

def make_frame_games(fi):
    pts, ws = [], []
    for gi in range(len(NAMES)):
        n = max(1, int(round(N_GAMES[fi] * MASS[fi, gi])))
        p = rng.normal(CENTERS[gi], SIGMA_POS, size=(n, 2))
        # heterogeneous per-game review counts (lognormal), scaled so genre total == MASS
        w = rng.lognormal(0.0, 0.8, size=n)
        w = w / w.sum() * MASS[fi, gi]
        pts.append(p); ws.append(w)
    return np.vstack(pts), np.concatenate(ws)

GAMES = [make_frame_games(fi) for fi in range(len(FRAMES))]

# ---------------------------------------------------------------- 2. heat grid
G = 200
XMIN, XMAX, YMIN, YMAX = -1.0, 11.0, -1.0, 11.0
xs = np.linspace(XMIN, XMAX, G)
ys = np.linspace(YMIN, YMAX, G)
CELL = xs[1] - xs[0]
KDE_SIGMA_PX = 7.0   # ~0.42 data units

def heat_field(pts, ws):
    hist = np.zeros((G, G))                      # axis0=y, axis1=x
    ix = np.clip(((pts[:, 0] - XMIN) / CELL).astype(int), 0, G - 1)
    iy = np.clip(((pts[:, 1] - YMIN) / CELL).astype(int), 0, G - 1)
    np.add.at(hist, (iy, ix), ws)
    field = gaussian_filter(hist, KDE_SIGMA_PX, mode="constant")
    field /= field.sum()                          # *** conservation: total heat == 1 ***
    return field

H = [heat_field(p, w) for p, w in GAMES]

# sanity: center-of-mass trajectory + total mass check
print("frame              total_heat   COM(x,y)")
for fi, (p, w) in enumerate(GAMES):
    com = (p * w[:, None]).sum(0) / w.sum()
    print(f"{FRAMES[fi]:18s} {H[fi].sum():.6f}    ({com[0]:.2f}, {com[1]:.2f})  games={len(w)}")

# ---------------------------------------------------------------- 3. game-level OT
def sinkhorn(a, b, C, eps, n_iter=400):
    K = np.exp(-C / eps)
    u = np.ones_like(a); v = np.ones_like(b)
    for _ in range(n_iter):
        u = a / (K @ v + 1e-300)
        v = b / (K.T @ u + 1e-300)
    return (u[:, None] * K) * v[None, :]          # transport plan P (n1 x n2)

def ot_displacement(X1, a, X2, b):
    """Barycentric OT map: per source game i, vector to its mass destination."""
    C = ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1)   # squared dist
    eps = 0.3 * np.median(C)
    P = sinkhorn(a / a.sum(), b / b.sum(), C, eps)
    dest = (P @ X2) / (P.sum(1, keepdims=True) + 1e-300)    # barycentric proj
    return dest - X1                                        # displacement per source game

def splat_vectors(pts, vecs, ws):
    aU = np.zeros((G, G)); aV = np.zeros((G, G)); aW = np.zeros((G, G))
    ix = np.clip(((pts[:, 0] - XMIN) / CELL).astype(int), 0, G - 1)
    iy = np.clip(((pts[:, 1] - YMIN) / CELL).astype(int), 0, G - 1)
    np.add.at(aU, (iy, ix), ws * vecs[:, 0])
    np.add.at(aV, (iy, ix), ws * vecs[:, 1])
    np.add.at(aW, (iy, ix), ws)
    sig = 9.0
    aU = gaussian_filter(aU, sig); aV = gaussian_filter(aV, sig); aW = gaussian_filter(aW, sig)
    return aU, aV, aW

flows_d = []
accU = np.zeros((G, G)); accV = np.zeros((G, G)); accW = np.zeros((G, G))
for t in range(len(GAMES) - 1):
    X1, a = GAMES[t]; X2, b = GAMES[t + 1]
    disp = ot_displacement(X1, a, X2, b)
    aU, aV, aW = splat_vectors(X1, disp, a)
    U_t = aU / (aW + 1e-9); V_t = aV / (aW + 1e-9)
    flows_d.append((U_t, V_t))
    accU += aU; accV += aV; accW += aW

Umean = accU / (accW + 1e-9)
Vmean = accV / (accW + 1e-9)
SPEED = np.hypot(Umean, Vmean)
# mask streamlines to where source heat actually exists
mask = accW < (accW.max() * 0.02)
Um = np.where(mask, np.nan, Umean); Vm = np.where(mask, np.nan, Vmean)

# ---------------------------------------------------------------- 4. particle advect
def sample_field(F, x, y):
    fx = np.clip((x - XMIN) / CELL, 0, G - 1.001)
    fy = np.clip((y - YMIN) / CELL, 0, G - 1.001)
    x0 = fx.astype(int); y0 = fy.astype(int)
    dx = fx - x0; dy = fy - y0
    return (F[y0, x0]*(1-dx)*(1-dy) + F[y0, x0+1]*dx*(1-dy)
            + F[y0+1, x0]*(1-dx)*dy + F[y0+1, x0+1]*dx*dy)

# seed particles from early-frame games (weighted)
ep, ew = GAMES[0]
idx = rng.choice(len(ew), size=140, p=ew/ew.sum())
seeds = ep[idx] + rng.normal(0, 0.15, size=(140, 2))
K = 12            # substeps per transition
GAIN = 1.0
paths, tcols = [], []
for s in seeds:
    pos = s.copy(); pts = [pos.copy()]; ts = [0.0]
    for ti, (Ud, Vd) in enumerate(flows_d):
        for k in range(K):
            ux = sample_field(Ud, np.array([pos[0]]), np.array([pos[1]]))[0]
            uy = sample_field(Vd, np.array([pos[0]]), np.array([pos[1]]))[0]
            step = np.array([ux, uy]) * GAIN / K
            n = np.hypot(*step)
            if n > 0.6:                       # clip runaway
                step *= 0.6 / n
            pos = pos + step
            pts.append(pos.copy()); ts.append((ti + (k + 1) / K) / len(flows_d))
    paths.append(np.array(pts)); tcols.append(np.array(ts))

# ---------------------------------------------------------------- 5. render
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.6, 1.6], hspace=0.28, wspace=0.15)

# row 1: heat frames
gmax = max(h.max() for h in H)
for fi in range(6):
    ax = fig.add_subplot(gs[0, fi])
    ax.imshow(H[fi], origin="lower", extent=[XMIN, XMAX, YMIN, YMAX],
              cmap="magma", vmin=0, vmax=gmax)
    ax.scatter(CENTERS[:, 0], CENTERS[:, 1], s=8, c="cyan", alpha=0.6)
    ax.set_title(FRAMES[fi], fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.text(0.5, 0.905, "Per-frame heat field  (review-weighted KDE, normalized to total=1)",
         ha="center", fontsize=11)

def label_genres(ax):
    for nm, (gx, gy) in GENRES.items():
        ax.text(gx, gy, nm, fontsize=8, ha="center", va="center",
                color="white", bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45))

# row 2: mean-field streamlines
ax2 = fig.add_subplot(gs[1, :])
ax2.imshow(H[0] + H[-1], origin="lower", extent=[XMIN, XMAX, YMIN, YMAX],
           cmap="Greys", alpha=0.35)
ax2.contour(xs, ys, H[0],  levels=4, colors="#3b6fff", linewidths=1.0, alpha=0.7)
ax2.contour(xs, ys, H[-1], levels=4, colors="#ff3b3b", linewidths=1.0, alpha=0.7)
SPEEDm = np.where(mask, np.nan, SPEED)
strm = ax2.streamplot(xs, ys, Um, Vm, color=SPEEDm, cmap="turbo",
                      density=1.5, linewidth=1.3, arrowsize=1.2)
label_genres(ax2)
ax2.set_title("Mean-field flow (optimal transport).  blue=Early heat  red=2024 heat  "
              "lines=net heat transfer (color=speed)", fontsize=10)
ax2.set_xlim(XMIN, XMAX); ax2.set_ylim(YMIN, YMAX)
fig.colorbar(strm.lines, ax=ax2, fraction=0.025, pad=0.01, label="flow speed")

# row 3: time-colored particle trajectories
ax3 = fig.add_subplot(gs[2, :])
ax3.imshow(H[0] + H[-1], origin="lower", extent=[XMIN, XMAX, YMIN, YMAX],
           cmap="Greys", alpha=0.25)
segs, segc = [], []
for path, tc in zip(paths, tcols):
    for i in range(len(path) - 1):
        segs.append([path[i], path[i + 1]]); segc.append(tc[i])
lc = LineCollection(segs, array=np.array(segc), cmap="plasma", linewidth=1.0, alpha=0.8)
ax3.add_collection(lc)
ax3.scatter(seeds[:, 0], seeds[:, 1], s=6, c="black", alpha=0.5, label="seed (Early hotspots)")
label_genres(ax3)
ax3.set_title("Heat-transfer streaks: particles seeded on Early hotspots, advected through "
              "v(t).  color = time (Early->2024)", fontsize=10)
ax3.set_xlim(XMIN, XMAX); ax3.set_ylim(YMIN, YMAX)
ax3.legend(loc="upper left", fontsize=8)
fig.colorbar(lc, ax=ax3, fraction=0.025, pad=0.01, label="time (0=Early, 1=2024)")

fig.suptitle("Game-review heat transfer over release cohorts  (SYNTHETIC data prototype)",
             fontsize=14, y=0.95)
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("\nwrote", OUT)
