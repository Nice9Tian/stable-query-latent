"""Small-multiples banner (SYNTHETIC): spatial consistency + time left->right.

Every panel is the SAME fixed 2D latent space (identical axes) at a different
time frame -> spatial geometry is consistent and interpretable across time.
Per panel: review-weighted heat field + OT flow arrows showing where that
frame's heat moves to the next frame. Fixed genre anchors (x) prove the
coordinates don't move.

Usage: python heat_flow_smallmultiples.py
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(7)
BASE = r"C:\Users\admin\AppData\Local\Temp\claude\C--Users-admin-Documents-eval-array-latent\2169e11b-e93d-4912-ae86-dbb6eeb74343\scratchpad"

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

G, (XMIN, XMAX, YMIN, YMAX) = 160, (-1.0, 11.0, -1.0, 11.0)
xs = np.linspace(XMIN, XMAX, G); ys = np.linspace(YMIN, YMAX, G)
CELL = xs[1] - xs[0]

def field(pts, ws):
    h = np.zeros((G, G))
    ix = np.clip(((pts[:, 0]-XMIN)/CELL).astype(int), 0, G-1)
    iy = np.clip(((pts[:, 1]-YMIN)/CELL).astype(int), 0, G-1)
    np.add.at(h, (iy, ix), ws)
    return gaussian_filter(h, 6.0, mode="constant") / max(1e-12, ws.sum())

H = [field(*g) for g in GAMES]
gmax = max(h.max() for h in H)

# ---- OT grid velocity fields (frame t -> t+1) ----
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
    return aU/(aW+1e-9), aV/(aW+1e-9), aW

flows = [splat(GAMES[t][0], ot_disp(GAMES[t][0], GAMES[t][1], GAMES[t+1][0], GAMES[t+1][1]),
               GAMES[t][1]) for t in range(NT-1)]

# ---- render: one row, identical axes per panel ----
fig, axes = plt.subplots(1, NT, figsize=(19, 4.2), sharex=True, sharey=True)
step = 14
gx, gy = np.meshgrid(xs[::step], ys[::step])
for t, ax in enumerate(axes):
    ax.imshow(H[t], origin="lower", extent=[XMIN, XMAX, YMIN, YMAX], cmap="magma",
              vmin=0, vmax=gmax, aspect="equal")
    if t < NT-1:
        U, V, W = flows[t]
        Us = U[::step, ::step]; Vs = V[::step, ::step]; Ws = W[::step, ::step]
        m = Ws > Ws.max()*0.04
        ax.quiver(gx[m], gy[m], Us[m], Vs[m], color="cyan", alpha=0.8,
                  scale=1, scale_units="xy", angles="xy", width=0.012)
    # fixed genre anchors (same coords every panel = spatial consistency)
    ax.scatter(GC[:, 0], GC[:, 1], s=22, marker="x", c="white", lw=1.2, alpha=0.85)
    ax.set_title(FRAMES[t], fontsize=10)
    ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX)
    ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 5, 10])
    ax.tick_params(labelsize=7)

# label genres once, on the first panel
for nm, (cx, cy) in GENRES.items():
    axes[0].annotate(nm.split("/")[0], (cx, cy), fontsize=7, color="white",
                     ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
axes[0].set_ylabel("latent dim 2", fontsize=10)
for ax in axes: ax.set_xlabel("latent dim 1", fontsize=9)
fig.suptitle("Heat over a FIXED latent space, time left->right  (SYNTHETIC small-multiples)  "
             "-  same axes every panel = spatial consistency;  cyan = OT flow to next frame",
             fontsize=12, y=1.02)
fig.tight_layout()
out = BASE + r"\heat_flow_smallmultiples.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)
