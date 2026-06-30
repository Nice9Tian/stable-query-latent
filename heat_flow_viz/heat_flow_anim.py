"""Displacement-interpolation animation of the (fake) heat-transfer process.

Reuses the synthetic data + game-level OT from the prototype, then renders
McCann (optimal-transport) interpolation between consecutive frames so the heat
visibly *flows* Early -> 2024. Emits an animated GIF and a filmstrip PNG.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(7)
BASE = r"C:\Users\admin\AppData\Local\Temp\claude\C--Users-admin-Documents-eval-array-latent\2169e11b-e93d-4912-ae86-dbb6eeb74343\scratchpad"
GIF = BASE + r"\heat_flow_anim.gif"
STRIP = BASE + r"\heat_flow_filmstrip.png"

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

G, (XMIN, XMAX, YMIN, YMAX) = 200, (-1.0, 11.0, -1.0, 11.0)
xs = np.linspace(XMIN, XMAX, G); CELL = xs[1] - xs[0]

def field(pts, ws):
    h = np.zeros((G, G))
    ix = np.clip(((pts[:, 0]-XMIN)/CELL).astype(int), 0, G-1)
    iy = np.clip(((pts[:, 1]-YMIN)/CELL).astype(int), 0, G-1)
    np.add.at(h, (iy, ix), ws)
    h = gaussian_filter(h, 7.0, mode="constant")
    return h / h.sum()

def sinkhorn(a, b, C, eps, it=400):
    K = np.exp(-C/eps); u = np.ones_like(a); v = np.ones_like(b)
    for _ in range(it):
        u = a/(K@v+1e-300); v = b/(K.T@u+1e-300)
    return (u[:, None]*K)*v[None, :]

def ot_disp(X1, a, X2, b):
    C = ((X1[:, None]-X2[None])**2).sum(-1); eps = 0.3*np.median(C)
    P = sinkhorn(a/a.sum(), b/b.sum(), C, eps)
    return (P@X2)/(P.sum(1, keepdims=True)+1e-300) - X1

DISP = [ot_disp(GAMES[t][0], GAMES[t][1], GAMES[t+1][0], GAMES[t+1][1])
        for t in range(5)]

# build interpolated frames: per transition, march source games along OT disp
SUB = 14
interp = []          # (label_time, heat_field, current_points, current_disp_for_quiver)
for t in range(5):
    X1, a = GAMES[t]; d = DISP[t]
    for k in range(SUB):
        s = k / SUB
        P = X1 + s * d
        interp.append(((t + s), field(P, a), P, a, d))
interp.append((5.0, field(*GAMES[5]), GAMES[5][0], GAMES[5][1], np.zeros_like(GAMES[5][0])))

gmax = max(f[1].max() for f in interp)

def tlabel(tt):
    lo = int(np.floor(min(tt, 4.999))); frac = tt - lo
    return f"{FRAMES[lo]}  ->  {FRAMES[min(lo+1,5)]}   ({frac*100:4.0f}%)" if frac > 1e-6 else FRAMES[int(round(tt))]

# ---- filmstrip (8 snapshots) ----
pick = np.linspace(0, len(interp)-1, 8).astype(int)
fig, axes = plt.subplots(1, 8, figsize=(20, 3.0))
for ax, idx in zip(axes, pick):
    tt, h, P, w, d = interp[idx]
    ax.imshow(h, origin="lower", extent=[XMIN, XMAX, YMIN, YMAX], cmap="magma", vmin=0, vmax=gmax)
    ax.set_title(tlabel(tt), fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    for nm, (gx, gy) in GENRES.items():
        ax.text(gx, gy, nm.split("/")[0], fontsize=6, ha="center", color="cyan", alpha=0.7)
fig.suptitle("Displacement-interpolation filmstrip (SYNTHETIC): heat flows Early -> 2024", fontsize=12)
fig.savefig(STRIP, dpi=110, bbox_inches="tight"); plt.close(fig)
print("wrote", STRIP)

# ---- GIF ----
figA, axA = plt.subplots(figsize=(6.5, 6))
def draw(i):
    axA.clear()
    tt, h, P, w, d = interp[i]
    axA.imshow(h, origin="lower", extent=[XMIN, XMAX, YMIN, YMAX], cmap="magma", vmin=0, vmax=gmax)
    if np.any(d):
        sk = slice(None, None, 7)
        axA.quiver(P[sk, 0], P[sk, 1], d[sk, 0], d[sk, 1], color="cyan",
                   alpha=0.5, scale=1, scale_units="xy", angles="xy", width=0.004)
    for nm, (gx, gy) in GENRES.items():
        axA.text(gx, gy, nm, fontsize=7, ha="center", color="white",
                 bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4))
    axA.set_title("Heat transfer (SYNTHETIC):  " + tlabel(tt), fontsize=10)
    axA.set_xlim(XMIN, XMAX); axA.set_ylim(YMIN, YMAX); axA.set_xticks([]); axA.set_yticks([])

anim = FuncAnimation(figA, draw, frames=len(interp), interval=90)
anim.save(GIF, writer=PillowWriter(fps=12))
print("wrote", GIF, "frames:", len(interp))
