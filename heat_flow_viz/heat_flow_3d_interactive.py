"""Interactive 3D heat-transfer (SYNTHETIC) for post-hoc interpretation.

Full 2D latent plane is preserved (NOT collapsed) so regions stay interpretable.
x,y = VICReg-2D latent ; z = time. Heat slices per frame + OT worldlines.
Rotate / zoom / hover in the browser. Hover a game point to read what it is
(real data: title / TAG / release date / review count).

Usage: python heat_flow_3d_interactive.py [N_LINES] [MIN_SEP]
Output: heat_flow_3d_interactive.html  (+ .png snapshot if kaleido present)
"""
from __future__ import annotations
import sys
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(7)
BASE = r"C:\Users\admin\AppData\Local\Temp\claude\C--Users-admin-Documents-eval-array-latent\2169e11b-e93d-4912-ae86-dbb6eeb74343\scratchpad"
N_LINES = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MIN_SEP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9

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
    pts, ws, gi_id = [], [], []
    for gi in range(len(GC)):
        n = max(1, int(round(N_GAMES[fi] * MASS[fi, gi])))
        pts.append(rng.normal(GC[gi], 0.85, size=(n, 2)))
        w = rng.lognormal(0, 0.8, size=n); ws.append(w / w.sum() * MASS[fi, gi])
        gi_id.append(np.full(n, gi))
    return np.vstack(pts), np.concatenate(ws), np.concatenate(gi_id)

GAMES = [make(fi) for fi in range(6)]
NT = len(GAMES)

G, (XMIN, XMAX, YMIN, YMAX) = 120, (-1.0, 11.0, -1.0, 11.0)
xs = np.linspace(XMIN, XMAX, G); ys = np.linspace(YMIN, YMAX, G)
CELL = xs[1] - xs[0]; XX, YY = np.meshgrid(xs, ys)

def field(pts, ws):
    h = np.zeros((G, G))
    ix = np.clip(((pts[:, 0]-XMIN)/CELL).astype(int), 0, G-1)
    iy = np.clip(((pts[:, 1]-YMIN)/CELL).astype(int), 0, G-1)
    np.add.at(h, (iy, ix), ws)
    return gaussian_filter(h, 5.0, mode="constant") / max(1e-12, ws.sum())

H = [field(g[0], g[1]) for g in GAMES]
gmax = max(h.max() for h in H)

# ---- OT velocity fields ----
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
    aU = gaussian_filter(aU, 7.0); aV = gaussian_filter(aV, 7.0); aW = gaussian_filter(aW, 7.0)
    return aU/(aW+1e-9), aV/(aW+1e-9)

flows = [splat(GAMES[t][0], ot_disp(GAMES[t][0], GAMES[t][1], GAMES[t+1][0], GAMES[t+1][1]),
               GAMES[t][1]) for t in range(NT-1)]

def sample(F, x, y):
    fx = np.clip((x-XMIN)/CELL, 0, G-1.001); fy = np.clip((y-YMIN)/CELL, 0, G-1.001)
    x0 = int(fx); y0 = int(fy); dx = fx-x0; dy = fy-y0
    return (F[y0, x0]*(1-dx)*(1-dy)+F[y0, x0+1]*dx*(1-dy)
            + F[y0+1, x0]*(1-dx)*dy+F[y0+1, x0+1]*dx*dy)

# ---- seed lines: greedy heat + redundancy pruning (matches user's 3d edit) ----
ep, ew, _ = GAMES[0]
order = np.argsort(ew)[::-1]; kept = []
for i in order:
    if all(np.hypot(*(ep[i]-ep[j])) >= MIN_SEP for j in kept): kept.append(i)
    if len(kept) >= N_LINES: break
seeds = ep[kept]; seed_w = ew[kept]; seed_w = seed_w/seed_w.sum()
print(f"pruned: kept {len(kept)}/{len(ew)} lines (N={N_LINES}, sep={MIN_SEP})")

K = 14
paths = []
for s in seeds:
    pos = s.copy(); rec = [(pos[0], pos[1], 0.0)]
    for ti, (U, V) in enumerate(flows):
        for k in range(K):
            step = np.array([sample(U, pos[0], pos[1]), sample(V, pos[0], pos[1])])/K
            n = np.hypot(*step)
            if n > 0.6: step *= 0.6/n
            pos = pos + step
            rec.append((pos[0], pos[1], ti+(k+1)/K))
    paths.append(np.array(rec))

# ---------------------------------------------------------------- plotly
fig = go.Figure()

# heat slices (semi-transparent sheets at z = frame)
for t in range(NT):
    fig.add_trace(go.Surface(
        x=XX, y=YY, z=np.full_like(XX, float(t)), surfacecolor=H[t],
        colorscale="Magma", cmin=0, cmax=gmax, opacity=0.55, showscale=False,
        name=FRAMES[t], hoverinfo="skip", contours_z=dict(show=False)))

# faint game points (hover = interpretation; real data -> title/TAG/date)
for t in range(NT):
    P, w, gid = GAMES[t]
    fig.add_trace(go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=np.full(len(P), float(t)), mode="markers",
        marker=dict(size=2.2, color="white", opacity=0.18),
        text=[f"{GN[g]}<br>{FRAMES[t]}<br>heat={wi:.4f}" for g, wi in zip(gid, w)],
        hoverinfo="text", showlegend=False))

# OT worldlines, width ~ corridor heat, color = time
wmax = seed_w.max()
for ci, (p, w) in enumerate(zip(paths, seed_w)):
    fig.add_trace(go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
        line=dict(color=p[:, 2], colorscale="Plasma", cmin=0, cmax=NT-1,
                  width=3 + 11*(w/wmax),
                  colorbar=dict(title="time", tickvals=list(range(NT)),
                                ticktext=FRAMES, len=0.6) if ci == 0 else None,
                  showscale=(ci == 0)),
        name=f"corridor {ci} (heat {w:.3f})",
        hovertext=f"corridor {ci}<br>heat share {w:.3f}", hoverinfo="name"))

# genre anchor labels at the top time level
fig.add_trace(go.Scatter3d(
    x=GC[:, 0], y=GC[:, 1], z=np.full(len(GC), float(NT-1)+0.15), mode="text",
    text=GN, textfont=dict(size=12, color="navy"), showlegend=False, hoverinfo="skip"))

fig.update_layout(
    title=f"3D heat transfer (SYNTHETIC): full 2D latent + time axis - {N_LINES} OT corridors",
    scene=dict(
        xaxis_title="latent dim 1", yaxis_title="latent dim 2",
        zaxis=dict(title="time", tickvals=list(range(NT)), ticktext=FRAMES),
        aspectmode="manual", aspectratio=dict(x=1, y=1, z=1.7),
        camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))),
    margin=dict(l=0, r=0, t=40, b=0), showlegend=False)

html = BASE + r"\heat_flow_3d_interactive.html"
fig.write_html(html, include_plotlyjs="cdn")
print("wrote", html)
# NOTE: do not use fig.write_image() here -- kaleido 0.2.1 hangs on Windows.
# Open the HTML in a browser for the interactive view instead.

# ---- compact JSON for an in-chat widget ----
import json
step = 5
hx = [round(v, 2) for v in xs[::step].tolist()]; hy = [round(v, 2) for v in ys[::step].tolist()]
data = {
    "frames": FRAMES,
    "gmax": round(float(gmax), 7), "hx": hx, "hy": hy,
    "heat": [H[t][::step, ::step].round(5).tolist() for t in range(NT)],
    "lines": [{"x": p[::2, 0].round(2).tolist(), "y": p[::2, 1].round(2).tolist(),
               "z": p[::2, 2].round(2).tolist(), "w": round(float(w), 3)}
              for p, w in zip(paths, seed_w)],
    "genres": [{"n": GN[i], "x": float(GC[i, 0]), "y": float(GC[i, 1])} for i in range(len(GN))],
}
pts = []
for t in range(NT):
    P, w, gid = GAMES[t]
    sel = rng.choice(len(P), size=min(45, len(P)), replace=False)
    for i in sel:
        pts.append([round(float(P[i, 0]), 2), round(float(P[i, 1]), 2), t, GN[gid[i]]])
data["points"] = pts
json.dump(data, open(BASE + r"\heat_flow_3d_widget.json", "w"), separators=(",", ":"))
print("wrote widget json:", len(json.dumps(data, separators=(',', ':'))), "bytes")
try:
    fig.write_image(BASE + r"\heat_flow_3d_interactive.png", width=1100, height=900, scale=2)
    print("wrote png snapshot")
except Exception as e:
    print("no static png (kaleido missing):", type(e).__name__)
