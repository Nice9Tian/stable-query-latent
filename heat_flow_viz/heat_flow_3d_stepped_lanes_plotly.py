"""Interactive 3D stepped swimlanes (SYNTHETIC) -> rotatable/hover HTML.

Columns stand at their TRUE latent (x,y); z = time steps; box footprint ~ heat.
OT transfer as lines between columns across boundaries. Open the HTML and rotate.

Usage: python heat_flow_3d_stepped_lanes_plotly.py [N_LANES]
"""
from __future__ import annotations
import sys
import numpy as np
import plotly.graph_objects as go
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

GAMES = [make(fi) for fi in range(6)]; NT = len(GAMES)
allp = np.vstack([g[0] for g in GAMES]); allw = np.concatenate([g[1] for g in GAMES])
samp = allp[rng.choice(len(allw), size=8000, p=allw/allw.sum())]
LC, _ = kmeans2(samp, NL, seed=3, minit="++", missing="raise")
def lane_of(P): return np.argmin(((P[:, None]-LC[None])**2).sum(-1), axis=1)
LAB = [lane_of(g[0]) for g in GAMES]
LANE_NAME = [GN[np.argmin(((LC[l]-GC)**2).sum(1))] for l in range(NL)]
heat = np.array([[GAMES[t][1][LAB[t] == l].sum() for t in range(NT)] for l in range(NL)])
hmax = heat.max()

def sinkhorn(a, b, C, eps, it=400):
    K = np.exp(-C/eps); u = np.ones_like(a); v = np.ones_like(b)
    for _ in range(it):
        u = a/(K@v+1e-300); v = b/(K.T@u+1e-300)
    return (u[:, None]*K)*v[None, :]
def onehot(lab, n):
    M = np.zeros((len(lab), n)); M[np.arange(len(lab)), lab] = 1; return M
Mt = [onehot(LAB[t], NL).T @ sinkhorn(GAMES[t][1]/GAMES[t][1].sum(), GAMES[t+1][1]/GAMES[t+1][1].sum(),
       ((GAMES[t][0][:, None]-GAMES[t+1][0][None])**2).sum(-1),
       0.3*np.median(((GAMES[t][0][:, None]-GAMES[t+1][0][None])**2).sum(-1))) @ onehot(LAB[t+1], NL)
      for t in range(NT-1)]

PAL = ["#d62728", "#ff7f0e", "#7fbf2f", "#17becf", "#3b3b9b", "#9467bd", "#e377c2", "#8c564b"]
ZSTEP, DZ, SMAX = 1.0, 0.72, 2.6

def box(cx, cy, s, z0, color, name, hover):
    x0, x1, y0, y1, z1 = cx-s/2, cx+s/2, cy-s/2, cy+s/2, z0+DZ
    X = [x0, x1, x1, x0, x0, x1, x1, x0]; Y = [y0, y0, y1, y1, y0, y0, y1, y1]
    Z = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 4, 4, 0, 0, 1, 1, 3, 3, 0, 0]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 2, 6, 3, 7]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 6, 7, 7, 4]
    return go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k, color=color, opacity=1.0,
                     flatshading=True, name=name, hovertext=hover, hoverinfo="text",
                     showscale=False)

fig = go.Figure()
for l in range(NL):
    cx, cy = LC[l]; col = PAL[l % len(PAL)]
    fig.add_trace(go.Scatter3d(x=[cx, cx], y=[cy, cy], z=[0, (NT-1)*ZSTEP+DZ],
                  mode="lines", line=dict(color="lightgrey", width=2), hoverinfo="skip",
                  showlegend=False))
    for t in range(NT):
        s = SMAX*np.sqrt(heat[l, t]/hmax)
        if s < 1e-3: continue
        fig.add_trace(box(cx, cy, s, t*ZSTEP, col, LANE_NAME[l],
                          f"{LANE_NAME[l]}<br>{FRAMES[t]}<br>heat={heat[l,t]:.3f}"))
    fig.add_trace(go.Scatter3d(x=[cx], y=[cy], z=[(NT-1)*ZSTEP+DZ+0.4], mode="text",
                  text=[LANE_NAME[l]], textfont=dict(size=12, color=col),
                  hoverinfo="skip", showlegend=False))

for t in range(NT-1):
    M = Mt[t]; mmax = M.max()
    for i in range(NL):
        for j in range(NL):
            if i == j: continue
            frac = M[i, j]/mmax
            if frac < 0.12: continue
            xi, yi = LC[i]; xj, yj = LC[j]
            fig.add_trace(go.Scatter3d(
                x=[xi, xj], y=[yi, yj], z=[t*ZSTEP+DZ, (t+1)*ZSTEP],
                mode="lines", line=dict(color=PAL[i % len(PAL)], width=1+8*frac),
                opacity=min(0.9, 0.3+frac),
                hovertext=f"{LANE_NAME[i]} -> {LANE_NAME[j]}<br>{FRAMES[t]}->{FRAMES[t+1]}"
                          f"<br>moved={M[i,j]:.3f}", hoverinfo="text", showlegend=False))

fig.update_layout(
    title=f"Interactive 3D stepped swimlanes (SYNTHETIC): columns at true latent positions, "
          f"z=time - {NL} lanes",
    scene=dict(xaxis_title="latent dim 1", yaxis_title="latent dim 2",
               zaxis=dict(title="time", tickvals=[t*ZSTEP for t in range(NT)], ticktext=FRAMES),
               aspectmode="manual", aspectratio=dict(x=1, y=1, z=1.5),
               camera=dict(eye=dict(x=1.7, y=1.7, z=0.9))),
    margin=dict(l=0, r=0, t=40, b=0), showlegend=False)

out = BASE + rf"\heat_flow_3d_stepped_lanes_interactive.html"
fig.write_html(out, include_plotlyjs="cdn")
print("wrote", out)   # NOTE: no write_image (kaleido hangs on Windows)
