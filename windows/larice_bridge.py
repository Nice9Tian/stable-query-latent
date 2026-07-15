# -*- coding: utf-8 -*-
"""larice_bridge — the embedded-Python side of the Windows GUI.

Called from C++ (pybind11 embedded interpreter). Contract: every public
function returns a JSON string {"ok": bool, ...} and never raises across
the boundary. Heavy imports (torch/transformers) happen lazily inside
functions so the module imports even before first-run bootstrap.

Inference pipeline (mirrors release/steam_reviews_framework exactly):
  text -> sentence split -> Qwen3-Embedding-0.6B (last-token pool)
       -> rown (per-row mean0/std1) -> champion tower (cegate2, pool readout)
       -> column-standardize by gallery stats (mu/sd)
       -> BackHead-NAME linear (W,b) + L2  = the ANCHOR shown to the user
  games: softmax(anchor @ gallery_head.T * exp(logt))   (2020-way)
  tags : ridge probe  (anchor - sc_mean)/sc_scale @ coef.T + intercept,
         displayed as clip(score,0,1); predicted flag = score >= threshold
"""
import csv
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
RUNTIME_REQS = APP_DIR / "runtime_requirements.txt"
LOG_FILE = APP_DIR / "larice_app.log"

_STATE = {}          # tower, head arrays, embedder, meta — filled by load_model


def _flog(msg):
    """Append to the on-disk log — the GUI has no console, this file is the
    only forensic trail when something hangs or dies inside embedded Python."""
    try:
        import datetime
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.datetime.now():%H:%M:%S}] {msg}\n")
    except Exception:
        pass


def _mklog(progress):
    cb = progress or (lambda s: None)

    def log(s):
        _flog(s)
        try:
            cb(s)
        except Exception:
            pass
    return log


def _fail(msg):
    _flog("FAIL: " + msg)
    return json.dumps({"ok": False, "error": msg}, ensure_ascii=False)


# ----------------------------- bootstrap -----------------------------------

def _bundled_python():
    """python.exe of the bundled runtime (PYTHONHOME when embedded)."""
    for base in (sys.base_prefix, sys.prefix, sys.exec_prefix):
        exe = Path(base) / "python.exe"
        if exe.exists():
            return str(exe)
    return sys.executable


def _deps_missing():
    missing = []
    for mod in ("numpy", "torch", "transformers"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    return missing


def _has_nvidia_gpu():
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                           text=True, timeout=10,
                           creationflags=getattr(subprocess,
                                                 "CREATE_NO_WINDOW", 0))
        return r.returncode == 0 and "GPU" in (r.stdout or "")
    except Exception:
        return False


def bootstrap(progress=None):
    """First run: pip-install the runtime deps into the bundled Python.
    On machines with an NVIDIA GPU, torch comes from the CUDA wheel index
    (default cu128; override with LARICE_TORCH_INDEX).
    progress: callable(str) streamed with pip output lines."""
    log = _mklog(progress)
    try:
        log("bootstrap: checking deps ...")
        missing = _deps_missing()
        if not missing:
            log("dependencies already present -- skipping pip")
            return json.dumps({"ok": True, "installed": []})
        log(f"installing dependencies (first run): {', '.join(missing)} ...")
        cmd = [_bundled_python(), "-m", "pip", "install",
               "--no-warn-script-location", "-r", str(RUNTIME_REQS)]
        extra_index = os.environ.get("LARICE_TORCH_INDEX", "")
        if not extra_index and "torch" in missing and _has_nvidia_gpu():
            extra_index = "https://download.pytorch.org/whl/cu128"
            log("NVIDIA GPU detected -- using the CUDA torch wheel index")
        if extra_index:
            cmd += ["--extra-index-url", extra_index]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                creationflags=getattr(subprocess,
                                                      "CREATE_NO_WINDOW", 0))
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log(line)
        proc.wait()
        if proc.returncode != 0:
            return _fail(f"pip exited with code {proc.returncode}")
        still = _deps_missing()
        if still:
            return _fail("still missing after install: " + ", ".join(still))
        log("dependencies ready")
        return json.dumps({"ok": True, "installed": missing})
    except Exception:
        return _fail(traceback.format_exc())


# ------------------------- champion tower (inference) -----------------------

def _build_tower(state_dict, device):
    """Minimal LariceTower (pool readout) — mirrors release/main_model."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Tower(nn.Module):
        def __init__(self, dm=128, heads=4, din=1024, hidden=256, nq=4):
            super().__init__()
            self.q0 = nn.Parameter(torch.zeros(1, nq, dm))
            self.attn = nn.MultiheadAttention(dm, heads, kdim=din, vdim=din,
                                              batch_first=True)
            self.head = nn.Sequential(nn.Linear(dm, hidden), nn.GELU(),
                                      nn.Linear(hidden, dm))

        def forward(self, x, mask=None):
            a, _ = self.attn(self.q0.expand(x.shape[0], -1, -1),
                             x.float(), x.float(),
                             key_padding_mask=mask, need_weights=False)
            return F.normalize(self.head(a.mean(1)), dim=-1)   # pool readout

    nq, dm = state_dict["q0"].shape[1], state_dict["q0"].shape[2]
    din = state_dict["attn.k_proj_weight"].shape[1] \
        if "attn.k_proj_weight" in state_dict else 1024
    hidden = state_dict["head.0.weight"].shape[0]
    tower = Tower(dm=dm, din=din, hidden=hidden, nq=nq)
    tower.load_state_dict(state_dict, strict=False)   # log_inv_tau et al. ok
    return tower.to(device).eval()


class _QwenEmbedder:
    """Qwen3-Embedding-0.6B, last-token pooling (LocalEmbedder recipe)."""

    MODEL = os.environ.get("LARICE_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")

    def __init__(self, device, max_length=2048, batch_size=16):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL,
                                                       padding_side="left")
        self.model = AutoModel.from_pretrained(self.MODEL).to(device).eval()
        self.device, self.max_length, self.batch_size = device, max_length, batch_size

    def embed(self, texts):
        import torch
        out = []
        for i in range(0, len(texts), self.batch_size):
            tok = self.tokenizer(texts[i:i + self.batch_size], padding=True,
                                 truncation=True, max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
            with torch.no_grad():
                hid = self.model(**tok).last_hidden_state
                am = tok["attention_mask"]
                if am[:, -1].sum() == am.shape[0]:          # left padding
                    vec = hid[:, -1]
                else:
                    idx = am.sum(1) - 1
                    vec = hid[torch.arange(hid.shape[0], device=hid.device), idx]
            out.append(vec.cpu().float())
        return torch.cat(out).numpy()


_SENT_RE = re.compile(r"[^。！？!?\.\n]+[。！？!?\.]?")


def _split_sentences(text, cap=512):
    parts = [m.group(0).strip() for m in _SENT_RE.finditer(text)]
    parts = [p for p in parts if p]
    return parts[:cap] if parts else [text.strip()]


# ------------------------------ model load ---------------------------------

def load_model(assets_dir="", progress=None):
    """Load tower + head + galleries from assets, write games_anchors.csv."""
    log = _mklog(progress)
    try:
        log("load_model: importing numpy ...")
        import numpy as np
        log("load_model: importing torch ...")
        import torch
        log(f"load_model: torch {torch.__version__} imported")

        adir = Path(assets_dir) if assets_dir else APP_DIR / "assets"
        tower_pt = adir / "tower.pt"
        pack_npz = adir / "champion_assets.npz"
        if not tower_pt.exists() or not pack_npz.exists():
            return _fail(
                f"模型资产缺失：需要 {tower_pt.name} 和 {pack_npz.name} 位于\n"
                f"{adir}\n先在训练环境运行 windows/export_assets.py 生成。")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"device: {device}")
        if device == "cpu" and _has_nvidia_gpu():
            log("警告：检测到 NVIDIA GPU 但当前 torch 是 CPU 版，嵌入会很慢。"
                "修复：用内置 python 执行 pip uninstall -y torch && pip install "
                "torch --index-url https://download.pytorch.org/whl/cu128")

        log("loading champion tower (cegate2) ...")
        sd = torch.load(tower_pt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        tower = _build_tower(sd, device)

        log("loading head + galleries ...")
        P = np.load(pack_npz, allow_pickle=True)
        st = {
            "tower": tower, "device": device, "np": np, "torch": torch,
            "W": P["head_W"].astype(np.float32),
            "b": P["head_b"].astype(np.float32),
            "logt": float(P["head_logt"]),
            "mu": P["feat_mu"].astype(np.float32),
            "sd": P["feat_sd"].astype(np.float32) + 1e-6,
            "Hgal": P["gallery_head"].astype(np.float32),   # [NG, D] L2 rows
            "Tgal": P["gallery_tower"].astype(np.float32),  # [NG, Dt]
            "names": [str(n) for n in P["names"]],
            "display": ([str(n) for n in P["display_names"]]
                        if "display_names" in P.files
                        else [str(n) for n in P["names"]]),
            "tag_coef": P["tag_coef"].astype(np.float32),
            "tag_int": P["tag_intercept"].astype(np.float32),
            "tag_mean": P["tag_scaler_mean"].astype(np.float32),
            "tag_scale": P["tag_scaler_scale"].astype(np.float32) + 1e-12,
            "tag_thr": float(P["tag_threshold"]),
            "tag_names": [str(t) for t in P["tag_names"]],
        }

        log("loading Qwen3-Embedding-0.6B (first time downloads from HF) ...")
        st["embedder"] = _QwenEmbedder(device)
        _STATE.update(st)

        csv_path = adir / "games_anchors.csv"
        if not csv_path.exists():
            log("writing games_anchors.csv (all-game anchor IDs) ...")
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                dim = st["Hgal"].shape[1]
                w.writerow(["anchor_id", "name", "title"]
                           + [f"a{i}" for i in range(dim)])
                for i, nm in enumerate(st["names"]):
                    v = st["Hgal"][i]      # head space = same space as session anchors
                    w.writerow([i, nm, st["display"][i]]
                               + [f"{x:.6f}" for x in v])
        log("model ready")
        return json.dumps({"ok": True, "device": device,
                           "n_games": len(st["names"]),
                           "n_tags": len(st["tag_names"]),
                           "anchor_dim": int(st["W"].shape[0]),
                           "games_csv": str(csv_path)}, ensure_ascii=False)
    except Exception:
        return _fail(traceback.format_exc())


# ------------------------------ inference ----------------------------------

def embed_and_predict(text, top_games=20, top_tags=23):
    try:
        if "tower" not in _STATE:
            return _fail("模型尚未加载")
        np, torch = _STATE["np"], _STATE["torch"]
        text = (text or "").strip()
        if not text:
            return _fail("描述为空")

        sents = _split_sentences(text)
        _flog(f"embed: {len(sents)} sentence(s), qwen forward ...")
        E = _STATE["embedder"].embed(sents)                    # [S, 1024]
        _flog("embed: qwen done, tower forward ...")
        m = E.mean(-1, keepdims=True)
        s = E.std(-1, keepdims=True)
        E = (E - m) / (s + 1e-6)                               # rown
        with torch.no_grad():
            x = torch.tensor(E, dtype=torch.float32,
                             device=_STATE["device"])[None]
            t = _STATE["tower"](x)[0].cpu().numpy()            # [Dt] L2

        _flog("embed: tower done, head + predictions ...")
        xs = (t - _STATE["mu"]) / _STATE["sd"]                 # column-std
        h = xs @ _STATE["W"].T + _STATE["b"]
        h = h / (np.linalg.norm(h) + 1e-8)                     # the ANCHOR

        sim = _STATE["Hgal"] @ h                               # [NG]
        logits = sim * np.exp(_STATE["logt"])
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = np.argsort(-p)[:int(top_games)]
        games = [[_STATE["display"][i], float(p[i])] for i in order]

        z = (h - _STATE["tag_mean"]) / _STATE["tag_scale"]
        ts = z @ _STATE["tag_coef"].T + _STATE["tag_int"]      # [n_tags]
        torder = np.argsort(-ts)[:int(top_tags)]
        tags = [[_STATE["tag_names"][i],
                 float(np.clip(ts[i], 0.0, 1.0)),
                 bool(ts[i] >= _STATE["tag_thr"])] for i in torder]

        _flog("embed: done")
        return json.dumps({"ok": True,
                           "anchor": [float(v) for v in h],
                           "n_sentences": len(sents),
                           "games": games, "tags": tags}, ensure_ascii=False)
    except Exception:
        return _fail(traceback.format_exc())


# ------------------------------- export ------------------------------------

def export_anchors_json(rows_json, path):
    """rows_json: JSON [[name, [dims...]], ...]; writes {name: [dims...]}.
    Duplicate names get a _2/_3... suffix so no anchor is silently lost."""
    try:
        rows = json.loads(rows_json)
        if not rows:
            return _fail("nothing to export")
        out, seen = {}, {}
        for name, vec in rows:
            n = seen.get(name, 0) + 1
            seen[name] = n
            out[name if n == 1 else f"{name}_{n}"] = vec
        tmp = Path(path).with_name(Path(path).name + ".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(path)
        return json.dumps({"ok": True, "path": str(path), "rows": len(rows)},
                          ensure_ascii=False)
    except Exception:
        return _fail(traceback.format_exc())


def export_anchors_csv(rows_json, path):
    """rows_json: JSON [[name, [dims...]], ...]; writes name + a0..aN."""
    try:
        rows = json.loads(rows_json)
        if not rows:
            return _fail("没有可导出的锚点")
        dim = len(rows[0][1])
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["name"] + [f"a{i}" for i in range(dim)])
            for name, vec in rows:
                w.writerow([name] + [f"{v:.6f}" for v in vec])
        return json.dumps({"ok": True, "path": path, "rows": len(rows)},
                          ensure_ascii=False)
    except Exception:
        return _fail(traceback.format_exc())
