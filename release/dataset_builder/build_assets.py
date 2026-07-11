# -*- coding: utf-8 -*-
"""Build every training/eval tensor asset from the corpora + review h5.

Steps (each resume-safe — existing outputs are skipped):
  1. games.npz            game-name universe (order = embedding_h5 order;
                          the gidx basis for EVERY other asset)
  2. *_views.npz          full-text doc views: wiki_clean / wiki_llm / sp_raw
                          (SaT sentence split, drop <10-char fragments,
                          NO sentence cap, embed everything)
  3. pool + anchors       review pool (2048/game, gold guarantee = 3 longest
                          reviews), anchors (sp doc prefix + whole reviews
                          @512), pseudo-queries (anchor-shaped, 4/game, flat)
  4. wiki_eval.npz        eval queries: wiki 4-variant docs, COMPLETE text

Usage:  python dataset_builder/build_assets.py [--step 1 2 3 4]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "reviews"))

from dataset_builder.paths import ASSETS, CORPORA, EMBED_H5, SPLIT_JSON

SEED = 20260711
CAP, TOPK = 2048, 3          # review pool budget / gold-guarantee count
GCAP = 512                   # anchor budget (doc prefix + whole reviews)
QCAP, QPG = 512, 4           # pseudo-queries: anchor-shaped, 4 per game
VARIANTS = ("neutral", "noname", "positive", "negative")


# ---------------------------------------------------------------- step 1 --
def build_games():
    out = ASSETS / "games.npz"
    if out.exists():
        print("games.npz exists — skip", flush=True)
        return
    import h5py
    with h5py.File(EMBED_H5, "r") as h:
        names = [g.decode() if isinstance(g, bytes) else str(g)
                 for g in h["game_names"][:]]
    ASSETS.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.array(names, object))
    print(f"games.npz: {len(names)} games", flush=True)


# ---------------------------------------------------------------- step 2 --
def build_views(jobs=(("wiki_clean", "wiki_clean_views.npz"),
                      ("wiki_llm", "wiki_llm_views.npz"),
                      ("sp_raw", "sp_raw_views.npz"))):
    todo = [(s, o) for s, o in jobs if not (ASSETS / o).exists()]
    if not todo:
        print("all views npz exist — skip", flush=True)
        return
    from embedding_data import DEFAULT_LOCAL_MODEL, LocalEmbedder
    from wtpsplit import SaT
    G = np.load(ASSETS / "games.npz", allow_pickle=True)
    gnames = [str(x) for x in G["names"]]
    appid2gidx = {n.split("_")[0]: i for i, n in enumerate(gnames)}
    sat = SaT("sat-3l-sm")
    try:
        sat.half().to("cuda")
        print("splitter: SaT (cuda)", flush=True)
    except Exception:
        print("splitter: SaT (cpu)", flush=True)
    emb = LocalEmbedder(DEFAULT_LOCAL_MODEL, device="cuda", batch_size=64)

    def list_files(src_name):
        if "/" in src_name:                     # variants-dir layout
            d, v = src_name.split("/", 1)
            return sorted(((x.name, x / f"{v}.txt")
                           for x in (CORPORA / d).iterdir()
                           if x.is_dir() and (x / f"{v}.txt").exists()
                           and x.name in appid2gidx), key=lambda p: p[0])
        return sorted(((f.stem.split("_")[0], f)
                       for f in (CORPORA / src_name).glob("*.txt")
                       if f.stem.split("_")[0] in appid2gidx),
                      key=lambda p: p[0])

    for src_name, out_name in todo:
        pairs = list_files(src_name)
        texts = [p[1].read_text(encoding="utf-8", errors="ignore")
                 for p in pairs]
        sls = [[s.strip() for s in sl if len(s.strip()) >= 10]
               for sl in sat.split(texts)]
        keep = [i for i, sl in enumerate(sls) if len(sl) >= 2]
        lens = [len(sls[i]) for i in keep]
        MAXLEN = max(lens)
        print(f"{src_name}: {len(pairs)} docs, {len(keep)} usable, "
              f"{sum(lens)} sentences, max {MAXLEN}", flush=True)
        flat = [s for i in keep for s in sls[i]]
        vecs = np.asarray(emb.embed(flat), dtype=np.float16)
        S = np.zeros((len(keep), MAXLEN, vecs.shape[1]), np.float16)
        S_len = np.zeros(len(keep), np.int32)
        gidx, names = [], []
        pos = 0
        for row, i in enumerate(keep):
            k = len(sls[i])
            S[row, :k] = vecs[pos:pos + k]
            S_len[row] = k
            pos += k
            gi = appid2gidx[pairs[i][0]]
            gidx.append(gi)
            names.append(gnames[gi])
        np.savez(ASSETS / out_name, S=S, S_len=S_len,
                 gidx=np.array(gidx, np.int64),
                 names=np.array(names, object))
        print(f"saved {out_name}: S{S.shape}", flush=True)


# ---------------------------------------------------------------- step 3 --
def build_pool_and_anchors():
    if (ASSETS / "ss_queries_rev.npz").exists():
        print("pool/anchors/queries exist — skip", flush=True)
        return
    import h5py
    rng = np.random.default_rng(SEED)
    G = np.load(ASSETS / "games.npz", allow_pickle=True)
    names = [str(x) for x in G["names"]]
    NG = len(names)
    SPV = np.load(ASSETS / "sp_raw_views.npz", allow_pickle=True)
    sp_row = {int(SPV["gidx"][i]): i for i in range(len(SPV["gidx"]))}
    SPV_S, SPV_len = SPV["S"], SPV["S_len"]
    gal = np.zeros((NG, GCAP, 1024), np.float16)
    gal_len = np.zeros(NG, np.int32)
    gal_doc_len = np.zeros(NG, np.int32)     # doc prefix length (0 = no doc)
    with h5py.File(EMBED_H5, "r") as h:
        egn = [g.decode() if isinstance(g, bytes) else str(g)
               for g in h["game_names"][:]]
        e2i = {n: i for i, n in enumerate(egn)}
        gro = h["game_review_offsets"][:]
        ro = h["review_offsets"][:]
        vecs = h["vectors"]
        pool = np.zeros((NG, CAP, 1024), np.float16)
        rid = np.full((NG, CAP), -1, np.int32)
        plen = np.zeros(NG, np.int32)
        for gi, n in enumerate(names):
            i = e2i[n]
            r0, r1 = int(gro[i]), int(gro[i + 1])
            starts = ro[r0:r1]
            lens = (ro[r0 + 1:r1 + 1] - starts).astype(np.int64)
            order_long = np.argsort(-lens)
            sel, used = [], 0
            for j in order_long[:TOPK]:                  # gold guarantee
                if used + int(lens[j]) <= CAP:
                    sel.append(int(j))
                    used += int(lens[j])
            rest = rng.permutation(np.setdiff1d(np.arange(len(lens)), sel))
            for j in rest:                               # whole-review fill
                if used + lens[j] <= CAP:
                    sel.append(int(j))
                    used += int(lens[j])
            row = 0                       # sorted segments = fast h5 reads;
            for k, j in enumerate(sorted(sel, key=lambda j: int(starts[j]))):
                s, L = int(starts[j]), int(lens[j])
                blk = vecs[s:s + L].astype(np.float32)   # per-row mean0/std1
                blk = (blk - blk.mean(-1, keepdims=True)) / \
                    (blk.std(-1, keepdims=True) + 1e-6)
                pool[gi, row:row + L] = blk.astype(np.float16)
                rid[gi, row:row + L] = k
                row += L
            plen[gi] = row
            # anchor: sp doc first, whole reviews after, to the 512 budget
            g_row = 0
            if gi in sp_row:
                dl = int(SPV_len[sp_row[gi]])
                gal[gi, :dl] = SPV_S[sp_row[gi], :dl]
                g_row = dl
                gal_doc_len[gi] = dl
            r_lens = np.bincount(rid[gi, :row])
            r_starts = np.zeros(len(r_lens), np.int64)
            r_starts[1:] = np.cumsum(r_lens)[:-1]
            for j in rng.permutation(len(r_lens)):
                L = int(r_lens[j])
                if g_row + L <= GCAP:
                    gal[gi, g_row:g_row + L] = pool[gi, r_starts[j]:r_starts[j] + L]
                    g_row += L
            gal_len[gi] = g_row
            if gi % 100 == 0:
                print(f"  game {gi}/{NG}: {row} sents, anchor {g_row}",
                      flush=True)
    np.save(ASSETS / "wscan_pool_rev.npy", pool)
    np.save(ASSETS / "wscan_pool_rev_rid.npy", rid)
    np.save(ASSETS / "wscan_pool_rev_len.npy", plen)
    np.savez(ASSETS / "wscan_gal_rev.npz", gal=gal, gal_len=gal_len,
             gal_doc_len=gal_doc_len)
    print(f"anchors: doc-bearing {sum(1 for g in range(NG) if g in sp_row)}"
          f"/{NG}", flush=True)

    # pseudo-queries: anchor-shaped (doc prefix + whole reviews @512), each
    # a fresh independent draw = a "second view" of the anchor; flat-stored.
    rngq = np.random.default_rng(SEED + 1)
    Sq, gq = [], []
    for g in range(NG):
        r = rid[g]
        nrev_g = int(r.max()) + 1
        lens_g = np.bincount(r[r >= 0], minlength=nrev_g).astype(np.int64)
        starts_g = np.zeros(nrev_g, np.int64)
        starts_g[1:] = np.cumsum(lens_g)[:-1]
        has_sp = g in sp_row
        dl = int(SPV_len[sp_row[g]]) if has_sp else 0
        for _ in range(QPG):
            parts = [SPV_S[sp_row[g], :dl]] if has_sp else []
            used = dl
            for j in rngq.permutation(len(lens_g)):
                L = int(lens_g[j])
                if used + L <= QCAP:
                    parts.append(pool[g, starts_g[j]:starts_g[j] + L])
                    used += L
            Sq.append(np.concatenate(parts))
            gq.append(g)
    Lq = [len(b) for b in Sq]
    off = np.zeros(len(Sq) + 1, np.int64)
    off[1:] = np.cumsum(Lq)
    np.save(ASSETS / "ss_queries_rev_S.npy",
            np.concatenate(Sq).astype(np.float16))
    np.savez(ASSETS / "ss_queries_rev.npz", off=off,
             gidx=np.array(gq, np.int64))
    print(f"pseudo-queries: {len(Sq)} ({off[-1]} sents flat)", flush=True)


# ---------------------------------------------------------------- step 4 --
def build_wiki_eval():
    out = ASSETS / "wiki_eval.npz"
    if out.exists():
        print("wiki_eval.npz exists — skip", flush=True)
        return
    import json
    from embedding_data import DEFAULT_LOCAL_MODEL, LocalEmbedder
    from wtpsplit import SaT
    G = np.load(ASSETS / "games.npz", allow_pickle=True)
    gnames = [str(x) for x in G["names"]]
    appid2gidx = {n.split("_")[0]: i for i, n in enumerate(gnames)}
    split = json.loads(SPLIT_JSON.read_text())
    universe = set(split["test"] + split["val"] + split["train"])
    VAR_DIR = CORPORA / "wiki_variants"
    games = []
    for d in sorted(VAR_DIR.iterdir()):
        if not d.is_dir() or d.name not in appid2gidx or d.name not in universe:
            continue
        texts = {}
        for v in VARIANTS:
            f = d / f"{v}.txt"
            if f.exists():
                t = f.read_text(encoding="utf-8", errors="ignore").strip()
                if len(t) >= 300:
                    texts[v] = t
        if len(texts) == len(VARIANTS):
            games.append((d.name, appid2gidx[d.name], texts))
    missing = universe - {g[0] for g in games}
    assert not missing, (f"{len(missing)} split games lack complete variants "
                         f"(corpora incomplete): {sorted(missing)[:5]}...")
    print(f"wiki-eval games: {len(games)}", flush=True)
    docs, meta = [], []
    for appid, gidx, texts in games:
        for v in VARIANTS:
            docs.append(texts[v])
            meta.append((gidx, gnames[gidx], v))
    sat = SaT("sat-3l-sm")
    try:
        sat.half().to("cuda")
    except Exception:
        pass
    # COMPLETE content: every non-empty sentence kept, no cap
    sent_lists = [[s.strip() for s in sl if s.strip()]
                  for sl in sat.split(docs)]
    lens = [len(s) for s in sent_lists]
    MAXLEN = max(lens)
    print(f"query docs {len(docs)}: {sum(lens)} sentences, max {MAXLEN}",
          flush=True)
    emb = LocalEmbedder(DEFAULT_LOCAL_MODEL, device="cuda", batch_size=64)
    flat = [s for sl in sent_lists for s in sl]
    vecs = np.asarray(emb.embed(flat), dtype=np.float16)
    S = np.zeros((len(docs), MAXLEN, vecs.shape[1]), np.float16)
    S_len = np.zeros(len(docs), np.int32)
    pos = 0
    for i, sl in enumerate(sent_lists):
        k = len(sl)
        S[i, :k] = vecs[pos:pos + k]
        S_len[i] = k
        pos += k
    np.savez(out, S=S, S_len=S_len,
             gidx=np.array([m[0] for m in meta], np.int64),
             names=np.array([m[1] for m in meta], object),
             variants=np.array([m[2] for m in meta], object))
    print("saved wiki_eval.npz", flush=True)


STEPS = {1: build_games, 2: build_views, 3: build_pool_and_anchors,
         4: build_wiki_eval}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", nargs="*", type=int, default=[1, 2, 3, 4])
    a = ap.parse_args()
    for s in a.step:
        STEPS[s]()
    print("build_assets done", flush=True)
