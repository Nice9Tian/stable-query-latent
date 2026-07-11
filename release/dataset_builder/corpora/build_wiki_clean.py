# -*- coding: utf-8 -*-
"""Build the FILTERED + MATCH-VALIDATED wiki corpus (user spec):

1. Section filter: keep the lead paragraph + whitelisted game-content sections
   (Gameplay/Plot/Synopsis/Story/Characters/Setting/Premise/Themes/Overview/
   World/Modes/Multiplayer/Features/Content); drop References/External links/
   Reception/Development/Release/... (everything else).
2. Match validation: cosine between the page's embedded lead sentences
   (wiki_views.npz) and every game's 512-sentence anchor centroid — the
   assigned game must rank top-3 among all 2,020, else the page is flagged
   as a suspected mismatch (sequel/franchise confusion) and EXCLUDED.
3. Output: wiki_clean/<same filename>.txt (filtered text, >=300 chars) +
   wiki_clean_manifest.json (per-file: kept sections, chars, match rank/cos,
   verdict) + a printed suspect list.
"""
import json
import re
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR = SCRIPT_DIR
import os as _os
# data/code separation: corpora TEXT lives under the data root, not the repo
SCRIPT_DIR = (Path(_os.environ["LARICE_CORPORA"]) if _os.environ.get("LARICE_CORPORA")
              else Path(__file__).resolve().parents[2] / "data" / "corpora")
SRC = SCRIPT_DIR / "wiki_descriptions"
DST = SCRIPT_DIR / "wiki_clean"
DST.mkdir(exist_ok=True)
SC = (Path(_os.environ["LARICE_ASSETS"]) if _os.environ.get("LARICE_ASSETS")
      else Path(__file__).resolve().parents[2] / "data" / "assets")

KEEP_KEYS = ("gameplay", "plot", "synopsis", "story", "character", "setting",
             "premise", "theme", "overview", "world", "mode", "multiplayer",
             "feature", "content",
             # user additions: keep artistic/scenario/reception content too
             "scenario", "art", "design", "music", "sound", "audio",
             "reception", "critical", "commercial")
MIN_CHARS, RANK_OK, MARGIN_OK = 300, 3, 0.015

HDR = re.compile(r"^(={2,})\s*(.+?)\s*={2,}\s*$")


def norm_title(s: str) -> str:
    s = re.sub(r"\((video game|game|\d{4}[^)]*)\)", " ", s.lower())
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def split_meta(text: str):
    """Strip the collector header from the body. Two header dialects exist:
    Game:/appid:/Source:/URL: lines AND Match score:/Retrieved: lines followed
    by a ---- separator (the second dialect previously leaked into wiki_clean
    on all 844 files — found 2026-07-11)."""
    HK = r"^(Game|appid|Source|URL|Match score|Retrieved):\s*(.*)$"
    lines = text.splitlines()
    meta = {}
    for ln in lines[:10]:
        m = re.match(HK, ln)
        if m:
            meta[m.group(1)] = m.group(2).strip()
        elif ln.strip() and not re.match(r"^-{4,}\s*$", ln):
            break
    body_start = 0
    for j, ln in enumerate(lines[:12]):
        if re.match(HK, ln) or re.match(r"^-{4,}\s*$", ln) or not ln.strip():
            body_start = j + 1
        else:
            break
    return meta, "\n".join(lines[body_start:])


def filter_sections(text: str):
    """Keep lead + whitelisted sections. A level-3+ subsection whose OWN title
    matches the whitelist is kept even if its parent section was dropped
    (e.g. '=== Art design ===' / '=== Music ===' under '== Development ==',
    '=== Critical ===' / '=== Commercial ===' under '== Reception ==')."""
    keep, kept_secs, dropped_secs = [], [], []
    state = True                      # lead is kept
    parent_keep = True
    for line in text.splitlines():
        m = HDR.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2)
            hit = any(k in title.lower() for k in KEEP_KEYS)
            if level == 2:            # top-level section sets the parent state
                parent_keep = hit
                state = hit
                (kept_secs if hit else dropped_secs).append(title)
            else:                     # subsection: parent verdict OR own match
                state = parent_keep or hit
                if hit and not parent_keep:
                    kept_secs.append(f"{title} (rescued)")
                elif not state:
                    dropped_secs.append(title)
            if state:
                keep.append(line)
            continue
        if state:
            keep.append(line)
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip()
    return body, kept_secs, dropped_secs


def main() -> None:
    G = np.load(SC / "games.npz", allow_pickle=True)
    GAL = np.load(SC / "wscan_gal.npz")
    WK = np.load(SC / "wiki_views.npz", allow_pickle=True)
    names = [str(x) for x in G["names"]]
    appid2gidx = {n.split("_")[0]: i for i, n in enumerate(names)}

    # game centroids from the 512-sentence anchor pools
    gal = GAL["gal"].astype(np.float32)               # (2020, 512, 1024)
    gcent = gal.mean(1)
    gcent /= np.linalg.norm(gcent, axis=1, keepdims=True) + 1e-8

    # wiki page centroids from the embedded lead-64 views
    Ws = WK["S"].astype(np.float32)                   # (924, L, 1024)
    Wlen = WK["S_len"]
    wcent = np.stack([Ws[i, :Wlen[i]].mean(0) for i in range(len(Ws))])
    wcent /= np.linalg.norm(wcent, axis=1, keepdims=True) + 1e-8
    wk_gidx = [int(x) for x in WK["gidx"]]
    row_of_gidx = {g: i for i, g in enumerate(wk_gidx)}

    sims = wcent @ gcent.T                            # (924, 2020)

    manifest, suspects, kept, thin = {}, [], 0, 0
    for f in sorted(SRC.glob("*.txt")):
        appid = f.name.split("_")[0]
        gidx = appid2gidx.get(appid)
        rec = {"file": f.name, "gidx": gidx}
        raw = f.read_text(encoding="utf-8", errors="ignore")
        meta, raw_body = split_meta(raw)
        body, ks, ds = filter_sections(raw_body)
        game_name = meta.get("Game") or (f.stem.split("_", 1)[1] if "_" in f.stem else f.stem)
        src_title = meta.get("Source", "")
        title_eq = norm_title(src_title) == norm_title(game_name) if src_title else False
        # NECESSARY condition (user): the game's English name must appear in the
        # article BODY itself (collector header already stripped by split_meta —
        # the header contains the name by construction and must not count).
        name_in_body = norm_title(game_name) in norm_title(raw_body)
        rec.update(chars_raw=len(raw), chars_clean=len(body),
                   kept_sections=ks, dropped_sections=ds,
                   source_title=src_title, title_eq=title_eq,
                   name_in_body=name_in_body)
        # disambiguation pages ("X may refer to:") carry no game content but
        # trivially pass name-in-body — hard-exclude them (10 found 2026-07-11)
        if re.search(r"may refer to\s*:", raw_body[:600], re.I):
            rec["verdict"] = "DISAMBIG"
            manifest[f.name] = rec
            suspects.append((f.name, None, None, src_title, "DISAMBIG"))
            continue
        # ---- match validation: title equality OR embedding confirmation ----
        verdict = "no-embedding"
        if gidx is not None and gidx in row_of_gidx:
            r = row_of_gidx[gidx]
            order = np.argsort(-sims[r])
            rank = int(np.where(order == gidx)[0][0]) + 1
            margin = float(sims[r, order[0]] - sims[r, gidx])
            rec.update(match_rank=rank, match_cos=float(sims[r, gidx]),
                       margin=margin, top1_game=names[order[0]],
                       top1_cos=float(sims[r, order[0]]))
            emb_ok = rank <= RANK_OK or margin <= MARGIN_OK
            # name-in-body is NECESSARY; title/embedding confirmation on top
            verdict = "ok" if (name_in_body and (title_eq or emb_ok)) else "SUSPECT"
        rec["verdict"] = verdict
        if verdict == "SUSPECT":
            why = []
            if not name_in_body:
                why.append("NAME-NOT-IN-BODY")
            if not title_eq:
                why.append("title!=")
            rec["why"] = why
            suspects.append((f.name, rec.get("match_rank"), rec.get("margin"),
                             src_title, "|".join(why)))
        elif len(body) < MIN_CHARS:
            rec["verdict"] = "thin"
            thin += 1
        else:
            (DST / f.name).write_text(body, encoding="utf-8")
            kept += 1
        manifest[f.name] = rec

    json.dump(manifest, open(SCRIPT_DIR / "wiki_clean_manifest.json", "w",
                             encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"kept {kept} | thin {thin} | suspects {len(suspects)} | "
          f"total {len(manifest)}", flush=True)
    print("\n=== SUSPECTED MISMATCHES (excluded, review these) ===")
    for name, rank, margin, src_title, why in sorted(
            suspects, key=lambda x: -(x[2] or 0)):
        print(f"  {name}: [{why}] page title {src_title!r}, "
              f"own-game rank {rank}, margin {margin}")


if __name__ == "__main__":
    main()
