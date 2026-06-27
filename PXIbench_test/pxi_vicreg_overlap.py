"""Find games shared by the PXI benchmark and the VICReg review training set.

The PXI benchmark (PXIbenchmark_data/benchmark.csv) scores ~361 real games on 11
Player Experience Inventory dimensions (5 psychological + 5 functional + challenge),
averaged over several survey respondents per game. The VICReg encoder is trained on
293 Steam games' reviews. Where the two overlap, we can probe the frozen VICReg
code against PXI experience dimensions -- a richer target than Steam tags ("not
just TAP").

Matching is name-based and conservative:
  * exact (normalized) name match -> trusted automatically
  * a small hand-vetted allowlist for edition/remaster suffixes
    ("Horizon Zero Dawn" == "Horizon Zero Dawn Complete Edition")
Franchise collisions are explicitly rejected (Ark/Lost Ark, League of Legends/
Ruined King, The Witcher/GWENT, Total War Warhammer 2 vs III, etc.).

Outputs PXIbench_test/pxi_vicreg_overlap.json: per matched game the appid, both
names, match type, and the averaged 11-dim PXI vector.
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BENCH = SCRIPT_DIR / "PXIbenchmark_data" / "benchmark.csv"
VIC_H5 = ROOT / "game_review_data" / "embedding_h5.h5"
GAMES_JSON = ROOT / "game_review_data" / "games.json"
OUT = SCRIPT_DIR / "pxi_vicreg_overlap.json"

PXI_DIMS = [
    "psychological_meaning", "psychological_mastery", "psychological_curiosity",
    "psychological_autonomy", "psychological_immersion",
    "functional_progress_feedback", "functional_ease_of_control",
    "functional_audiovisual_appeal", "functional_goals_and_rules", "functional_challenge",
]

# Hand-vetted same-game matches whose names differ only by edition/remaster suffix.
# Keyed by PXI game_name -> Steam appid in the VICReg set.
VETTED_VARIANTS = {
    "Horizon zero dawn": "1151640",   # Horizon Zero Dawn Complete Edition
    "Nioh 2": "1325200",              # Nioh 2 - The Complete Edition
    "Nier Replicant": "1113560",      # NieR Replicant ver.1.22474487139...
    "Spider-Man": "1817070",          # Marvel's Spider-Man Remastered
    "Uncharted": "1659420",           # UNCHARTED: Legacy of Thieves Collection
    "Star Wars Battlefront": "1237950",  # STAR WARS Battlefront II
}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower().replace("’", "'")).strip()


def load_vic_names():
    with h5py.File(VIC_H5, "r") as h5:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in h5["game_names"][:]]
    appids = sorted(set(n.split("_")[0] for n in names))
    games = json.loads(GAMES_JSON.read_text(encoding="utf-8"))
    out = {}
    for aid in appids:
        rec = games.get(aid)
        if isinstance(rec, dict) and rec.get("name"):
            out[aid] = rec["name"]
    return out


def load_pxi():
    rows = list(csv.DictReader(open(BENCH, encoding="utf-8-sig")))
    by_game = defaultdict(list)
    genre = {}
    for r in rows:
        by_game[r["game_name"]].append(r)
        genre[r["game_name"]] = r["genre_name"]
    vectors = {}
    for name, recs in by_game.items():
        vec = {}
        for dim in PXI_DIMS:
            vals = [float(rr[dim]) for rr in recs if rr.get(dim) not in (None, "")]
            vec[dim] = sum(vals) / len(vals) if vals else None
        vectors[name] = {"pxi": vec, "n_samples": len(recs), "genre": genre[name]}
    return vectors


def main():
    vic = load_vic_names()           # appid -> steam name
    vic_norm = {norm(n): aid for aid, n in vic.items()}
    pxi = load_pxi()                 # pxi name -> {pxi, n_samples, genre}

    matches = []
    used_appids = set()
    for pname, info in pxi.items():
        aid = None
        mtype = None
        if norm(pname) in vic_norm:
            aid, mtype = vic_norm[norm(pname)], "exact"
        elif pname in VETTED_VARIANTS:
            aid, mtype = VETTED_VARIANTS[pname], "vetted_variant"
        if aid is None or aid in used_appids:
            continue
        used_appids.add(aid)
        matches.append({
            "appid": aid,
            "pxi_name": pname,
            "steam_name": vic.get(aid, ""),
            "match": mtype,
            "genre": info["genre"],
            "n_pxi_samples": info["n_samples"],
            "pxi": info["pxi"],
        })

    matches.sort(key=lambda m: (m["match"], m["pxi_name"]))
    OUT.write_text(json.dumps({"dims": PXI_DIMS, "matches": matches}, ensure_ascii=False, indent=2), encoding="utf-8")

    exact = [m for m in matches if m["match"] == "exact"]
    vetted = [m for m in matches if m["match"] == "vetted_variant"]
    print(f"PXI games={len(pxi)}  VICReg games={len(vic)}")
    print(f"OVERLAP: {len(matches)} games  (exact={len(exact)}, vetted_variant={len(vetted)})")
    for m in matches:
        print(f"  [{m['match']:14s}] {m['pxi_name']:28s} = {m['steam_name']}  (appid {m['appid']}, {m['genre']})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
