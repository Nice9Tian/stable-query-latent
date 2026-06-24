"""Thin wrapper — building test_games.json now lives in tag_build.py.

Kept for backward compatibility. `tag_build.py` writes test_games.json and
non_emotional_tags.json as part of its normal run. This rebuilds just those two
from the existing tag_vocab.json / tag_groups.json, delegating to tag_build's
build_test_games so the logic has a single home.
"""

import json
import sys
from pathlib import Path

import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tag_build import build_test_games, build_tag_groups  # noqa: E402

H5 = SCRIPT_DIR / "h5" / "game_review_cleaned_3_sentences.h5"
GAMES_JSON = ROOT / "game_review_data" / "Steam Games Metadata and Player Reviews (2020–2024" / "games.json"
TAGS_DIR = SCRIPT_DIR / "tags"


def decode(v):
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def main():
    with h5py.File(H5, "r") as h5:
        game_names = [decode(n) for n in h5["game_names"][:]]
    appids = [n.split("_")[0] for n in game_names]
    games = json.loads(GAMES_JSON.read_text(encoding="utf-8"))
    vocab = json.loads((TAGS_DIR / "tag_vocab.json").read_text(encoding="utf-8"))["tags"]
    groups_path = TAGS_DIR / "tag_groups.json"
    groups = (json.loads(groups_path.read_text(encoding="utf-8"))
              if groups_path.exists() else build_tag_groups(vocab))
    subjective = set(groups.get("subjective", []))

    test_games, keep_tags = build_test_games(game_names, appids, games, vocab, subjective)
    (TAGS_DIR / "test_games.json").write_text(
        json.dumps(test_games, ensure_ascii=False, indent=2), encoding="utf-8")
    (TAGS_DIR / "non_emotional_tags.json").write_text(
        json.dumps({"tags": keep_tags, "dropped_emotional": sorted(subjective)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"intersection games={len(test_games)} non_emotional_tags={len(keep_tags)} "
          f"dropped_emotional={len(subjective)}")
    print(f"wrote {TAGS_DIR / 'test_games.json'}")
    print(f"wrote {TAGS_DIR / 'non_emotional_tags.json'}")


if __name__ == "__main__":
    main()
