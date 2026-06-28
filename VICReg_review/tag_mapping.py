"""Load and apply the TAG tag mapping.

`tags/tag_mapping.json` is the single source of truth: every fine Steam tag maps
to one coarse TAG class or to "del".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MAPPING = SCRIPT_DIR / "tags" / "tag_mapping.json"


def load_tag_mapping(path: str | Path = DEFAULT_MAPPING) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = {str(k): str(v) for k, v in payload.get("mapping", {}).items()}
    delete_value = str(payload.get("delete_value", "del"))
    names = sorted({v for v in mapping.values() if v != delete_value})
    return {
        "path": str(Path(path)),
        "delete_value": delete_value,
        "mapping": mapping,
        "keywords": {str(k): [str(x) for x in v] for k, v in payload.get("keywords", {}).items()},
        "tag_names": names,
        "raw": payload,
    }


def map_tag_dict(tags: dict[str, float] | list[str] | tuple[str, ...], spec: dict) -> dict[str, float]:
    """Map a Steam tag dict/list into {tag_name: max_source_weight}."""
    if isinstance(tags, dict):
        items = [(str(name), float(weight)) for name, weight in tags.items()]
    else:
        items = [(str(name), 1.0) for name in (tags or [])]
    mapping = spec["mapping"]
    delete_value = spec["delete_value"]
    out: dict[str, float] = {}
    for fine, weight in items:
        coarse = mapping.get(fine, fine if fine in spec["tag_names"] else delete_value)
        if coarse == delete_value:
            continue
        out[coarse] = max(out.get(coarse, 0.0), float(weight))
    return out


def vectorize_tags(mapped: dict[str, float], tag_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.zeros(len(tag_names), dtype=np.float32)
    index = {name: i for i, name in enumerate(tag_names)}
    for name, weight in mapped.items():
        if name in index:
            raw[index[name]] = max(raw[index[name]], float(weight))
    labels = (raw > 0).astype(np.uint8)
    return labels, raw


def keyword_scores(text: str, tag_names: list[str], spec: dict | None = None) -> np.ndarray:
    spec = spec or load_tag_mapping()
    lower = str(text or "").lower()
    scores = np.zeros(len(tag_names), dtype=np.float32)
    for index, name in enumerate(tag_names):
        keywords = [name] + spec.get("keywords", {}).get(name, [])
        seen: set[str] = set()
        for keyword in keywords:
            key = str(keyword).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            if key in lower:
                scores[index] += 1.0 if len(key) > 3 else 0.5
    max_score = float(scores.max()) if scores.size else 0.0
    return scores / max_score if max_score > 0 else scores
