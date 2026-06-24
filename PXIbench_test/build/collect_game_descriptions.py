"""Collect short internet-sourced description txt files for PXI benchmark games.

The script reads unique game names from PXIbenchmark_data/benchmark.csv, searches
Wikipedia through the public MediaWiki API, and writes one .txt file per game.
Descriptions are intentionally short and source-linked so they can be used as a
real-text replacement for the synthetic PXI pseudo text.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
PXI_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = PXI_DIR / "PXIbenchmark_data" / "benchmark.csv"
DEFAULT_OUTPUT_DIR = PXI_DIR / "PXIbenchmark_descriptions"
WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "studable-query-latent/1.0 (PXI benchmark text collection)"


QUERY_ALIASES = {
    "21": "Twenty One card game",
    "8 ball pool": "8 Ball Pool",
    "Agar": "Agar.io",
    "Assassin’s Creed": "Assassin's Creed",
    "Ark": "ARK Survival Evolved",
    "Asphalt nitro": "Asphalt Nitro",
    "Badland": "Badland video game",
    "Banjo Kazooie": "Banjo-Kazooie",
    "Battlefield": "Battlefield (video game series)",
    "Bioshock": "BioShock",
    "Black desert": "Black Desert Online",
    "Bloons": "Bloons TD",
    "Clash Royal": "Clash Royale",
    "Company of heroes": "Company of Heroes",
    "Counter Strike": "Counter-Strike",
    "Civilization": "Civilization video game series",
    "Darkest Dungeon": "Darkest Dungeon video game",
    "Dead by daylight": "Dead by Daylight",
    "Diablo 3": "Diablo III",
    "Dirt rally": "Dirt Rally",
    "Divinity Original Sin 2": "Divinity: Original Sin II",
    "Don’t Blank": "Don't Blink game",
    "Don’t Starve": "Don't Starve",
    "Dragon age": "Dragon Age",
    "Dream gonden": "Dream Garden game",
    "Elder Scrolls Online": "The Elder Scrolls Online",
    "Europa Universalis": "Europa Universalis IV",
    "F1": "Formula One video game series",
    "Farm Simulator": "Farming Simulator",
    "Farmville": "FarmVille",
    "FIFA": "FIFA video game series",
    "Final Fantasy": "Final Fantasy video game series",
    "Garden escape": "Gardenscapes",
    "GemCraft": "GemCraft",
    "Glory of King": "Honor of Kings",
    "GTA": "Grand Theft Auto",
    "Harten Jagen": "Hearts (card game)",
    "Heroes and Generals": "Heroes & Generals",
    "Heroes of the storm": "Heroes of the Storm",
    "Honkai impact": "Honkai Impact 3rd",
    "Home Sweet Home": "Home Sweet Home (2017 video game)",
    "Hollow knight": "Hollow Knight",
    "Horizon zero dawn": "Horizon Zero Dawn",
    "Into The Breach": "Into the Breach",
    "Kim Kardashian: Hollywood": "Kim Kardashian: Hollywood",
    "King of Glory": "Honor of Kings",
    "Krunker.io": "Krunker",
    "Lazor": "Lazors",
    "Lord of The Rings Online": "The Lord of the Rings Online",
    "Missions and racing fighting": "Mission Against Terror",
    "MMM Fingers": "Mmm Fingers",
    "MOHA": "Medal of Honor: Allied Assault",
    "Mortal Combat": "Mortal Kombat",
    "My time in portia": "My Time at Portia",
    "NBA": "NBA 2K video game series",
    "Nier Automata": "Nier: Automata",
    "No man’s sky": "No Man's Sky",
    "Off-world Trading Company": "Offworld Trading Company",
    "Okami HD": "Okami",
    "Op de Bus": "Op de Bus card game",
    "Patern Go": "Pattern Go game",
    "Payday 2": "Payday 2 video game",
    "Pokemon": "Pokemon video game series",
    "Pokemon Go": "Pokemon Go",
    "Pikmin III deluxe": "Pikmin 3 Deluxe",
    "PUBG": "PUBG: Battlegrounds",
    "Ratchet & clank": "Ratchet & Clank",
    "Score!": "Score! Hero",
    "Shadow Of Mordor": "Middle-earth: Shadow of Mordor",
    "Shogun": "Shogun: Total War",
    "Simcity": "SimCity",
    "Skyrim": "The Elder Scrolls V: Skyrim",
    "Smite": "SMITE",
    "Snake": "Snake video game genre",
    "Spider-Man": "Marvel's Spider-Man",
    "Starcraft": "StarCraft",
    "Subnautica Below Zero": "Subnautica: Below Zero",
    "Sudoku": "Sudoku",
    "Superman": "Superman video game",
    "Team Fortress": "Team Fortress Classic",
    "TERA": "TERA video game",
    "The Elder Scrolls": "The Elder Scrolls",
    "The last of us": "The Last of Us",
    "Tom Clanc’s Rainbow Six Siege": "Tom Clancy's Rainbow Six Siege",
    "Tommy Hawk Pro Skater I": "Tony Hawk's Pro Skater",
    "Total War Warhammer 2": "Total War: Warhammer II",
    "Tribalwars": "Tribal Wars",
    "Trigger fist": "Trigger Fist",
    "Warband": "Mount & Blade: Warband",
    "Warblade": "Warblade video game",
    "Warcraft": "Warcraft video game series",
    "Warzone": "Call of Duty: Warzone",
    "Where is Wally": "Where's Wally? video game",
    "Wii Sport (tennis)": "Wii Sports",
    "Witcher 3": "The Witcher 3: Wild Hunt",
    "Worldcraft": "WorldCraft video game",
    "WWW": "World Wide Web game",
    "Atelier Escha and Logy": "Atelier Escha & Logy: Alchemists of the Dusk Sky",
    "Crusader King": "Crusader Kings",
    "Detroit: Becoming Human": "Detroit: Become Human",
    "Drangon box": "DragonBox",
    "Fall Guy": "Fall Guys",
    "Great Ace Attorney Chronicles": "The Great Ace Attorney Chronicles",
    "It takes two": "It Takes Two",
    "Lightmatter": "Lightmatter video game",
    "Maplestory": "MapleStory",
    "Middle earth: shadow of war": "Middle-earth: Shadow of War",
    "Plants vs Zombies2": "Plants vs. Zombies 2",
    "Phantasy Star": "Phantasy Star video game series",
    "PCM 2020": "Pro Cycling Manager 2020",
    "Prince Of Persia: Forgotten Sands": "Prince of Persia: The Forgotten Sands",
    "Raceroom Racing Experience": "RaceRoom Racing Experience",
    "Sea of Theives": "Sea of Thieves",
    "Sekiro shadows die twice": "Sekiro: Shadows Die Twice",
    "Solasta: Crown if the Magister": "Solasta: Crown of the Magister",
    "South park the stick of the truth": "South Park: The Stick of Truth",
    "Superhot VR": "Superhot VR",
    "Tamashii": "Tamashii video game",
    "Uncharted 4: A Thief’s End": "Uncharted 4: A Thief's End",
}


NEGATIVE_TITLE_TERMS = (
    "soundtrack",
    "album",
    "film",
    "television",
    "episode",
    "song",
    "novel",
    "company",
    "designer",
    "disambiguation",
    "music",
    "goldsrc",
)
POSITIVE_TEXT_TERMS = (
    "video game",
    "mobile game",
    "computer game",
    "browser game",
    "arcade game",
    "board game",
    "card game",
    "puzzle game",
    "platform game",
    "sandbox game",
    "stealth game",
    "action game",
    "fighting game",
    "shooter game",
    "tower defense game",
    "match-3",
    "casual game",
    "adventure game",
    "action-adventure game",
    "simulation game",
    "life simulation game",
    "social simulation game",
    "construction and management simulation",
    "sports game",
    "racing game",
    "rhythm game",
    "music game",
    "role-playing game",
    "online role-playing game",
    "massively multiplayer online role-playing game",
    "mmorpg",
    "first-person shooter",
    "third-person shooter",
    "strategy game",
    "real-time strategy",
    "turn-based strategy",
    "survival game",
    "battle royale game",
    "multiplayer",
    "free-to-play",
    "game developed",
    "game published",
    "game series",
    "social network game",
    "word puzzle",
    "logic puzzle",
)
GAMEPLAY_SECTION_TERMS = ("gameplay", "game play", "mechanics", "features")
STORY_SECTION_TERMS = ("plot", "story", "synopsis", "setting", "premise", "campaign")
GAMEPLAY_SENTENCE_TERMS = (
    "player",
    "players",
    "gameplay",
    "control",
    "combat",
    "puzzle",
    "race",
    "racing",
    "build",
    "manage",
    "explore",
    "multiplayer",
    "mode",
    "match",
)
STORY_SENTENCE_TERMS = (
    "plot",
    "story",
    "set in",
    "takes place",
    "follows",
    "assumes the role",
    "campaign",
    "protagonist",
    "world",
)


@dataclass
class Game:
    game_id: str
    name: str
    genres: str


@dataclass
class WikiPage:
    title: str
    pageid: int
    extract: str
    url: str
    score: float


def read_games(path: Path) -> list[Game]:
    by_name: dict[str, Game] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row["game_name"].strip()
            game_id = row["game_id"].strip()
            genre = (row.get("genre_name") or row.get("genre_names") or "").strip()
            if name not in by_name:
                by_name[name] = Game(game_id=game_id, name=name, genres=genre)
            elif genre and genre not in by_name[name].genres.split("|"):
                current = by_name[name].genres
                by_name[name] = Game(game_id=game_id, name=name, genres=f"{current}|{genre}".strip("|"))
    return sorted(by_name.values(), key=lambda game: (game.game_id, game.name.lower()))


def wiki_get(session: requests.Session, **params) -> dict:
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = session.get(WIKI_API, params=params, timeout=30)
            if response.status_code in {429, 500, 502, 503, 504}:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(2**attempt, 30)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(2**attempt, 30))
    if last_error:
        raise last_error
    response.raise_for_status()
    return response.json()


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def search_pages(session: requests.Session, game_name: str, limit: int = 8) -> list[dict]:
    query = QUERY_ALIASES.get(game_name, game_name)
    searches = [f'{query} video game', query]
    seen: set[int] = set()
    results: list[dict] = []
    for search in searches:
        data = wiki_get(
            session,
            action="query",
            list="search",
            srsearch=search,
            srlimit=limit,
            srprop="snippet",
        )
        for item in data.get("query", {}).get("search", []):
            pageid = int(item["pageid"])
            if pageid not in seen:
                seen.add(pageid)
                results.append(item)
    return results


def candidate_titles(game_name: str) -> list[str]:
    names = [QUERY_ALIASES.get(game_name, game_name), game_name]
    titles: list[str] = []
    for name in names:
        variants = [
            name,
            f"{name} (video game)",
            f"{name} (video game series)",
            f"{name} (franchise)",
            f"{name} (series)",
        ]
        if name.lower().endswith("video game series"):
            variants.append(name.removesuffix(" video game series") + " (video game series)")
        for title in variants:
            if title not in titles:
                titles.append(title)
    return titles


def fetch_title_extracts(session: requests.Session, titles: Iterable[str]) -> list[dict]:
    title_list = [title for title in titles if title]
    if not title_list:
        return []
    data = wiki_get(
        session,
        action="query",
        prop="extracts|info",
        explaintext=1,
        redirects=1,
        inprop="url",
        titles="|".join(title_list),
    )
    pages = data.get("query", {}).get("pages", [])
    return [page for page in pages if not page.get("missing")]


def fetch_title_infos(session: requests.Session, titles: Iterable[str]) -> list[dict]:
    title_list = [title for title in titles if title]
    if not title_list:
        return []
    data = wiki_get(
        session,
        action="query",
        prop="info",
        redirects=1,
        inprop="url",
        titles="|".join(title_list),
    )
    pages = data.get("query", {}).get("pages", [])
    return [page for page in pages if not page.get("missing")]


def title_key(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("_", " ").strip()).casefold()


def fetch_direct_page_cache(
    session: requests.Session,
    games: Iterable[Game],
    *,
    chunk_size: int = 45,
) -> dict[str, list[dict]]:
    all_titles: list[str] = []
    seen_titles: set[str] = set()
    for game in games:
        for title in candidate_titles(game.name):
            key = title_key(title)
            if key not in seen_titles:
                seen_titles.add(key)
                all_titles.append(title)

    cache: dict[str, list[dict]] = {}
    for start in range(0, len(all_titles), chunk_size):
        chunk = all_titles[start : start + chunk_size]
        data = wiki_get(
            session,
            action="query",
            prop="info",
            redirects=1,
            inprop="url",
            titles="|".join(chunk),
        )
        pages_by_title = {
            title_key(page.get("title", "")): page
            for page in data.get("query", {}).get("pages", [])
            if not page.get("missing")
        }
        normalized = {
            title_key(item.get("from", "")): title_key(item.get("to", ""))
            for item in data.get("query", {}).get("normalized", [])
        }
        redirects = {
            title_key(item.get("from", "")): title_key(item.get("to", ""))
            for item in data.get("query", {}).get("redirects", [])
        }
        for title in chunk:
            keys = [title_key(title)]
            if keys[-1] in normalized:
                keys.append(normalized[keys[-1]])
            if keys[-1] in redirects:
                keys.append(redirects[keys[-1]])
            page = None
            for key in reversed(keys):
                page = pages_by_title.get(key)
                if page:
                    break
            if page:
                cache.setdefault(title_key(title), []).append(page)
                cache.setdefault(title_key(page.get("title", "")), []).append(page)
        print(f"cached direct pages {min(start + chunk_size, len(all_titles))}/{len(all_titles)}")
    return cache


def fetch_extracts(session: requests.Session, pageids: Iterable[int]) -> list[dict]:
    ids = "|".join(str(pageid) for pageid in pageids)
    if not ids:
        return []
    data = wiki_get(
        session,
        action="query",
        prop="extracts|info",
        explaintext=1,
        redirects=1,
        inprop="url",
        pageids=ids,
    )
    pages = data.get("query", {}).get("pages", [])
    return [page for page in pages if not page.get("missing")]


def fetch_page_extract(session: requests.Session, pageid: int) -> dict | None:
    pages = fetch_extracts(session, [pageid])
    if not pages:
        return None
    page = pages[0]
    return page if page.get("extract", "").strip() else None


def score_page(game_name: str, page: dict) -> float:
    title = page.get("title", "")
    text = (title + "\n" + page.get("extract", "")[:1600]).lower()
    score = similarity(QUERY_ALIASES.get(game_name, game_name), title) * 4.0
    score += max(similarity(game_name, title) * 2.0, 0)
    if any(term in text for term in POSITIVE_TEXT_TERMS):
        score += 2.0
    if any(term in title.lower() for term in NEGATIVE_TITLE_TERMS):
        score -= 3.0
    if "(video game)" in title.lower() or "(series)" in title.lower():
        score += 1.0
    return score


def is_likely_game_page(page: dict) -> bool:
    title = page.get("title", "").lower()
    extract = page.get("extract", "").strip().lower()
    first_block = extract[:1800]
    if any(term in title for term in NEGATIVE_TITLE_TERMS):
        return False
    if "(disambiguation)" in title:
        return False
    if any(term in title for term in ("video game", "game series", "franchise")):
        return True
    return any(term in first_block for term in POSITIVE_TEXT_TERMS)


def find_best_page(
    session: requests.Session,
    game_name: str,
    *,
    search_fallback: bool = False,
) -> WikiPage | None:
    direct_pages = [
        page
        for page in fetch_title_extracts(session, candidate_titles(game_name))
        if page.get("extract", "").strip()
        and is_likely_game_page(page)
    ]
    if direct_pages:
        pages = direct_pages
    elif search_fallback:
        candidates = search_pages(session, game_name)
        pages = [
            page
            for page in fetch_extracts(session, [item["pageid"] for item in candidates])
            if page.get("extract", "").strip()
            and is_likely_game_page(page)
            and score_page(game_name, page) >= 6.0
        ]
    else:
        pages = []
    if not pages:
        return None
    ranked = sorted(pages, key=lambda page: score_page(game_name, page), reverse=True)
    best = ranked[0]
    score = score_page(game_name, best)
    title = best.get("title", "")
    extract = best.get("extract", "").strip()
    if score < 2.2 and not any(term in extract.lower()[:1400] for term in POSITIVE_TEXT_TERMS):
        return None
    url = best.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
    return WikiPage(title=title, pageid=int(best["pageid"]), extract=extract, url=url, score=score)


def find_best_page_from_cache(game_name: str, cache: dict[str, list[dict]]) -> WikiPage | None:
    pages_by_id: dict[int, dict] = {}
    for title in candidate_titles(game_name):
        for page in cache.get(title_key(title), []):
            if page.get("extract", "").strip() and is_likely_game_page(page):
                pages_by_id[int(page["pageid"])] = page
    if not pages_by_id:
        return None
    ranked = sorted(pages_by_id.values(), key=lambda page: score_page(game_name, page), reverse=True)
    best = ranked[0]
    score = score_page(game_name, best)
    title = best.get("title", "")
    url = best.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
    return WikiPage(
        title=title,
        pageid=int(best["pageid"]),
        extract=best.get("extract", "").strip(),
        url=url,
        score=score,
    )


def find_best_page_from_cache_with_extract(
    session: requests.Session,
    game_name: str,
    cache: dict[str, list[dict]],
) -> WikiPage | None:
    candidates_by_id: dict[int, dict] = {}
    for title in candidate_titles(game_name):
        for page in cache.get(title_key(title), []):
            pageid = int(page["pageid"])
            candidates_by_id.setdefault(pageid, page)

    for page in sorted(candidates_by_id.values(), key=lambda item: score_page(game_name, item), reverse=True):
        full_page = fetch_page_extract(session, int(page["pageid"]))
        if full_page and is_likely_game_page(full_page):
            title = full_page.get("title", "")
            url = full_page.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            return WikiPage(
                title=title,
                pageid=int(full_page["pageid"]),
                extract=full_page.get("extract", "").strip(),
                url=url,
                score=score_page(game_name, full_page),
            )
    return None


def split_sections(extract: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"Lead": []}
    current = "Lead"
    heading_re = re.compile(r"^(={2,6})\s*(.+?)\s*\1$")
    for line in extract.splitlines():
        match = heading_re.match(line.strip())
        if match:
            current = match.group(2).strip()
            sections.setdefault(current, [])
            continue
        if line.strip():
            sections.setdefault(current, []).append(line.strip())
    return {name: "\n".join(lines).strip() for name, lines in sections.items() if "\n".join(lines).strip()}


def sentences(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    clean = []
    for part in parts:
        part = part.strip()
        word_count = len(part.split())
        if 7 <= word_count <= 60:
            clean.append(part)
    return clean


def select_by_section(sections: dict[str, str], terms: tuple[str, ...], limit: int) -> list[str]:
    selected: list[str] = []
    for name, text in sections.items():
        lowered = name.lower()
        if any(term in lowered for term in terms):
            for sent in sentences(text):
                selected.append(sent)
                if len(selected) >= limit:
                    return selected
    return selected


def select_by_keywords(section_texts: Iterable[str], terms: tuple[str, ...], limit: int) -> list[str]:
    selected: list[str] = []
    for text in section_texts:
        for sent in sentences(text):
            lowered = sent.lower()
            if any(term in lowered for term in terms):
                selected.append(sent)
                if len(selected) >= limit:
                    return selected
    return selected


def build_description(game: Game, page: WikiPage | None, retrieved_on: str) -> str:
    header = [
        f"Game: {game.name}",
        f"PXI game_id: {game.game_id}",
        f"PXI genre: {game.genres or 'unknown'}",
        f"Retrieved: {retrieved_on}",
        "",
    ]
    if page is None:
        body = [
            "Source: no high-confidence Wikipedia page was found automatically.",
            "",
            "Description:",
            (
                f"{game.name} appears in the PXI benchmark under the genre label "
                f"'{game.genres or 'unknown'}'. No reliable internet summary was matched by the "
                "automated collector, so this file is left as a placeholder for manual review."
            ),
        ]
        return "\n".join(header + body).strip() + "\n"

    sections = split_sections(page.extract)
    lead = sentences(sections.get("Lead", ""))[:2]
    gameplay = select_by_section(sections, GAMEPLAY_SECTION_TERMS, 3)
    if not gameplay:
        gameplay = select_by_keywords(sections.values(), GAMEPLAY_SENTENCE_TERMS, 3)
    story = select_by_section(sections, STORY_SECTION_TERMS, 3)
    if not story:
        story = select_by_keywords(sections.values(), STORY_SENTENCE_TERMS, 3)

    if not lead:
        lead = sentences(page.extract)[:2]
    if not gameplay:
        gameplay = ["The retrieved source did not expose a separate gameplay description in plain text."]
    if not story:
        story = ["No fixed plot or story premise was identified in the retrieved source."]

    body = [
        f"Source: {page.title}",
        f"URL: {page.url}",
        f"Match score: {page.score:.2f}",
        "",
        "Overview:",
        " ".join(lead),
        "",
        "Gameplay and features:",
        " ".join(gameplay),
        "",
        "Story or premise:",
        " ".join(story),
    ]
    return "\n".join(header + body).strip() + "\n"


def safe_filename(name: str) -> str:
    cleaned = name.strip()
    cleaned = cleaned.replace(":", " -")
    cleaned = re.sub(r'[<>"/\\|?*]', "_", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return f"{cleaned or 'unknown_game'}.txt"


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def write_metadata_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "game_id",
        "game_name",
        "filename",
        "pxi_genre",
        "source_title",
        "source_url",
        "match_score",
        "status",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.15, help="Pause between API lookups.")
    parser.add_argument("--limit", type=int, help="Only process the first N games.")
    parser.add_argument(
        "--search-fallback",
        action="store_true",
        help="Use Wikipedia full-text search when direct title/alias lookup fails.",
    )
    args = parser.parse_args()

    games = read_games(args.input)
    if args.limit:
        games = games[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retrieved_on = time.strftime("%Y-%m-%d")
    metadata: list[dict[str, object]] = []
    direct_cache = fetch_direct_page_cache(session, games)

    for index, game in enumerate(games, start=1):
        filename = safe_filename(game.name)
        output_path = args.output_dir / filename
        if output_path.exists() and not args.overwrite:
            status = "skipped_existing"
            metadata.append(
                {
                    "game_id": game.game_id,
                    "game_name": game.name,
                    "filename": filename,
                    "pxi_genre": game.genres,
                    "source_title": "",
                    "source_url": "",
                    "match_score": "",
                    "status": status,
                }
            )
            print(f"[{index}/{len(games)}] skipped {game.name}")
            continue

        page = find_best_page_from_cache_with_extract(session, game.name, direct_cache)
        if page is None and args.search_fallback:
            page = find_best_page(session, game.name, search_fallback=True)
        text = build_description(game, page, retrieved_on)
        atomic_write_text(output_path, text)
        status = "ok" if page else "needs_manual_review"
        metadata.append(
            {
                "game_id": game.game_id,
                "game_name": game.name,
                "filename": filename,
                "pxi_genre": game.genres,
                "source_title": page.title if page else "",
                "source_url": page.url if page else "",
                "match_score": f"{page.score:.2f}" if page else "",
                "status": status,
            }
        )
        source = page.title if page else "NO MATCH"
        print(f"[{index}/{len(games)}] {status}: {game.name} -> {source}")
        if args.sleep:
            time.sleep(args.sleep)

    write_metadata_csv(args.output_dir / "metadata.csv", metadata)
    atomic_write_text(
        args.output_dir / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    print(f"Done. Wrote {len(games)} description files to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
