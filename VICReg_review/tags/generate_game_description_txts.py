from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "test_games.json"
OUTPUT_DIR = SCRIPT_DIR / "game_descriptions"
CACHE_DIR = SCRIPT_DIR / "_steam_appdetails_cache"

STEAM_APPDETAILS = "https://store.steampowered.com/api/appdetails"
STEAM_STORE = "https://store.steampowered.com/app/{appid}/"
OFFLINE = True


def strip_html(text: object) -> str:
    if not text:
        return ""
    s = str(text)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s*", "\n", s)
    return s.strip()


def sentence_split(text: str) -> list[str]:
    text = strip_html(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    clean = []
    for part in parts:
        part = part.strip(" \n\t-")
        if 35 <= len(part) <= 520:
            clean.append(part)
    return clean


def top_sentences(texts: list[str], keywords: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    scored: list[tuple[int, int, str]] = []
    for text in texts:
        for idx, sent in enumerate(sentence_split(text)):
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            score = sum(1 for kw in keywords if kw.lower() in key)
            if score:
                scored.append((score, -idx, sent))
    scored.sort(reverse=True)
    return [s for _, _, s in scored[:limit]]


def first_sentences(texts: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for sent in sentence_split(text):
            key = sent.lower()
            if key not in seen:
                seen.add(key)
                out.append(sent)
            if len(out) >= limit:
                return out
    return out


def wrap_bullets(sentences: list[str], fallback: str) -> str:
    if not sentences:
        return f"- {fallback}"
    return "\n".join(f"- {sent}" for sent in sentences)


def fetch_appdetails(appid: str) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{appid}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if OFFLINE:
        return {"_error": "offline mode; used local test_games.json"}

    params = {
        "appids": appid,
        "l": "schinese",
        "cc": "us",
        "filters": "basic",
    }
    url = f"{STEAM_APPDETAILS}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error = ""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(0.15)
            return data
        except Exception as exc:  # network APIs occasionally throttle or reset
            last_error = repr(exc)
            time.sleep(1.5 * (attempt + 1))

    return {"_error": last_error}


def merge_details(local: dict, fetched: dict, appid: str) -> dict:
    remote = {}
    app = fetched.get(appid) if isinstance(fetched, dict) else None
    if isinstance(app, dict) and app.get("success") and isinstance(app.get("data"), dict):
        remote = app["data"]

    merged = dict(local)
    for key in [
        "name",
        "short_description",
        "detailed_description",
        "about_the_game",
        "supported_languages",
    ]:
        if remote.get(key):
            merged[key] = remote[key]
    if remote.get("genres"):
        merged["genres"] = [g.get("description", "") for g in remote["genres"] if g.get("description")]
    if remote.get("categories"):
        merged["categories"] = [c.get("description", "") for c in remote["categories"] if c.get("description")]
    if remote.get("release_date", {}).get("date"):
        merged["release_date"] = remote["release_date"]["date"]
    return merged


def format_tags(tags: object, limit: int = 18) -> str:
    if isinstance(tags, dict):
        return "、".join(list(tags.keys())[:limit])
    if isinstance(tags, list):
        return "、".join(str(x) for x in tags[:limit])
    return ""


def generate_text(appid: str, game: dict, fetched_ok: bool) -> str:
    name = game.get("name") or f"App {appid}"
    genres = "、".join(game.get("genres") or []) or "未标明"
    categories = "、".join((game.get("categories") or [])[:16]) or "未标明"
    tags = format_tags(game.get("tags"))
    release_date = game.get("release_date") or "未知"

    texts = [
        game.get("short_description", ""),
        game.get("about_the_game", ""),
        game.get("detailed_description", ""),
    ]
    story_keywords = [
        "story", "world", "kingdom", "city", "planet", "war", "evil", "dark", "mystery",
        "discover", "save", "survive", "escape", "journey", "quest", "ancient", "future",
        "故事", "世界", "王国", "城市", "战争", "黑暗", "神秘", "拯救", "生存", "逃离", "旅程",
    ]
    mechanic_keywords = [
        "play", "combat", "battle", "build", "craft", "explore", "co-op", "multiplayer",
        "single-player", "upgrade", "customize", "weapon", "skill", "cards", "deck",
        "strategy", "turn-based", "open world", "survival", "base", "procedural",
        "玩法", "战斗", "建造", "探索", "合作", "多人", "升级", "自定义", "武器", "技能", "卡牌", "策略",
    ]

    story = top_sentences(texts, story_keywords, 5)
    if len(story) < 3:
        story.extend(s for s in first_sentences(texts, 5) if s not in story)
    story = story[:6]

    mechanics = top_sentences(texts, mechanic_keywords, 8)
    if len(mechanics) < 4:
        mechanics.extend(s for s in first_sentences(texts[1:], 8) if s not in mechanics)
    mechanics = mechanics[:9]

    if not story:
        story_fallback = f"{name} 的公开商店资料没有给出完整剧情梗概，可从简介判断其主题与氛围主要由标签和类型定义。"
    else:
        story_fallback = ""
    if not mechanics:
        mechanic_fallback = f"{name} 的公开资料未详细展开系统规则；可优先参考类型、分类和 Steam 标签理解其玩法。"
    else:
        mechanic_fallback = ""

    source_note = "Steam appdetails 接口（简体中文）+ 本地 test_games.json"
    if not fetched_ok:
        source_note = "本地 test_games.json；Steam appdetails 本次抓取失败或无返回"

    lines = [
        f"游戏ID：{appid}",
        f"游戏名称：{name}",
        f"发售日期：{release_date}",
        f"类型：{genres}",
        f"功能/模式：{categories}",
        f"Steam 标签：{tags or '未提供'}",
        "",
        "故事/世界观说明：",
        wrap_bullets(story, story_fallback),
        "",
        "玩法机制说明：",
        wrap_bullets(mechanics, mechanic_fallback),
        "",
        "整理备注：",
        "- 以上内容根据公开商店资料、游戏简介、类型与标签整理为中文说明，偏向概括故事背景与可观察玩法机制。",
        f"- 来源：{source_note}",
        f"- 商店页面：{STEAM_STORE.format(appid=appid)}",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    games = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input": str(INPUT_PATH),
        "output_dir": str(OUTPUT_DIR),
        "total": len(games),
        "written": 0,
        "steam_fetch_failed": [],
    }

    for idx, (appid, local_game) in enumerate(games.items(), start=1):
        final = OUTPUT_DIR / f"{appid}.txt"
        if final.exists():
            manifest["written"] = idx
            continue
        fetched = fetch_appdetails(appid)
        fetched_ok = isinstance(fetched, dict) and isinstance(fetched.get(appid), dict) and fetched[appid].get("success")
        if not fetched_ok:
            manifest["steam_fetch_failed"].append(appid)
        game = merge_details(local_game, fetched, appid)
        text = generate_text(appid, game, fetched_ok)
        tmp = OUTPUT_DIR / f"{appid}.txt.tmp"
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(final)
        manifest["written"] = idx
        (OUTPUT_DIR / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if idx % 25 == 0 or idx == len(games):
            print(f"written {idx}/{len(games)}")


if __name__ == "__main__":
    main()
