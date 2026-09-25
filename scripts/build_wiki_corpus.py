"""One-time wiki corpus builder.

Walks every page in the Stardew Valley wiki, parses infoboxes with BeautifulSoup,
extracts structured entities, and writes them to the local SQLite database.

Run:
    python scripts/build_wiki_corpus.py [--refresh] [--limit N]

After running, ``src.transcribe`` uses the emitted vocabulary for STT
correction and ``src.chat`` uses the populated entity tables to answer
factual questions deterministically.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from html import unescape
from pathlib import Path
from typing import Any, Iterable

# Allow ``python scripts/build_wiki_corpus.py`` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "beautifulsoup4 is required to build the corpus.\n"
        "Install it with: pip install beautifulsoup4==4.12.3"
    ) from exc

import requests

from src.config import BASE_DIR
from src.wiki_index import (
    StardewWikiIndex,
    CACHE_DB_PATH,
)

WIKI_API_URL = "https://stardewvalleywiki.com/mediawiki/api.php"
USER_AGENT = "StardewBot/1.0 (local voice assistant)"
REQUEST_PAUSE_SECONDS = 0.15
DEFAULT_BATCH_LIMIT = 500
DEFAULT_PAGE_LIMIT: int | None = None  # All pages by default.

# Heuristic field-to-kind mapping using infobox header keywords that appear
# on the Stardew Valley wiki. Recognising the kind enables deterministic
# answer composition downstream.
#
# Cues are ordered from most to least distinctive because detection returns
# on the FIRST matching cue and kinds are iterated in dict order. Keep cues
# specific: generic fields like "Sell Price" or "Season" appear on both crop
# and fish infoboxes and previously caused fish pages to classify as crops.
KIND_INFOCUES: dict[str, tuple[str, ...]] = {
    "crop": ("Growth Time", "Regrowth"),
    "fish": ("Difficulty", "Behavior", "Fishing XP"),
    "npc": ("Birthday", "Lives In", "Marriage", "Favorite Gift"),
    "location": ("Open Hours", "Closed", "Inhabitants"),
    "bundle": ("Bundles", "Required Items", "Gold Reward"),
    "tool": ("Upgrades", "Materials"),
    "monster": ("HP", "Damage", "Drops"),
}

# Real fish pages carry categories like "Pond fish", "Lake fish", "River fish",
# "Ocean fish", "Island fish". Fishing *equipment* pages carry categories like
# "Fishing Poles", "Fish Tanks", "Fishing Tackle" — those also START with
# "fish", which is exactly why a naive startswith() check miscategorised rods
# and tanks as fish. Full-match on "(<word> )?fish" keeps them apart.
_FISH_CATEGORY_RE = re.compile(r"^(?:[a-z0-9]+ )?fish$")


def _request_json(params: dict[str, Any]) -> dict[str, Any]:
    """Fetch a MediaWiki API response. Retries with exponential backoff on 429/5xx."""
    headers = {"User-Agent": USER_AGENT}
    delay = REQUEST_PAUSE_SECONDS
    max_attempts = 4
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = requests.get(WIKI_API_URL, params=params, headers=headers, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
                last_exc = requests.HTTPError(f"{response.status_code} {response.reason}")
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
    raise last_exc if last_exc else RuntimeError("MediaWiki request failed")


def list_all_titles(limit: int | None = None) -> list[str]:
    """Iterate ``allpages`` and return every non-redirect main-namespace title."""
    titles: list[str] = []
    apcontinue: str | None = None
    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": DEFAULT_BATCH_LIMIT,
            "apnamespace": 0,
            "apfilterredir": "nonredirects",
            "format": "json",
            "formatversion": 2,
        }
        if apcontinue:
            params["apcontinue"] = apcontinue
        data = _request_json(params)
        titles.extend(page["title"] for page in data.get("query", {}).get("allpages", []))
        cont = data.get("continue", {}).get("apcontinue")
        if not cont:
            break
        apcontinue = cont
        if limit is not None and len(titles) >= limit:
            return titles[:limit]
    return titles


def fetch_page_html(title: str) -> str:
    data = _request_json(
        {
            "action": "parse",
            "page": title,
            "prop": "text",
            "format": "json",
            "formatversion": 2,
        }
    )
    parse = data.get("parse") or {}
    return parse.get("text", "")


def fetch_page_categories(title: str) -> list[str]:
    """Pull category memberships via the categories prop."""
    titles: list[str] = []
    clcontinue: str | None = None
    while True:
        params = {
            "action": "query",
            "prop": "categories",
            "titles": title,
            "cllimit": 500,
            "clshow": "!hidden",
            "format": "json",
            "formatversion": 2,
        }
        if clcontinue:
            params["clcontinue"] = clcontinue
        data = _request_json(params)
        pages = data.get("query", {}).get("pages", [])
        for page in pages:
            for cat in page.get("categories", []):
                title_value = cat.get("title", "")
                if title_value.startswith("Category:"):
                    titles.append(title_value[len("Category:"):])
        cont = data.get("continue", {}).get("clcontinue")
        if not cont:
            break
        clcontinue = cont
    return titles


def parse_infobox(soup: BeautifulSoup) -> dict[str, str]:
    """Extract key/value pairs from the wiki infobox table.

    The Stardew Valley wiki's infobox is ``<table id="infoboxtable">`` and
    marks keys with ``<td id="infoboxsection">`` and values with
    ``<td id="infoboxdetail">`` — it uses NO ``infobox`` class and NO ``th``
    keys. Generic MediaWiki infoboxes (``class="infobox"`` + ``th``/``td``)
    are still supported as a fallback so both layouts parse.
    """
    table = soup.find("table", id="infoboxtable")
    if table is None:
        table = soup.find(
            "table",
            class_=lambda c: bool(c) and "infobox" in (c if isinstance(c, list) else [c]),
        )
    if table is None:
        return {}
    fields: dict[str, str] = {}
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            # Title/image/description rows span the table — not key/value.
            continue
        header = cells[0]
        if header.name == "td" and header.get("id") != "infoboxsection":
            # First cell is a td but not a section key (e.g. colspan rows).
            continue
        key = header.get_text(" ", strip=True)
        if not key:
            continue
        value_cell = row.find("td", id="infoboxdetail") or cells[-1]
        if value_cell is header:
            continue
        value = value_cell.get_text(" ", strip=True)
        fields[key] = value or value_cell.get("title") or ""
    return fields


def detect_kind(infobox: dict[str, str], categories: Iterable[str]) -> str:
    """Classify a wiki page into one of the structured entity kinds."""
    categories_lower = {c.lower() for c in categories}

    # Infobox-based detection. Accept a single strong cue to avoid dropping
    # minimal infoboxes (e.g. an NPC page that only lists "Birthday").
    if infobox:
        for kind, cues in KIND_INFOCUES.items():
            for cue in cues:
                if cue in infobox:
                    return kind

    # Fish category matching uses a full-match regex so "Fishing Poles" and
    # "Fish Tanks" (equipment) never match, while "Pond fish" and "Lake fish"
    # always do.
    if any(_FISH_CATEGORY_RE.match(c) for c in categories_lower):
        return "fish"

    # Fall back to category matching for pages without a rich infobox.
    for kind, prefix in (
        ("crop", "crops"),
        ("npc", "villagers"),
        ("npc", "marriageable"),
        ("location", "locations"),
        ("bundle", "bundles"),
        ("tool", "tools"),
        ("monster", "monsters"),
    ):
        if any(c.startswith(prefix) for c in categories_lower):
            return kind
    return "article"


def extract_body_text(soup: BeautifulSoup) -> str:
    """Strip scripts/tables/nav and return the readable body."""
    for tag in soup(["script", "style"]):
        tag.decompose()
    for selector in ("table", "div.thumb", "div.navbox", "div#toc", "div.printfooter"):
        for hit in soup.select(selector):
            hit.decompose()
    text = soup.get_text(" ", strip=True)
    text = unescape(text)
    return " ".join(text.split())


def build_aliases(title: str, infobox: dict[str, str], soup: BeautifulSoup) -> list[str]:
    """Collect alternate names used by the wiki for fuzzy matching."""
    aliases: set[str] = set()
    # Wiki redirect entries (top of page in some skins) mention aliases.
    for span in soup.select(".mw-redirectedfrom"):
        aliases.add(span.get_text(" ", strip=True))
    # Any "Also known as" / "Nickname" infobox field.
    for key in ("Also Known As", "Nickname", "Aliases"):
        if infobox.get(key):
            for piece in infobox[key].replace("&", ",").split(","):
                cleaned = piece.strip()
                if cleaned and cleaned.lower() != title.lower():
                    aliases.add(cleaned)
    return sorted(aliases)


# Sortable-table values embed a hidden "data-sort-value=\"15\">" prefix in
# their text; strip it so spoken/priced values stay clean ("15g", not
# "data-sort-value=\"15\"> 15g"). Control characters (NUL etc. from wiki
# markup) are removed so values never corrupt TTS or downstream text tools.
_DATA_SORT_RE = re.compile(r'data-sort-value=\\?"[^\\"]*\\?">?')
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_value(text: str) -> str:
    return " ".join(_CONTROL_RE.sub(" ", _DATA_SORT_RE.sub(" ", text)).split())


def normalise_value(value: str) -> str:
    return _clean_value(value)


def persist_page(
    connection: sqlite3.Connection,
    title: str,
    page_id: int | None,
    url: str,
    infobox: dict[str, str],
    body: str,
    categories: list[str],
) -> str:
    """Insert clean page row. Returns canonical lower name."""
    canonical = title.lower()
    connection.execute(
        """
        INSERT INTO stardew_page_clean (title, page_id, url, infobox_json, body_text, categories, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(title) DO UPDATE SET
            page_id=excluded.page_id,
            url=excluded.url,
            infobox_json=excluded.infobox_json,
            body_text=excluded.body_text,
            categories=excluded.categories,
            fetched_at=excluded.fetched_at
        """,
        (
            title,
            page_id,
            url,
            json.dumps(infobox, ensure_ascii=False),
            body[:20000],  # cap to keep DB responsive
            json.dumps(categories, ensure_ascii=False),
            time.time(),
        ),
    )
    return canonical


def persist_entity(
    connection: sqlite3.Connection,
    title: str,
    canonical: str,
    kind: str,
    infobox: dict[str, str],
    aliases: list[str],
) -> None:
    connection.execute(
        """
        INSERT INTO stardew_entity (canonical_name, display_name, kind, aliases, source_page, attributes, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(canonical_name) DO UPDATE SET
            display_name=excluded.display_name,
            kind=excluded.kind,
            aliases=excluded.aliases,
            source_page=excluded.source_page,
            attributes=excluded.attributes,
            fetched_at=excluded.fetched_at
        """,
        (
            canonical,
            title,
            kind,
            json.dumps(aliases, ensure_ascii=False),
            title,
            json.dumps(infobox, ensure_ascii=False),
            time.time(),
        ),
    )


def persist_vocabulary(
    connection: sqlite3.Connection,
    word: str,
    display: str,
    kind: str,
) -> None:
    if not word:
        return
    connection.execute(
        """
        INSERT INTO stardew_vocabulary (word, display, kind)
        VALUES (?, ?, ?)
        ON CONFLICT(word) DO UPDATE SET
            display=COALESCE(excluded.display, stardew_vocabulary.display),
            kind=COALESCE(excluded.kind, stardew_vocabulary.kind)
        """,
        (word.lower(), display, kind),
    )


def strip_tables_and_return_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def enrich_vocabulary_from_page(
    connection: sqlite3.Connection,
    title: str,
    aliases: list[str],
    infobox: dict[str, str],
    kind: str,
) -> None:
    persist_vocabulary(connection, title, title, kind)
    for alias in aliases:
        persist_vocabulary(connection, alias, title, kind)


def build_corpus(limit: int | None = DEFAULT_PAGE_LIMIT, refresh: bool = False) -> int:
    connection = sqlite3.connect(str(CACHE_DB_PATH))
    connection.row_factory = sqlite3.Row
    # Wait instead of failing when another process (e.g. the assistant or a
    # monitoring script) holds the DB — transient locks otherwise abort pages.
    connection.execute("PRAGMA busy_timeout = 10000")
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    StardewWikiIndex._init_extended_schema(connection)

    if refresh:
        connection.execute("DELETE FROM stardew_vocabulary")
        connection.execute("DELETE FROM stardew_entity")
        connection.execute("DELETE FROM stardew_page_clean")
        connection.commit()

    titles = list_all_titles(limit=limit)
    print(f"[Build] {len(titles)} candidate wiki pages to ingest.")

    # Resume support: skip pages already parsed and stored by a previous
    # (interrupted) run. Only valid without --refresh, which wipes first.
    already = set()
    if not refresh:
        already = {
            row[0].lower()
            for row in connection.execute("SELECT title FROM stardew_page_clean").fetchall()
        }
        if already:
            print(f"[Build] Resuming: {len(already)} pages already stored will be skipped.")

    seen_canonical: set[str] = set()
    processed = 0
    stored = 0
    skipped = 0
    failed = 0
    for title in titles:
        canonical = title.lower()
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        if canonical in already:
            skipped += 1
            continue
        try:
            html = fetch_page_html(title)
            soup = strip_tables_and_return_soup(html)
            infobox = {key: normalise_value(value) for key, value in parse_infobox(soup).items()}
            body = extract_body_text(soup)
            # Skip the categories request when the infobox already tells us
            # the kind — that halves HTTP round-trips for entity pages.
            kind = detect_kind(infobox, ())
            if kind == "article":
                categories = fetch_page_categories(title)
                kind = detect_kind(infobox, categories)
            else:
                categories = []
            aliases = build_aliases(title, infobox, soup)
            url = f"https://stardewvalleywiki.com/{title.replace(' ', '_')}"
            persist_page(connection, title, None, url, infobox, body, categories)
            if infobox or kind != "article":
                persist_entity(connection, title, canonical, kind, infobox, aliases)
            enrich_vocabulary_from_page(connection, title, aliases, infobox, kind)
            stored += 1
        except Exception as exc:  # surface failures but keep going
            failed += 1
            print(f"[Build] ! {title}: {exc}")
        processed += 1
        if processed % 25 == 0:
            connection.commit()
            print(f"[Build] {processed}/{len(titles)} processed ({stored} stored, {skipped} skipped, {failed} failed)")
        time.sleep(REQUEST_PAUSE_SECONDS)

    connection.commit()
    connection.execute("INSERT INTO stardew_corpus_meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", ("last_built", str(time.time())))
    connection.execute("INSERT INTO stardew_corpus_meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", ("page_count", str(stored)))
    connection.commit()
    # Refresh the body-text FTS mirror used by the structured RAG path.
    from src.wiki_index import StardewWikiIndex as _Index
    mirrored = _Index.rebuild_page_fts(connection)
    print(f"[Build] FTS mirror rebuilt with {mirrored} rows.")
    print(f"[Build] Done. processed={processed} stored={stored} skipped={skipped} failed={failed}")
    return stored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local Stardew wiki corpus.")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of pages ingested")
    parser.add_argument("--refresh", action="store_true", help="wipe structured tables before rebuilding")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_corpus(limit=args.limit, refresh=args.refresh)


if __name__ == "__main__":
    main()
