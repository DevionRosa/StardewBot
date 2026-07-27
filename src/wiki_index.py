"""Local Stardew Valley wiki retrieval with structured lookups.

Two layers:

* Legacy RAG (``search`` / ``context`` / ``answer_from_wiki``) — used as a
  fallback when the structured lookups cannot answer a question.
* Structured corpus populated by ``scripts/build_wiki_corpus.py`` —
  ``stardew_entity`` rows hold typed facts (crop, fish, NPC, etc.) that the
  bot can answer with zero LLM involvement, eliminating the largest
  hallucination source.

The structured lookups are intentionally small and fast (single-digit
millisecond lookups) so the runtime can stay offline and local.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests

try:
    from bs4 import BeautifulSoup  # type: ignore
    _BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None  # type: ignore
    _BS4_AVAILABLE = False

from .config import BASE_DIR

WIKI_API_URL = "https://stardewvalleywiki.com/mediawiki/api.php"
WIKI_BASE_URL = "https://stardewvalleywiki.com"
CACHE_DIR = BASE_DIR / "data"
CACHE_DB_PATH = CACHE_DIR / "stardew_wiki_cache.sqlite"
SEARCH_LIMIT = 5
MAX_EXTRACT_CHARS = 5000

STOPWORDS = {
    "a", "about", "after", "all", "an", "and", "any", "are", "at", "be", "can", "do",
    "does", "for", "from", "get", "have", "how", "i", "in", "is", "it", "me", "my",
    "of", "on", "or", "should", "tell", "the", "their", "them", "there", "this", "to",
    "what", "when", "where", "which", "who", "why", "with", "you", "your",
}

TOKEN_RE = re.compile(r"[a-z0-9']+")
TAG_RE = re.compile(r"<[^>]+>")

# Wiki infobox kind detection cues. Keep aligned with scripts/build_wiki_corpus.py.
KIND_INFOCUES: dict[str, tuple[str, ...]] = {
    "crop": ("Growth Time", "Regrowth", "Seed", "Sell Price", "Season"),
    "fish": ("Time", "Location", "Weather", "Difficulty", "Behavior"),
    "npc": ("Birthday", "Lives In", "Address", "Marriage"),
    "location": ("Inhabitants", "Features", "Open Hours"),
    "bundle": ("Bundles", "Reward", "Requirements"),
    "tool": ("Material", "Upgrades", "Uses"),
    "monster": ("HP", "Damage", "Defense", "Drops"),
}

# Mapping from common intent verbs to structured fields, used by ``compose_answer``.
INTENT_FIELDS: dict[str, tuple[str, ...]] = {
    "when": ("Season", "Time", "Weather", "Growth Time"),
    "where": ("Location", "Found In", "Lives In"),
    "how_long": ("Growth Time", "Time"),
    "how_much": ("Sell Price", "Price", "Energy", "Healing"),
    "gift": ("Loved Gifts", "Liked Gifts", "Disliked Gifts", "Hated Gifts"),
    "love": ("Loved Gifts",),
    "like": ("Liked Gifts",),
    "dislike": ("Disliked Gifts",),
    "hate": ("Hated Gifts",),
    "marriage": ("Marriage", "Spouse"),
    "birthday": ("Birthday",),
}


@dataclass(frozen=True)
class WikiPassage:
    title: str
    extract: str
    url: str
    score: float


@dataclass(frozen=True)
class StardewEntity:
    canonical_name: str
    display_name: str
    kind: str
    aliases: tuple[str, ...]
    source_page: str
    attributes: dict[str, str]

    def get(self, key: str, default: str = "") -> str:
        # Case-insensitive lookup because wiki infobox fields vary in casing.
        for attr_key, attr_value in self.attributes.items():
            if attr_key.lower() == key.lower():
                return attr_value
        return default


class StardewWikiIndex:
    """SQLite-backed wiki cache with structured and free-form retrieval."""

    def __init__(self, db_path: Optional[Path] = None):
        # Resolve ``CACHE_DB_PATH`` at call time, not at class definition
        # time. Earlier the default was a module global evaluated once at
        # import, which made monkey-patching ``wiki_index.CACHE_DB_PATH``
        # ineffective and confused tests that patched the path to point at a
        # tmp DB.
        if db_path is None:
            db_path = CACHE_DB_PATH
        self.db_path = db_path
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.connection.close()

    # ------------------------------------------------------------------ schemas

    def _init_schema(self) -> None:
        """Create all legacy + structured tables and FTS5 mirror if available."""
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_pages (
                title TEXT PRIMARY KEY,
                page_id INTEGER,
                url TEXT NOT NULL,
                extract TEXT NOT NULL,
                source_query TEXT,
                fetched_at REAL NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_wiki_pages_title ON wiki_pages(title)"
        )

        StardewWikiIndex._init_extended_schema(self.connection)
        self.connection.commit()

    @staticmethod
    def _init_extended_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stardew_page_clean (
                title TEXT PRIMARY KEY,
                page_id INTEGER,
                url TEXT NOT NULL,
                infobox_json TEXT NOT NULL,
                body_text TEXT NOT NULL,
                categories TEXT NOT NULL,
                fetched_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stardew_entity (
                canonical_name TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                kind TEXT NOT NULL,
                aliases TEXT NOT NULL,
                source_page TEXT NOT NULL,
                attributes TEXT NOT NULL,
                fetched_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_stardew_entity_kind ON stardew_entity(kind)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_stardew_entity_source ON stardew_entity(source_page)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stardew_vocabulary (
                word TEXT PRIMARY KEY,
                display TEXT NOT NULL,
                kind TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stardew_corpus_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )

        # Optional full-text index — falls back silently when FTS5 is unavailable.
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS stardew_entity_fts USING fts5(
                    canonical_name, display_name, kind, aliases, attributes,
                    content='', tokenize='porter unicode61'
                )
                """
            )
        except sqlite3.OperationalError:
            pass

    # ------------------------------------------------------------ helpers ---

    @staticmethod
    def _normalize_query(query: str) -> list[str]:
        tokens = [token for token in TOKEN_RE.findall(query.lower()) if token not in STOPWORDS]
        if tokens:
            return tokens
        return [token for token in TOKEN_RE.findall(query.lower()) if token]

    @staticmethod
    def _title_to_url(title: str) -> str:
        page_name = quote(title.replace(" ", "_"), safe=":/_-()[]!,'")
        return f"{WIKI_BASE_URL}/{page_name}"

    @staticmethod
    def _html_to_text(html_text: str) -> str:
        if not _BS4_AVAILABLE:
            # Regex fallback when BeautifulSoup is unavailable. Strips
            # script/style blocks and any HTML tags, flattens whitespace.
            cleaned = re.sub(
                r"<script.*?</script>|<style.*?</style>",
                " ", html_text, flags=re.S | re.I,
            )
            cleaned = TAG_RE.sub(" ", cleaned)
            cleaned = unescape(cleaned)
            return " ".join(cleaned.split())
        soup = BeautifulSoup(html_text, "html.parser")  # type: ignore[misc]
        for tag in soup(["script", "style"]):
            tag.decompose()
        for selector in ("table", "div.thumb", "div.navbox", "div#toc"):
            for tag in soup.select(selector):
                tag.decompose()
        text = unescape(soup.get_text(" ", strip=True))
        return " ".join(text.split())

    @staticmethod
    def _shorten(text: str, max_chars: int = 260) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0].strip() + "..."

    @staticmethod
    def _extract_infobox(html_text: str) -> dict[str, str]:
        if not _BS4_AVAILABLE:
            # Without BeautifulSoup we cannot reliably parse infobox tables,
            # so return empty rather than risk garbage extraction.
            return {}
        soup = BeautifulSoup(html_text, "html.parser")  # type: ignore[misc]
        table = soup.find(
            "table",
            class_=lambda cls: bool(cls) and "infobox" in (cls if isinstance(cls, list) else [cls]),
        )
        if table is None:
            return {}
        fields: dict[str, str] = {}
        for row in table.find_all("tr"):
            header = row.find("th")
            cell = row.find("td")
            if header is None or cell is None:
                continue
            key = header.get_text(" ", strip=True)
            if not key:
                continue
            value = cell.get_text(" ", strip=True)
            fields[key] = value or cell.get("title") or ""
        return {key: " ".join(value.split()) for key, value in fields.items()}

    @staticmethod
    def _detect_kind(infobox: dict[str, str], categories_lower: set[str]) -> str:
        # Single-cue infobox detection: any one matching clue is enough,
        # matching ``build_wiki_corpus.detect_kind`` so the index and the
        # builder stay in sync.
        if infobox:
            for kind, cues in KIND_INFOCUES.items():
                for cue in cues:
                    if cue in infobox:
                        return kind
        for kind, prefix in (
            ("crop", "crops"),
            ("fish", "fish"),
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

    # ----------------------------------------------------------- legacy RAG ---

    def _score_passage(self, query_tokens: set[str], title: str, extract: str) -> float:
        title_tokens = set(self._normalize_query(title))
        extract_tokens = set(self._normalize_query(extract[:2000]))
        title_overlap = len(query_tokens & title_tokens)
        extract_overlap = len(query_tokens & extract_tokens)
        phrase_bonus = 0.0
        normalized_title = title.lower()
        if normalized_title in " ".join(query_tokens):
            phrase_bonus += 4.0
        if any(token == normalized_title for token in query_tokens):
            phrase_bonus += 5.0
        return (title_overlap * 4.0) + extract_overlap + phrase_bonus

    def _load_cached_pages(self) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                "SELECT title, page_id, url, extract, source_query FROM wiki_pages"
            )
        )

    def _search_cached(self, query: str, limit: int) -> list[WikiPassage]:
        query_tokens = set(self._normalize_query(query))
        if not query_tokens:
            return []

        passages: list[WikiPassage] = []
        for row in self._load_cached_pages():
            score = self._score_passage(query_tokens, row["title"], row["extract"])
            if score <= 0:
                continue
            passages.append(
                WikiPassage(
                    title=row["title"],
                    extract=row["extract"],
                    url=row["url"],
                    score=score,
                )
            )

        passages.sort(key=lambda passage: (-passage.score, passage.title.lower()))
        return passages[:limit]

    def _search_remote_titles(self, query: str, limit: int) -> list[str]:
        search_terms = " ".join(self._normalize_query(query)) or query.strip()
        if not search_terms:
            return []
        response = requests.get(
            WIKI_API_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": search_terms,
                "srlimit": max(limit, SEARCH_LIMIT),
                "format": "json",
                "formatversion": 2,
                "utf8": 1,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("query", {}).get("search", [])
        return [result["title"] for result in results if result.get("title")]

    def _fetch_remote_page(self, title: str, source_query: str | None = None) -> WikiPassage | None:
        response = requests.get(
            WIKI_API_URL,
            params={
                "action": "parse",
                "page": title,
                "prop": "text",
                "format": "json",
                "formatversion": 2,
                "utf8": 1,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        parse_block = data.get("parse", {})
        if not parse_block:
            return None
        html_text = parse_block.get("text", "")
        if not html_text:
            return None

        extract = self._html_to_text(html_text)
        if len(extract) > MAX_EXTRACT_CHARS:
            extract = extract[:MAX_EXTRACT_CHARS].rsplit(" ", 1)[0].strip() + "..."
        page_title = parse_block.get("title") or title
        page_id = parse_block.get("pageid")
        url = self._title_to_url(page_title)
        self._store_page(page_title, page_id, url, extract, source_query)
        return WikiPassage(title=page_title, extract=extract, url=url, score=0.0)

    def _store_page(self, title: str, page_id: int | None, url: str, extract: str, source_query: str | None) -> None:
        self.connection.execute(
            """
            INSERT INTO wiki_pages (title, page_id, url, extract, source_query, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(title) DO UPDATE SET
                page_id=excluded.page_id,
                url=excluded.url,
                extract=excluded.extract,
                source_query=excluded.source_query,
                fetched_at=excluded.fetched_at
            """,
            (title, page_id, url, extract, source_query, time.time()),
        )
        self.connection.commit()

    def search(self, query: str, limit: int = SEARCH_LIMIT) -> list[WikiPassage]:
        cached = self._search_cached(query, limit)
        try:
            remote_titles = self._search_remote_titles(query, limit)
        except Exception as exc:
            print(f"[Wiki] Search failed for '{query}': {exc}")
            return cached
        for title in remote_titles[:limit]:
            try:
                self._fetch_remote_page(title, source_query=query)
            except Exception as exc:
                print(f"[Wiki] Fetch failed for '{title}': {exc}")
        merged = self._search_cached(query, limit * 2)
        if not merged:
            return cached
        deduped: list[WikiPassage] = []
        seen: set[str] = set()
        for passage in merged:
            key = passage.title.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(passage)
            if len(deduped) >= limit:
                break
        return deduped or cached

    def context(self, query: str, limit: int = SEARCH_LIMIT) -> str:
        passages = self.search(query, limit)
        if not passages:
            return ""
        return "\n\n".join(f"[{passage.title}] {passage.extract}" for passage in passages[:limit])

    def answer_from_context(self, query: str, limit: int = SEARCH_LIMIT) -> str:
        """Return a strict excerpt from the top passage (used as a fallback)."""
        passages = self.search(query, limit)
        if not passages:
            return ""
        passage = passages[0]
        return self._shorten(passage.extract, max_chars=320)

    # ----------------------------------------------------------- entity API ---

    def _row_to_entity(self, row: sqlite3.Row) -> StardewEntity:
        return StardewEntity(
            canonical_name=row["canonical_name"],
            display_name=row["display_name"],
            kind=row["kind"],
            aliases=tuple(json.loads(row["aliases"] or "[]")),
            source_page=row["source_page"],
            attributes={key: value for key, value in json.loads(row["attributes"] or "{}").items()},
        )

    @staticmethod
    def _candidate_terms(query: str) -> list[str]:
        normalised = query.lower()
        # Strip leading questions verbs that often precede the entity name.
        for prefix in (
            "tell me about ",
            "what about ",
            "what is ",
            "who is ",
            "where is ",
            "where can i find ",
            "where to find ",
            "when does ",
            "how long does ",
            "how much does ",
            "can i ",
            "do i ",
        ):
            if normalised.startswith(prefix):
                normalised = normalised[len(prefix):]
                break
        # Strip trailing punctuation and reciprocate question marks.
        normalised = normalised.strip(" ?!.,")
        tokens = StardewWikiIndex._normalize_query(normalised)
        if tokens:
            # Try 1-grams and 2-gram phrases from the end of the question.
            return [normalised] + tokens + [" ".join(tokens[-2:]), " ".join(tokens[-3:])]
        return [normalised]

    def lookup_entity(self, query: str) -> StardewEntity | None:
        """Best-effort structured lookup. Returns the highest-scoring row."""
        candidates = [candidate for candidate in self._candidate_terms(query) if candidate]
        for candidate in candidates:
            row = self.connection.execute(
                "SELECT * FROM stardew_entity WHERE canonical_name = ? OR display_name = ? LIMIT 1",
                (candidate, candidate),
            ).fetchone()
            if row:
                return self._row_to_entity(row)

        # Alias match (JSON LIKE). Cheap fall-through that still beats LLM guessing.
        for candidate in candidates:
            pattern = f'%"{candidate.lower()}"%'
            row = self.connection.execute(
                "SELECT * FROM stardew_entity WHERE lower(aliases) LIKE ? LIMIT 1",
                (pattern,),
            ).fetchone()
            if row:
                return self._row_to_entity(row)

        # FTS5 search if present, with structured LIKE as fallback.
        try:
            query_text = " ".join(candidates) or query
            rows = self.connection.execute(
                "SELECT se.* FROM stardew_entity_fts fts JOIN stardew_entity se "
                "ON se.canonical_name = fts.canonical_name "
                "WHERE stardew_entity_fts MATCH ? ORDER BY rank LIMIT 1",
                (query_text,),
            ).fetchall()
            if rows:
                return self._row_to_entity(rows[0])
        except sqlite3.OperationalError:
            pass

        last = candidates[-1] if candidates else query
        if last:
            like = f"%{last.lower()}%"
            rows = self.connection.execute(
                "SELECT * FROM stardew_entity WHERE lower(display_name) LIKE ? "
                "ORDER BY CASE WHEN lower(display_name) = ? THEN 0 "
                "WHEN lower(display_name) LIKE ? THEN 1 ELSE 2 END, "
                "length(display_name) ASC LIMIT 1",
                (like, last.lower(), f"{like}%"),
            ).fetchall()
            if rows:
                return self._row_to_entity(rows[0])

            # Last resort: scan attributes JSON. Rank by token overlap count.
            best_row: sqlite3.Row | None = None
            best_score = 0
            query_token_set = set(self._normalize_query(query))
            for row in self.connection.execute(
                "SELECT * FROM stardew_entity WHERE lower(attributes) LIKE ?",
                (like,),
            ).fetchall():
                attrs_text = " ".join(row["attributes"].lower().split())
                row_tokens = set(self._normalize_query(attrs_text))
                score = len(query_token_set & row_tokens)
                if score > best_score:
                    best_score = score
                    best_row = row
            if best_row is not None:
                return self._row_to_entity(best_row)
        return None

    def lookup_body(self, title: str, max_chars: int = 2000) -> str:
        """Return the cleaned body text for ``title`` (None-safe)."""
        if not title:
            return ""
        row = self.connection.execute(
            "SELECT body_text FROM stardew_page_clean WHERE title = ?",
            (title,),
        ).fetchone()
        if not row:
            return ""
        text = row["body_text"] or ""
        if len(text) <= max_chars:
            return text
        # Truncate at the nearest sentence boundary to keep the excerpt fluent.
        truncated = text[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars // 2:
            return truncated[: last_period + 1].strip()
        return truncated.strip() + "..."

    def search_entities(self, query: str, limit: int = 5) -> list[StardewEntity]:
        """Return all entities that match the query, ranked by relevance."""
        candidates = [candidate for candidate in self._candidate_terms(query) if candidate]
        entities: dict[str, StardewEntity] = {}

        try:
            fts_query = " OR ".join(candidates) if candidates else query
            rows = self.connection.execute(
                "SELECT se.* FROM stardew_entity_fts fts JOIN stardew_entity se "
                "ON se.canonical_name = fts.canonical_name "
                "WHERE stardew_entity_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_query, limit),
            ).fetchall()
            for row in rows:
                entities[row["canonical_name"]] = self._row_to_entity(row)
        except sqlite3.OperationalError:
            pass

        for candidate in candidates:
            like = f"%{candidate.lower()}%"
            for row in self.connection.execute(
                "SELECT * FROM stardew_entity WHERE lower(display_name) LIKE ? OR "
                "lower(aliases) LIKE ? OR lower(canonical_name) LIKE ? LIMIT ?",
                (like, like, like, limit),
            ).fetchall():
                entities.setdefault(row["canonical_name"], self._row_to_entity(row))
                if len(entities) >= limit:
                    return list(entities.values())[:limit]
        return list(entities.values())[:limit]

    @staticmethod
    def compose_answer(entity: StardewEntity, query: str) -> str:
        """Build a deterministic, dangerously-correct answer from the infobox."""
        q_lower = query.lower()
        # Pull fields relevant to the question's intent.
        fields = list(entity.attributes.items())
        priority_keys: list[str] = []
        for intent, keys in INTENT_FIELDS.items():
            if intent in q_lower:
                priority_keys.extend(keys)
                break
        # Always start with the most relevant fields to the intent.
        priority_keys.extend([key for key, _ in fields if key not in priority_keys])
        # Cap emitted attributes to keep spoken answers short.
        parts: list[str] = []
        for key in priority_keys:
            value = entity.get(key)
            if not value:
                continue
            formatted_key = key.replace("_", " ").strip()
            parts.append(f"{formatted_key}: {value}")
            if len(parts) >= 6:
                break
        if not parts:
            return f"{entity.display_name} is in the wiki, but I do not have a structured field for that question."

        head = f"{entity.display_name}"
        if entity.aliases:
            head += f" (a.k.a. {', '.join(entity.aliases[:2])})"
        return f"{head}. " + "; ".join(parts) + "."

    def get_vocabulary(self) -> set[str]:
        """Return every Stardew vocabulary word (lowercase) for STT correction."""
        rows = self.connection.execute("SELECT word FROM stardew_vocabulary").fetchall()
        return {row["word"] for row in rows if row["word"]}

    def get_vocabulary_with_display(self) -> dict[str, str]:
        rows = self.connection.execute("SELECT word, display FROM stardew_vocabulary").fetchall()
        return {row["word"]: row["display"] for row in rows if row["word"]}

    # ----------------------------------------------------------- diagnostics ---

    def corpus_stats(self) -> dict[str, int]:
        stats = {
            "page_clean": int(self.connection.execute("SELECT COUNT(*) FROM stardew_page_clean").fetchone()[0]),
            "entity": int(self.connection.execute("SELECT COUNT(*) FROM stardew_entity").fetchone()[0]),
            "vocabulary": int(self.connection.execute("SELECT COUNT(*) FROM stardew_vocabulary").fetchone()[0]),
            "legacy_pages": int(self.connection.execute("SELECT COUNT(*) FROM wiki_pages").fetchone()[0]),
        }
        return stats

    def needs_internet_for_query(self, query: str) -> bool:
        """True if no cached page would help — used to decide live fetch necessity."""
        return not self._search_cached(query, limit=1) and self.lookup_entity(query) is None


# ---------------------------------------------------------------- module API

_INDEX: StardewWikiIndex | None = None
_INDEX_LOCK = threading.Lock()


def get_wiki_index() -> StardewWikiIndex:
    """Thread-safe module-level singleton."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _INDEX_LOCK:
        if _INDEX is None:
            _INDEX = StardewWikiIndex()
        return _INDEX


def get_wiki_context(query: str, limit: int = SEARCH_LIMIT) -> str:
    return get_wiki_index().context(query, limit)


def answer_from_wiki(query: str, limit: int = SEARCH_LIMIT) -> str:
    return get_wiki_index().answer_from_context(query, limit)


def is_probably_stardew_query(query: str) -> bool:
    return bool(get_wiki_context(query, limit=1))


# Validators used by tests + chat. Use sparingly — these hit SQLite each time.

def get_stardew_vocabulary() -> set[str]:
    return get_wiki_index().get_vocabulary()


def lookup_stardew_entity(query: str) -> StardewEntity | None:
    return get_wiki_index().lookup_entity(query)


def search_stardew_entities(query: str, limit: int = 5) -> list[StardewEntity]:
    return get_wiki_index().search_entities(query, limit=limit)


def compose_stardew_answer(query: str) -> str | None:
    entity = get_wiki_index().lookup_entity(query)
    if entity is None:
        return None
    return StardewWikiIndex.compose_answer(entity, query)


def lookup_stardew_body(query: str) -> str:
    """Convenience wrapper that resolves a query to a body excerpt via entity rows."""
    entity = get_wiki_index().lookup_entity(query)
    if entity is None:
        return ""
    return get_wiki_index().lookup_body(entity.source_page)
