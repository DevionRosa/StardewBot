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
MAX_EXTRACT_CHARS = 12000

STOPWORDS = {
    "a", "about", "after", "all", "an", "and", "any", "are", "at", "be", "can", "catch", "do",
    "does", "find", "for", "from", "get", "have", "how", "i", "in", "is", "it", "me", "my",
    "of", "on", "or", "should", "tell", "the", "their", "them", "there", "this", "to",
    "what", "when", "where", "which", "who", "why", "with", "you", "your",
}

TOKEN_RE = re.compile(r"[a-z0-9']+")
TAG_RE = re.compile(r"<[^>]+>")
# Sortable-table values embed a hidden "data-sort-value=\"15\">" prefix in
# their text; strip it so prices read "15g", not "data-sort-value=\"15\"> 15g".
_DATA_SORT_RE = re.compile(r'data-sort-value=\\?"[^\\"]*\\?">?')

# Wiki infobox kind detection cues. Keep aligned with scripts/build_wiki_corpus.py.
#
# Cues are ordered from most to least distinctive because detection returns
# on the FIRST matching cue and kinds are iterated in dict order. Generic
# fields like "Sell Price" or "Season" appear on both crop and fish infoboxes
# and previously caused fish pages to classify as crops.
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
        # The builder script writes to this DB while the assistant runs;
        # wait out transient locks instead of raising "database is locked".
        try:
            self.connection.execute("PRAGMA busy_timeout = 10000")
        except sqlite3.Error:
            pass
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
        self._ensure_fts_populated()
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
        # Body-text FTS mirror over ``stardew_page_clean`` — powers the
        # structured RAG path so questions fall back to real wiki text from
        # the 2,000-page corpus instead of a near-empty legacy cache.
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS stardew_page_clean_fts USING fts5(
                    title, body_text,
                    content='', tokenize='porter unicode61'
                )
                """
            )
        except sqlite3.OperationalError:
            pass

    @staticmethod
    def rebuild_page_fts(connection: sqlite3.Connection) -> int:
        """Rebuild the body-text FTS mirror from ``stardew_page_clean``.

        The mirror is content-less (``content=''``), so rows must be inserted
        explicitly with the source table's rowid. Rebuilds are cheap (~2k
        rows) and keep the mirror consistent after corpus (re)builds.

        Returns the number of rows indexed and records it in
        ``stardew_corpus_meta`` — contentless FTS5 tables report COUNT(*) as
        0, so the flag is the only reliable population signal.
        """
        try:
            # The mirror copies the source table 1:1 by rowid, so the source
            # count after the insert is the indexed count. (cursor.rowcount
            # is unreliable for INSERT...SELECT in Python's sqlite3.)
            # Contentless FTS5 tables reject plain ``DELETE FROM``; the
            # special 'delete-all' command is the supported way to clear one.
            connection.execute(
                "INSERT INTO stardew_page_clean_fts(stardew_page_clean_fts) VALUES('delete-all')"
            )
            connection.execute(
                """
                INSERT INTO stardew_page_clean_fts(rowid, title, body_text)
                SELECT rowid, title, body_text FROM stardew_page_clean
                """
            )
            inserted = int(
                connection.execute("SELECT COUNT(*) FROM stardew_page_clean").fetchone()[0]
            )
            connection.commit()
            connection.execute(
                "INSERT INTO stardew_corpus_meta(key, value) VALUES('fts_row_count', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(inserted),),
            )
            connection.commit()
            return inserted
        except sqlite3.OperationalError:
            # FTS5 unavailable — structured RAG silently degrades to legacy.
            return 0

    def _ensure_fts_populated(self) -> None:
        """Self-heal the FTS mirror if the corpus outran it.

        Covers DBs built before the mirror existed or partially rebuilt:
        when the recorded mirror row count lags the source table, rebuild
        once at startup (~a second for the full corpus) instead of answering
        from an empty index forever.
        """
        try:
            source = self.connection.execute(
                "SELECT COUNT(*) FROM stardew_page_clean"
            ).fetchone()[0]
            flag_row = self.connection.execute(
                "SELECT value FROM stardew_corpus_meta WHERE key = 'fts_row_count'"
            ).fetchone()
            mirror_count = int(flag_row[0]) if flag_row and flag_row[0] else -1
            if mirror_count < source:
                rebuilt = self.rebuild_page_fts(self.connection)
                print(f"[Wiki] Rebuilt page FTS mirror: {rebuilt} rows")
        except (sqlite3.OperationalError, ValueError):
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
        """Parse the wiki infobox.

        The Stardew Valley wiki's infobox is ``<table id="infoboxtable">`` and
        marks keys with ``<td id="infoboxsection">`` and values with
        ``<td id="infoboxdetail">`` — it uses NO ``infobox`` class and NO
        ``th`` keys. Generic MediaWiki infoboxes (``class="infobox"`` with
        ``th``/``td``) are still supported as a fallback so both layouts parse.
        """
        if not _BS4_AVAILABLE:
            # Without BeautifulSoup we cannot reliably parse infobox tables,
            # so return empty rather than risk garbage extraction.
            return {}
        soup = BeautifulSoup(html_text, "html.parser")  # type: ignore[misc]
        table = soup.find("table", id="infoboxtable")
        if table is None:
            table = soup.find(
                "table",
                class_=lambda cls: bool(cls) and "infobox" in (cls if isinstance(cls, list) else [cls]),
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
            value = " ".join(_DATA_SORT_RE.sub(" ", value).split())
            fields[key] = value or value_cell.get("title") or ""
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
        # Fish categories need a full match so "Fishing Poles" and "Fish
        # Tanks" (equipment) never match, while "Pond fish" and "Lake fish"
        # always do.
        if any(_FISH_CATEGORY_RE.match(c) for c in categories_lower):
            return "fish"
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
            # Slice generously past the limit, then trim to the last complete
            # sentence so the stored extract ends on a fluent boundary.
            extract = extract[: MAX_EXTRACT_CHARS + 1000].rsplit(".", 1)[0].strip() + "."
            extract = extract[:MAX_EXTRACT_CHARS]
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

    # ------------------------------------------------- structured RAG (FTS5) ---

    @staticmethod
    def _fts_escape(term: str) -> str:
        """Quote a term for FTS5 MATCH so hyphens/quotes never parse as syntax.

        A bare ``stardew-era`` tokenises as ``stardew`` NEAR ``era`` and
        embedded quotes can raise OperationalErrors. Double quotes are
        doubled inside the quoted string per FTS5 string rules.
        """
        escaped = term.replace('"', '""')
        return f'"{escaped}"'

    def _search_clean(self, query: str, limit: int) -> list[WikiPassage]:
        """Full-text search over the structured corpus (``stardew_page_clean``).

        Returns [] when the corpus is empty or FTS5 is unavailable — callers
        fall back to the legacy cache and remote search in that case.
        """
        terms = self._normalize_query(query)
        if not terms:
            return []
        match_query = " OR ".join(self._fts_escape(term) for term in terms)
        try:
            rows = self.connection.execute(
                """
                SELECT pc.title, pc.body_text, pc.url
                FROM stardew_page_clean_fts fts
                JOIN stardew_page_clean pc ON pc.rowid = fts.rowid
                WHERE stardew_page_clean_fts MATCH ?
                ORDER BY bm25(stardew_page_clean_fts)
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        term_set = set(terms)
        passages: list[WikiPassage] = []
        seen: set[str] = set()
        for row in rows:
            title_lower = row["title"].lower()
            if title_lower in seen:
                continue
            seen.add(title_lower)
            title_tokens = set(self._normalize_query(row["title"]))
            # Prefer pages whose title matches the question (entity pages)
            # over pages that merely mention the term in their body.
            if term_set <= title_tokens:
                score = 3.0
            elif term_set & title_tokens:
                score = 2.0
            else:
                score = 1.0
            passages.append(
                WikiPassage(
                    title=row["title"],
                    extract=row["body_text"] or "",
                    url=row["url"],
                    score=score,
                )
            )
        passages.sort(key=lambda passage: (-passage.score, passage.title.lower()))
        return passages[:limit]

    def search(self, query: str, limit: int = SEARCH_LIMIT) -> list[WikiPassage]:
        # Local-first: the structured corpus (built by
        # scripts/build_wiki_corpus.py) covers every game page, so prefer it
        # and skip slow live HTTP round-trips whenever it can answer. The
        # legacy cache + remote search remain the fallback for corpus gaps.
        local_hits = self._search_clean(query, limit)
        if local_hits:
            return local_hits
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

    def _entity_match_score(self, entity: StardewEntity, candidates: list[str]) -> int:
        """Rank how strongly ``entity`` matches the question's candidate terms.

        Exact canonical/display name beats alias beats substring. Prevents the
        last-resort substring scan from promoting incidental matches (e.g. a
        page that merely mentions the query word) over the real entity.
        """
        canonical = entity.canonical_name.lower()
        display = entity.display_name.lower()
        aliases = {alias.lower() for alias in entity.aliases}
        best = 0
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            if candidate == canonical or candidate == display:
                best = max(best, 4)
            elif candidate in aliases:
                best = max(best, 3)
            elif candidate in display or candidate in canonical:
                best = max(best, 2)
        return best

    def lookup_entity(self, query: str) -> StardewEntity | None:
        """Best-effort structured lookup. Returns the highest-scoring row."""
        candidates = [candidate for candidate in self._candidate_terms(query) if candidate]

        # Score every exact/alias match across ALL candidates instead of
        # returning on the first hit. Question-leading tokens can be STT
        # noise ("wheat can i fish bullhead" — Vosk misheard "where"), and
        # first-hit order would let that token's entity shadow the real
        # target mentioned later in the sentence. Later candidates get a
        # small position bonus so trailing entity nouns win ties.
        scored: dict[str, tuple[float, sqlite3.Row]] = {}

        def consider(row: sqlite3.Row, base: float, index: int) -> None:
            # Category pages ("Fish", "Crops", "Monsters", ...) share their
            # name with their kind. They are hub pages, not the specific
            # thing the user asked about — penalise them so "bullhead"
            # outranks a later generic "fish" token even with the position
            # bonus.
            display_lower = row["display_name"].lower()
            kind = row["kind"]
            if kind and display_lower in (kind, kind + "s", kind + "es"):
                base -= 25.0
            score = base + index * 0.5
            key = row["canonical_name"]
            if key not in scored or scored[key][0] < score:
                scored[key] = (score, row)

        for index, candidate in enumerate(candidates):
            term = candidate.strip().lower()
            if not term:
                continue
            row = self.connection.execute(
                "SELECT * FROM stardew_entity WHERE lower(canonical_name) = ? "
                "OR lower(display_name) = ? LIMIT 1",
                (term, term),
            ).fetchone()
            if row:
                consider(row, 100.0, index)
                continue
            pattern = f'%"{term}"%'
            row = self.connection.execute(
                "SELECT * FROM stardew_entity WHERE lower(aliases) LIKE ? LIMIT 1",
                (pattern,),
            ).fetchone()
            if row:
                consider(row, 80.0, index)

        if scored:
            best = max(scored.values(), key=lambda pair: pair[0])
            return self._row_to_entity(best[1])

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
            # Fuzzy: catch STT slips like 'bullhed' → 'bullhead'. Try exact,
            # prefix, substring, then a bounded Levenshtein scan (1 edit for
            # ≤4 chars, 2 for ≤7, 3 beyond — matching the STT corrector's
            # tiered thresholds). Keeps typos fixable without opening the
            # 2-edit door to unrelated short words.
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

            try:
                from ._transcribe_distance import levenshtein
            except ImportError:  # pragma: no cover - helper always ships
                def levenshtein(a: str, b: str) -> int:
                    if a == b:
                        return 0
                    if not a:
                        return len(b)
                    if not b:
                        return len(a)
                    prev = list(range(len(b) + 1))
                    for i, ca in enumerate(a, 1):
                        current = [i]
                        for j, cb in enumerate(b, 1):
                            current.append(min(current[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
                        prev = current
                    return prev[-1]

            len_lower = len(last)
            max_distance = 1 if len_lower <= 4 else (2 if len_lower <= 7 else 3)
            best_row: sqlite3.Row | None = None
            best_distance = max_distance + 1
            for row in self.connection.execute("SELECT * FROM stardew_entity").fetchall():
                display = row["display_name"].lower()
                if abs(len(display) - len_lower) > max_distance:
                    continue
                distance = levenshtein(last.lower(), display)
                if distance < best_distance:
                    best_distance = distance
                    best_row = row
                    if distance == 0:
                        break
            if best_row is not None and best_distance <= max_distance:
                return self._row_to_entity(best_row)

            # Last resort: scan attributes JSON. Rank by token overlap count.
            best_row = None
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

    # Spoken-answer composition. Every field not listed here is dropped:
    # raw wiki fields like "Fishing XP", "Size (inches)", "Recipe", or
    # "Healing" produced word-salad answers when an infobox was read aloud.
    _COMPACT_FIELDS: tuple[str, ...] = (
        "Location", "Season", "Time", "Weather", "Growth Time", "Regrowth",
        "Sell Price", "Seed Price", "Lives In", "Address", "Birthday",
        "Marriage", "Best Gifts", "Loved Gifts",
    )
    # Natural spoken templates per field. Kind-specific overrides below
    # reword fields whose generic phrasing would sound wrong (fish are
    # "caught", crops "grow").
    _FIELD_TEMPLATES: dict[str, str] = {
        "Location": "is found in {value}",
        "Season": "grows in {value}",
        "Time": "is active {value}",
        "Weather": "in {value} weather",
        "Growth Time": "takes {value} to grow",
        "Regrowth": "regrows in {value}",
        "Sell Price": "sells for {value}",
        "Seed Price": "seeds cost {value}",
        "Lives In": "lives in {value}",
        "Address": "can be found at {value}",
        "Birthday": "has a birthday on {value}",
        "Marriage": "is marriageable",
        "Best Gifts": "loves {value}",
        "Loved Gifts": "loves {value}",
    }
    _FISH_FIELD_TEMPLATES: dict[str, str] = {
        "Location": "can be caught at {value}",
        "Season": "can be caught in {value}",
    }

    @staticmethod
    def _spoken_value(key: str, value: str) -> str:
        """Normalise wiki shorthand and separators so values stay intelligible
        when spoken (and never carry control characters into TTS)."""
        value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f\ufffd]", " ", value)
        # Typographic separators: bullets become "and", time ranges read
        # "6am to 8pm" instead of an unpronounceable dash.
        value = value.replace("\u2022", "and").replace("\u2013", " to ").replace("\u2014", " to ")
        value = " ".join(value.split())
        lowered = value.lower()
        if key == "Season" and lowered in ("all", "any"):
            return "all seasons"
        if key == "Time" and lowered == "any":
            return "at any time"
        if key == "Weather" and lowered == "any":
            return "any weather"
        return value

    @classmethod
    def compose_answer(cls, entity: StardewEntity, query: str) -> str:
        """Build a deterministic, spoken-style answer from the infobox.

        Previously this read up to six raw infobox fields aloud in
        ``Key: value; Key: value`` form, producing word salad like
        ``Fishing XP: 18 21 24; Size (inches): 12-31``. Now: pick intent-
        relevant fields from a compact whitelist, render them through
        natural-language templates, and join into one flowing sentence that
        stays inside the TTS comfort zone.
        """
        q_lower = query.lower()
        # Fields relevant to the question's intent come first.
        priority: list[str] = []
        for intent, keys in INTENT_FIELDS.items():
            if intent in q_lower:
                priority.extend(key for key in keys if key not in priority)
                break
        priority.extend(
            key for key in cls._COMPACT_FIELDS if key not in priority
        )

        templates = dict(cls._FIELD_TEMPLATES)
        if entity.kind == "fish":
            templates.update(cls._FISH_FIELD_TEMPLATES)

        phrases: list[str] = []
        total = len(entity.display_name) + 12
        for key in priority:
            raw = entity.get(key)
            if not raw:
                continue
            if key == "Marriage" and raw.strip().lower() in ("no", "false"):
                continue
            value = cls._spoken_value(key, raw)
            # Gift lists read terribly in full — keep the head of the list.
            if key in ("Best Gifts", "Loved Gifts") and len(value) > 60:
                value = value[:60].rsplit(" ", 1)[0].rstrip(",") + ", and more"
            phrase = templates.get(key, key.lower()).format(value=value)
            if len(phrase) > 110:
                continue
            phrases.append(phrase)
            total += len(phrase) + 3
            if total >= 240 or len(phrases) >= 3:
                break
        if not phrases:
            return (
                f"{entity.display_name} is in the wiki, but I do not have a "
                "structured field for that question."
            )
        head = f"{entity.display_name} {phrases[0]}"
        if len(phrases) == 1:
            return head + "."
        if len(phrases) == 2:
            return f"{head} and {phrases[1]}."
        return f"{head}, " + ", ".join(phrases[1:-1]) + f", and {phrases[-1]}."

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
