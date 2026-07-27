"""Stardew Valley answer pipeline.

Three tiers, ordered by accuracy:

    1. Deterministic structured lookup — a populated ``stardew_entity`` row is
       composed into a spoken answer with zero LLM involvement.
    2. Strict LLM rephraser — wiki ``body_text`` and curated facts are passed
       to Ollama with a system prompt that forbids adding facts outside the
       provided context. The LLM can only *rephrase*; it cannot invent.
    3. Free-form LLM — last-resort only when (1) and (2) cannot answer. Will
       accept some hallucination risk in exchange for an actual response.

The corpus-empty guard means the bot refuses free-form until
``python scripts/build_wiki_corpus.py`` has been run. Running the builder once
is the price of admission for high factual accuracy.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional

import requests

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS
from .stardew_knowledge import get_system_prompt, retrieve_stardew_facts
from .wiki_index import (
    StardewEntity,
    StardewWikiIndex,
    answer_from_wiki,
    get_wiki_context,
    get_wiki_index,
)

_model_cache = {"models": [], "timestamp": 0}
_model_cache_ttl = 60

_resolved_model_name: Optional[str] = None
_model_lock = threading.Lock()

# Reserved fallback lexicon. Used only before the corpus is populated — once
# ``scripts/build_wiki_corpus.py`` runs, vocabulary comes from the DB.
_RESERVED_KEYWORDS = (
    "stardew", "pelican town", "farm", "farming", "farmer", "fishing", "fish",
    "mine", "mines", "crop", "crops", "season", "seed", "seeds", "growth",
    "harvest", "plant", "planting", "fertilizer", "sprinkler", "tool", "tools",
    "ore", "coal", "copper", "iron", "gold", "iridium",
    "recipe", "recipes", "gift", "gifts", "heart", "hearts", "friendship",
    "marriage", "festival", "bundle", "bundles", "community center",
    "sebastian", "alex", "elliott", "harvey", "sam", "shane",
    "abigail", "emily", "haley", "leah", "maru", "penny",
    "willy", "clint", "pierre", "caroline", "jodi", "kent", "lewis", "marnie",
    "robin", "dwarf", "krobus", "linus",
    "parsnip", "turnip", "potato", "cauliflower", "green bean", "strawberry",
    "blueberry", "cranberry", "melon", "pumpkin", "ancient fruit", "starfruit",
    "corn", "hops", "pepper", "sunflower", "tomato", "yam", "eggplant",
    "rhubarb", "pike",
)  # noqa: E501 — keyword tuple intentionally long for fallback coverage

# When the user phrases a question with one of these intent verbs and the
# entity matches the kind, tier 1 is considered a hit. Closing this gap
# prevents the "I asked 'when does X grow' and got tier 2" failure mode.
INTENT_BY_KIND: dict[str, tuple[str, ...]] = {
    "crop": ("when", "season", "grow", "plant", "harvest", "replant", "how long", "price"),
    "fish": ("when", "where", "catch", "find", "season", "time"),
    "npc": ("who", "love", "like", "birthday", "live", "where", "about"),
    "location": ("where", "what", "inside", "about"),
    "monster": ("where", "fight", "drop", "about"),
    "bundle": ("what", "need", "reward", "about"),
}


# ---------------------------------------------------------------- Ollama glue

def _list_local_models() -> List[str]:
    global _model_cache
    now = time.time()
    if _model_cache["models"] and (now - _model_cache["timestamp"]) < _model_cache_ttl:
        return _model_cache["models"]
    try:
        response = requests.get(f"{OLLAMA_BASE_URL.replace('/v1', '')}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        models = [model["name"] for model in data.get("models", []) if model.get("name")]
        _model_cache = {"models": models, "timestamp": now}
        return models
    except Exception as exc:
        print(f"[Chat] Failed to list models: {exc}. Using cached or default.")
        return _model_cache["models"]


def _resolve_model_name() -> str:
    """Thread-safe lookup of the locally installed Ollama model."""
    global _resolved_model_name
    if _resolved_model_name:
        return _resolved_model_name
    with _model_lock:
        if _resolved_model_name:
            return _resolved_model_name
        available = _list_local_models()
        if OLLAMA_MODEL in available:
            _resolved_model_name = OLLAMA_MODEL
        elif available:
            _resolved_model_name = available[0]
        else:
            _resolved_model_name = OLLAMA_MODEL
        return _resolved_model_name


def _reset_model_resolution() -> None:
    """Used by tests so model-name caching does not bleed across runs."""
    global _resolved_model_name
    with _model_lock:
        _resolved_model_name = None


def _post_chat(messages: List[dict], num_predict: int, temperature: float = 0.1) -> str:
    """Send a chat request to the native Ollama endpoint and return content."""
    native_url = f"{OLLAMA_BASE_URL.replace('/v1', '').rstrip('/')}/api/chat"
    payload = {
        "model": _resolve_model_name(),
        "messages": messages,
        "options": {"temperature": temperature, "num_predict": num_predict},
        "stream": False,
    }
    response = requests.post(native_url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    if "message" in data:
        return (data["message"].get("content") or "").strip()
    if "response" in data:
        return (data["response"] or "").strip()
    return ""


# ---------------------------------------------------------------- detection

def _is_stardew_mention(question: str, vocab: set[str]) -> bool:
    """Heuristic Stardew detection. Cheap, no HTTP."""
    lowered = question.lower()
    tokens = {token.strip(".,?!'\"") for token in lowered.split()}
    if any(token in vocab for token in tokens):
        return True
    return any(keyword in lowered for keyword in _RESERVED_KEYWORDS)


def is_stardew_question(question: str) -> bool:
    return bool(get_wiki_context(question, limit=1))


# ----------------------------------------------------------- answer tiers ---

def fallback_response() -> str:
    return (
        "I can only help with Stardew Valley questions. Ask me about NPCs, "
        "crops, mining, fishing, festivals, or anything else from the game."
    )


def _corpus_empty_message() -> str:
    return (
        "I don't have a local Stardew knowledge base yet. "
        "Please run `python scripts/build_wiki_corpus.py` once to download the "
        "wiki, then ask your question again."
    )


def _looks_like_wiki_dump(answer: str) -> bool:
    """Conservative marker-based test for wiki regurgitation."""
    if not answer:
        return True
    if len(answer) > 250 and ":" in answer and "?" not in answer:
        return True
    if answer.count("[") > 2 or answer.count("\n") > 4:
        return True
    return False


def _has_relevant_entity_data(entity: StardewEntity, question: str) -> bool:
    """True if the entity's attributes can plausibly answer the question."""
    if not entity.attributes:
        return False
    q_lower = question.lower()
    # Direct key-word match (e.g., "Season" appears in the question).
    for key in entity.attributes:
        if key.lower() in q_lower:
            return True
    intent_words = {"about", "info", "information", "details", "describe"}
    if any(word in q_lower for word in intent_words):
        return True
    # Kind-aware intent verbs: "when is pumpkin grown" → crop entity → tier 1.
    for intent in INTENT_BY_KIND.get(entity.kind, ()):
        if intent in q_lower:
            return True
    return False


def _safe_sentence(text: str, max_chars: int = 240) -> str:
    """Trim text to a clean sentence ending at or before ``max_chars``."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rstrip()
    last_period = trimmed.rfind(".")
    if last_period >= max_chars // 2:
        return trimmed[: last_period + 1]
    return trimmed.rstrip(" ,;:") + "..."


def _available_body_text(entity: StardewEntity, index) -> str:
    """Best-effort body_text pull for the entity."""
    try:
        return index.lookup_body(entity.source_page)
    except Exception as exc:
        print(f"[Chat] body_text fetch failed for {entity.source_page}: {exc}")
        return ""


# --------------------------------------------------------------- main API ---

def answer_question(question: str) -> str:
    """Return the best available answer for ``question``."""
    if not question.strip():
        return fallback_response()

    index = get_wiki_index()

    corpus_vocab: set[str] = set()
    try:
        corpus_vocab = index.get_vocabulary()
    except Exception as exc:
        print(f"[Chat] Corpus vocab unavailable: {exc}")

    # Local-only emptiness check — never triggers HTTP.
    stats: dict = {}
    try:
        stats = index.corpus_stats()
    except Exception:
        stats = {}
    corpus_empty = not stats or stats.get("entity", 0) == 0 and stats.get("legacy_pages", 0) == 0

    # Single guard: refuse off-topic + corpus-empty, off-topic + corpus-empty
    # is the only path where the upstream user has nothing useful to hear.
    if corpus_empty and not _is_stardew_mention(question, set()):
        return fallback_response()

    if corpus_empty:
        return _corpus_empty_message()

    # ------------------ Tier 1: deterministic structured lookup -----------------
    entity: Optional[StardewEntity] = None
    try:
        entity = index.lookup_entity(question)
    except Exception as exc:
        print(f"[Chat] Entity lookup failed: {exc}")

    if entity is not None:
        if entity.attributes and _has_relevant_entity_data(entity, question):
            return StardewWikiIndex.compose_answer(entity, question)
        # Entity exists but infobox is sparse — fall through to strict rephrase
        # so the LLM can summarise the body rather than invent details.
        body_text = _available_body_text(entity, index)
        return _strict_rephrase(
            question,
            body_text=body_text,
            knowledge_context=retrieve_stardew_facts(question, max_results=2),
        )

    # ------------------ Tier 2: strict LLM rephraser ----------------------------
    wiki_context = ""
    knowledge_context = ""
    try:
        wiki_context = get_wiki_context(question, limit=2)
        knowledge_context = retrieve_stardew_facts(question, max_results=2)
    except Exception as exc:
        print(f"[Chat] Retrieval failed: {exc}")

    if wiki_context or knowledge_context:
        return _strict_rephrase(
            question,
            wiki_context=wiki_context,
            knowledge_context=knowledge_context,
        )

    # ------------------ Tier 3: free-form fallback ------------------------------
    return _freeform(question)


def _strict_rephrase(
    question: str,
    wiki_context: str = "",
    body_text: str = "",
    knowledge_context: str = "",
) -> str:
    """LLM may only rephrase. Refuses unless context clearly answers."""
    context_blocks: list[str] = []
    if body_text:
        context_blocks.append(f"Page Body:\n{body_text}")
    if wiki_context:
        context_blocks.append(f"Wiki Excerpts:\n{wiki_context}")
    if knowledge_context:
        context_blocks.append(f"Verified Facts:\n{knowledge_context}")

    if not context_blocks:
        return "I don't have that information in the wiki."

    system = (
        "You are a precise Stardew Valley encyclopedia assistant.\n"
        "You may ONLY use facts that appear in the Context sections below. "
        "If the answer cannot be derived from the context, reply exactly: "
        "\"I don't have that information in the wiki.\" Do NOT add facts from "
        "your training data. Do NOT quote the wiki verbatim — rephrase into "
        "1-2 short conversational sentences."
    )
    user_payload = (
        "Context:\n" + "\n\n".join(context_blocks)
        + f"\n\nQuestion: {question}\n\nAnswer (1-2 sentences):"
    )

    try:
        answer = _post_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user_payload}],
            num_predict=120,
            temperature=0.1,
        )
    except Exception as exc:
        print(f"[Chat] Strict rephraser failed: {exc}")
        answer = ""

    if answer and not _looks_like_wiki_dump(answer):
        return answer

    # Safety net: never let a wiki dump reach the TTS pipeline.
    if wiki_context:
        snippet = answer_from_wiki(question, limit=1)
        if snippet and len(snippet) <= 300:
            return snippet
    if body_text:
        snippet = _safe_sentence(body_text, max_chars=240)
        if snippet:
            return snippet
    return "I don't have that information in the wiki."


def _freeform(question: str) -> str:
    """Last-resort path. Reached only after every grounded approach failed."""
    system = get_system_prompt() + (
        "\nNo wiki context was found for this question. Be honest about "
        "uncertainty, prefer concise answers (1-2 sentences), and never "
        "invent specific numerical game values."
    )
    try:
        return _post_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": question}],
            num_predict=120,
            temperature=0.2,
        )
    except Exception as exc:
        print(f"[Chat] Free-form fallback failed: {exc}")
        return "I couldn't reach the local LLM. Make sure Ollama is running on http://localhost:11434."
