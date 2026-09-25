"""Speech-to-text with Stardew-aware post-correction.

Wraps Vosk's ``KaldiRecognizer`` and applies a token-level fuzzy correction
step against the local Stardew vocabulary. Without this step, the recognizer
frequently mis-hears proper nouns ("Shane" -> "shame", "Krobus" -> "crow bus"),
which then breaks every downstream lookup. Catching these mistakes early
raises end-to-end accuracy more than any model swap.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Dict

from vosk import KaldiRecognizer, Model, SetLogLevel

from .config import SAMPLE_RATE, VOSK_MODEL_DIR
from ._transcribe_distance import levenshtein as _levenshtein

SetLogLevel(-1)

# Module-level singleton guards. Acquisition order is
# ``_recognizer_lock`` -> ``_vocab_lock`` / ``_model_lock``; each
# inner lock is acquired and released before the outer one continues,
# so there is no nested holding today. The rule that keeps it that
# way: never call ``_get_recognizer`` while holding either inner
# lock — that would deadlock because ``threading.Lock`` is
# *non-reentrant* (the same thread that owns the lock cannot re-acquire it).
_model: Model | None = None
_model_lock = threading.Lock()
_recognizer = None
_recognizer_lock = threading.Lock()

# Lazy vocabulary loaded from the wiki corpus on first use.
_vocabulary: Dict[str, str] = {}
_vocab_loaded: bool = False
_vocab_lock = threading.Lock()

# Single-letter / stop tokens we never want to "correct" into something else.
#
# The second block is critical: game entities like "Wheat" live in the wiki
# vocabulary, and without these protections the fuzzy corrector happily
# rewrote question words into them — "where can I find" → "wheat can I find"
# (2 edits, same first letter) on EVERY question. Common words Vosk already
# transcribes reliably must never be fuzzy-matched away.
_PROTECTED_TOKENS = {
    "i", "a", "an", "the", "to", "of", "and", "or", "my", "in", "on", "at",
    "for", "be", "is", "it", "you", "we", "do", "so", "as", "if",
    "can", "yes", "no", "not", "but", "or", "if", "by", "an", "or",
    # Question words and common verbs/modals.
    "where", "what", "when", "who", "whose", "why", "how", "which", "whats",
    "there", "their", "they", "them", "then", "than", "that", "this",
    "these", "those", "with", "will", "shall", "would", "could", "should",
    "might", "must", "have", "been", "being", "about", "after", "again",
    "tell", "give", "show", "make", "need", "want", "know", "does", "did",
    "get", "got", "use", "sell", "buy", "best", "good", "much", "many",
    "more", "most", "some", "such", "only", "also", "very", "just", "like",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']*")


def _ensure_vocabulary_loaded() -> Dict[str, str]:
    """Read the Stardew vocabulary table from SQLite, once per session."""
    global _vocabulary, _vocab_loaded
    if _vocab_loaded:
        return _vocabulary
    with _vocab_lock:
        if _vocab_loaded:
            return _vocabulary
        try:
            from .wiki_index import get_wiki_index

            _vocabulary = get_wiki_index().get_vocabulary_with_display()
        except Exception as exc:  # pragma: no cover - DB unavailable during tests
            print(f"[STT] Vocabulary unavailable: {exc}")
            _vocabulary = {}
        _vocab_loaded = True
    return _vocabulary


def _preserve_case(token: str, replacement: str) -> str:
    """Match the case of the original token (UPPER, Title, lower)."""
    if token.isupper() and len(token) > 1:
        return replacement.upper()
    if token.istitle():
        return replacement[:1].upper() + replacement[1:]
    return replacement.lower()


def _correct_token(token: str, vocab: Dict[str, str]) -> str:
    """Return a canonically-cased Stardew word if the token is close to one.

    The threshold policy is conservative on purpose: requiring a matching
    first letter prevents mass substitutions like ``Benny`` -> ``Penny``,
    which would otherwise be accepted as a single-edit fix.
    """
    if not vocab or len(token) < 4 or token.lower() in _PROTECTED_TOKENS:
        return token

    lowered = token.lower()
    if lowered in vocab:
        return _preserve_case(token, vocab[lowered])

    # Tiered thresholds: 1 edit for <=4 chars, 2 for 5..7, 3 for >=8.
    if len(lowered) <= 4:
        max_distance = 1
    elif len(lowered) <= 7:
        max_distance = 2
    else:
        max_distance = 3

    best_word: str | None = None
    best_distance = max_distance + 1
    first_char = lowered[0]
    for vocab_word, display in vocab.items():
        if abs(len(vocab_word) - len(lowered)) > max_distance:
            continue
        # First character mismatch is a near-certain wrong substitution for
        # proper nouns (Benny → Penny, Shean → Shane, …). Always skip them.
        if vocab_word[0] != first_char:
            continue
        distance = _levenshtein(lowered, vocab_word)
        if distance < best_distance:
            best_distance = distance
            best_word = display
            if distance == 1:
                break
    if best_word is None:
        return token
    return _preserve_case(token, best_word)


def fuzzy_correct_transcript(transcript: str) -> str:
    """Apply Stardew vocabulary correction to every token in the transcript."""
    vocab = _ensure_vocabulary_loaded()
    if not vocab or not transcript.strip():
        return transcript
    return _TOKEN_RE.sub(lambda match: _correct_token(match.group(0), vocab), transcript)


def _get_model() -> Model:
    """Lazy-load the Vosk model once per session.

    Uses double-checked locking so concurrent callers (the main
    thread + a future second pipeline thread, e.g.) don't both call
    ``Model(...)`` or clobber ``_model`` mid-load. The outer
    ``if _model is None`` is the optimistic fast path; the inner
    ``if`` under ``_model_lock`` handles the rare race. Matches the
    pattern already used by ``_ensure_vocabulary_loaded`` above and
    ``_get_recognizer`` below.

    Print one line on the cold path so the THINKING-spinner phase of
    the user's first turn isn't silent — without it, a multi-second
    Vosk cold load reads as "the app is hung".
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not VOSK_MODEL_DIR.exists():
                    raise FileNotFoundError(
                        f"Missing Vosk model directory: {VOSK_MODEL_DIR}"
                    )
                print("[STT] Loading model (first turn, may take a few seconds)…")
                _model = Model(str(VOSK_MODEL_DIR))
    return _model


def _build_recognizer() -> KaldiRecognizer:
    """Build a recognizer, optionally using a strict Stardew grammar.

    Free-form is the default — a constrained grammar hurts coverage of follow-up
    questions. Set ``STARDEW_VOSK_GRAMMAR=strict`` to enable the restricted
    grammar for the small set of pure entity-name scenarios.
    """
    grammar: str | None = None
    if os.getenv("STARDEW_VOSK_GRAMMAR", "").lower() == "strict":
        vocab = _ensure_vocabulary_loaded()
        if vocab:
            grammar = json.dumps(sorted(set(vocab.values())))
    if grammar:
        return KaldiRecognizer(_get_model(), SAMPLE_RATE, grammar)
    return KaldiRecognizer(_get_model(), SAMPLE_RATE)


def _get_recognizer() -> KaldiRecognizer:
    """Thread-safe singleton accessor that resets state between calls."""
    global _recognizer
    with _recognizer_lock:
        if _recognizer is None:
            _recognizer = _build_recognizer()
        else:
            _recognizer.Reset()
        return _recognizer


def transcribe_audio(audio_bytes: bytes, correct: bool = True) -> str:
    """Transcribe ``audio_bytes`` and optionally correct proper nouns."""
    if not audio_bytes:
        return ""
    start_time = time.time()
    recognizer = _get_recognizer()
    recognizer.AcceptWaveform(audio_bytes)
    result = json.loads(recognizer.FinalResult())
    transcript = result.get("text", "").strip()
    elapsed = time.time() - start_time
    print(f"[STT] Transcription in {elapsed:.2f}s")
    if not correct or not transcript:
        return transcript
    corrected = fuzzy_correct_transcript(transcript)
    if corrected != transcript:
        print(f"[STT] Corrected: {transcript!r} -> {corrected!r}")
    return corrected
