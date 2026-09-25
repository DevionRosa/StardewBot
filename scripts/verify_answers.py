"""End-to-end verification of the accuracy fixes against the real corpus.

Run after ``python scripts/build_wiki_corpus.py`` finishes:

    python scripts/verify_answers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import wiki_index
from src.chat import answer_question
from src.transcribe import fuzzy_correct_transcript


def main() -> None:
    index = wiki_index.get_wiki_index()
    stats = index.corpus_stats()
    print("[Stats]", stats)

    kinds = index.connection.execute(
        "SELECT kind, COUNT(*) c FROM stardew_entity GROUP BY kind ORDER BY c DESC"
    ).fetchall()
    print("[Kinds]", ", ".join(f"{row['kind']}={row['c']}" for row in kinds))

    # Key entities must exist with structured attributes.
    failures: list[str] = []
    for name, kind in (("Bullhead", "fish"), ("Flounder", "fish"), ("Pumpkin", "crop")):
        entity = index.lookup_entity(f"tell me about {name.lower()}")
        if entity is None or entity.kind != kind or not entity.attributes:
            failures.append(f"{name}: missing entity/attributes (got {entity and entity.kind})")
        else:
            sample = dict(list(entity.attributes.items())[:4])
            print(f"[Entity] {name} ({entity.kind}): {sample}")
    if failures:
        for line in failures:
            print("[FAIL]", line)
        sys.exit(1)

    # STT guardrails: question words must survive correction.
    for sentence in (
        "where can i find the bullhead fish",
        "what can i catch in the ocean",
        "who is abigail",
    ):
        corrected = fuzzy_correct_transcript(sentence)
        status = "OK" if corrected == sentence else f"MANGLED -> {corrected!r}"
        print(f"[STT] {sentence!r}: {status}")

    # The user's real questions. Tier 1 (deterministic) must fire for fish.
    questions = [
        "where can i find the bullhead fish",
        "where can i catch a flounder in stardew valley",
        "the wiki says the bullhead fish can be found in the mountain lake is that true",
        "when is pumpkin grown",
        "tell me about abigail",
    ]
    llm_guard = wiki_index  # noqa: F841 — imported for clarity in output below
    for question in questions:
        print(f"\nQ: {question}")
        print("A:", answer_question(question))


if __name__ == "__main__":
    main()
