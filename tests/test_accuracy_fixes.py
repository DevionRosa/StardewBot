"""Regression tests for the accuracy fixes.

Each test class maps to a real failure observed in the voice pipeline:

* STT corrector rewrote ``where``/``what`` into ``wheat`` on every question.
* Curated "fishing" facts were Pike-only and hijacked every fish question.
* The infobox parser returned {} for every wiki page (wrong table selector),
  so Bullhead/Flounder had no structured data and tier 1 never fired.
* ``startswith("fish")`` category matching classified fishing rods and tanks
  as ``fish`` entities.
* Tier 2 had no structured-RAG fallback over the 2,000-page corpus.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_smoke import WikiIndexTestBase, _json_attrs


BULLHEAD_INFOBOX = {
    "Location": "Mountain Lake",
    "Time": "Any",
    "Season": "All",
    "Weather": "Any",
    "Difficulty": "46",
    "Behavior": "Smooth",
}

BULLHEAD_HTML = """
<html><body>
<table id="infoboxtable">
  <tr><td colspan="2" id="infoboxheader">Bullhead</td></tr>
  <tr><td colspan="2">A relative of the catfish.</td></tr>
  <tr><td id="infoboxsection">Location</td><td id="infoboxdetail">Mountain Lake</td></tr>
  <tr><td id="infoboxsection">Time</td><td id="infoboxdetail">Any</td></tr>
  <tr><td id="infoboxsection">Season</td><td id="infoboxdetail">All</td></tr>
  <tr><td id="infoboxsection">Weather</td><td id="infoboxdetail">Any</td></tr>
  <tr><td id="infoboxsection">Difficulty</td><td id="infoboxdetail">46</td></tr>
  <tr><td id="infoboxsection">Behavior</td><td id="infoboxdetail">Smooth</td></tr>
</table>
</body></html>
"""


def _seed_bullhead(index) -> None:
    index.connection.execute(
        "INSERT INTO stardew_page_clean (title, page_id, url, infobox_json, body_text, categories, fetched_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "Bullhead", 1, "x", json.dumps(BULLHEAD_INFOBOX),
            "The Bullhead is a fish found in Mountain Lake. It can be caught in any season, "
            "any weather, and at any time of day.",
            json.dumps(["Lake fish", "Spring fish"]), 0,
        ),
    )
    index.connection.execute(
        "INSERT INTO stardew_entity (canonical_name, display_name, kind, aliases, source_page, attributes, fetched_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("bullhead", "Bullhead", "fish", "[]", "Bullhead", _json_attrs(BULLHEAD_INFOBOX), 0),
    )
    index.connection.execute(
        "INSERT INTO stardew_vocabulary (word, display, kind) VALUES ('bullhead', 'Bullhead', 'fish')"
    )
    index.connection.commit()


class TestSTTCorrectorGuardrails(unittest.TestCase):
    """'where can i find' must never become 'wheat can i find' again."""

    def setUp(self) -> None:
        from src import transcribe
        self._orig_vocab = transcribe._vocabulary
        self._orig_loaded = transcribe._vocab_loaded
        # Vocabulary includes real game words that are 1-2 edits from common
        # English words — exactly what broke the pipeline before.
        transcribe._vocabulary = {
            "wheat": "Wheat", "whet": "Whet", "pike": "Pike",
            "bullhead": "Bullhead", "shane": "Shane", "seeds": "Seeds", "pam": "Pam",
        }
        transcribe._vocab_loaded = True

    def tearDown(self) -> None:
        from src import transcribe
        transcribe._vocabulary = self._orig_vocab
        transcribe._vocab_loaded = self._orig_loaded

    def test_question_words_not_corrected(self) -> None:
        from src.transcribe import fuzzy_correct_transcript
        for sentence in (
            "where can i find the bullhead fish",
            "what can i plant in fall",
            "who is shane",
            "how much does pike sell for",
        ):
            self.assertEqual(
                fuzzy_correct_transcript(sentence), sentence,
                f"common question word was rewritten: {sentence!r}",
            )

    def test_close_entity_words_still_corrected(self) -> None:
        from src.transcribe import fuzzy_correct_transcript
        self.assertEqual(fuzzy_correct_transcript("where is Shame"), "where is Shane")
        self.assertEqual(fuzzy_correct_transcript("catching bullhed fish"), "catching bullhead fish")


class TestCuratedFactRetrieval(unittest.TestCase):
    """Pike-only fishing facts must not hijack other fish questions."""

    def test_fish_question_gets_no_pike_fact(self) -> None:
        from src.stardew_knowledge import retrieve_stardew_facts
        context = retrieve_stardew_facts("where can i catch a bullhead fish")
        self.assertNotIn("Pike", context, "Pike facts leaked into a Bullhead question")

    def test_exact_fish_question_gets_its_own_fact(self) -> None:
        from src.stardew_knowledge import retrieve_stardew_facts
        context = retrieve_stardew_facts("where can i catch a pike fish")
        self.assertIn("Pike", context)

    def test_unrelated_topic_returns_empty(self) -> None:
        from src.stardew_knowledge import retrieve_stardew_facts
        self.assertEqual(retrieve_stardew_facts("who is Krobus"), "")


class TestInfoboxParser(WikiIndexTestBase):
    """The wiki uses <table id="infoboxtable"> with td-based keys."""

    def test_stardew_wiki_infobox_layout_parses(self) -> None:
        fields = self.index._extract_infobox(BULLHEAD_HTML)
        self.assertEqual(fields["Location"], "Mountain Lake")
        self.assertEqual(fields["Season"], "All")
        self.assertEqual(fields["Difficulty"], "46")

    def test_generic_infobox_layout_still_parses(self) -> None:
        html = (
            "<html><body><table class='infobox'>"
            "<tr><th>Growth Time</th><td>13 days</td></tr>"
            "</table></body></html>"
        )
        fields = self.index._extract_infobox(html)
        self.assertEqual(fields["Growth Time"], "13 days")

    def test_data_sort_value_artifacts_stripped(self) -> None:
        from scripts.build_wiki_corpus import normalise_value
        self.assertEqual(normalise_value('data-sort-value="15"> 15g'), "15g")

    def test_builder_detects_fish_from_infobox(self) -> None:
        from scripts.build_wiki_corpus import detect_kind
        self.assertEqual(detect_kind({"Difficulty": "46", "Behavior": "Smooth"}, []), "fish")
        self.assertEqual(detect_kind({"Growth Time": "13 days"}, []), "crop")

    def test_category_matching_keeps_equipment_out_of_fish(self) -> None:
        from scripts.build_wiki_corpus import detect_kind
        self.assertEqual(detect_kind({}, ["Fishing Poles"]), "article")
        self.assertEqual(detect_kind({}, ["Fish Tanks"]), "article")
        self.assertEqual(detect_kind({}, ["Fishing Tackle"]), "article")
        self.assertEqual(detect_kind({}, ["Pond fish"]), "fish")
        self.assertEqual(detect_kind({}, ["Lake fish"]), "fish")

    def test_index_detect_kind_matches_builder(self) -> None:
        self.assertEqual(self.index._detect_kind({}, {"fishing poles"}), "article")
        self.assertEqual(self.index._detect_kind({}, {"pond fish"}), "fish")
        self.assertEqual(self.index._detect_kind({"Difficulty": "46"}, set()), "fish")


class TestDeterministicFishAnswers(WikiIndexTestBase):
    """The original failing question must now be answered without an LLM."""

    def test_bullhead_question_answered_from_infobox(self) -> None:
        _seed_bullhead(self.index)
        from src import chat, wiki_index
        original = chat._post_chat
        chat._post_chat = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("tier 1 must answer fish questions without the LLM")
        )
        try:
            answer = chat.answer_question("where can i find the bullhead fish")
        finally:
            chat._post_chat = original
            wiki_index._INDEX = None
        self.assertIn("Mountain Lake", answer)

    def test_composed_answer_is_spoken_style(self) -> None:
        _seed_bullhead(self.index)
        entity = self.index.lookup_entity("where can i catch a bullhead")
        self.assertIsNotNone(entity)
        answer = self.index.compose_answer(entity, "where can i catch a bullhead")
        self.assertIn("Bullhead", answer)
        self.assertIn("caught at Mountain Lake", answer)
        self.assertNotIn(":", answer, "raw Key: value formatting leaked into spoken answer")

    def test_lookup_entity_resists_leading_noise_token(self) -> None:
        _seed_bullhead(self.index)
        from src import wiki_index
        # Seed a decoy entity that matches the leading STT-noise token.
        self.index.connection.execute(
            "INSERT INTO stardew_entity (canonical_name, display_name, kind, aliases, source_page, attributes, fetched_at)"
            " VALUES ('wheat', 'Wheat', 'crop', '[]', 'Wheat', '{}', 0)"
        )
        self.index.connection.commit()
        entity = wiki_index.lookup_stardew_entity("wheat can i fish the bullhead fish")
        self.assertIsNotNone(entity)
        self.assertEqual(entity.display_name, "Bullhead", "leading noise token hijacked the lookup")


class TestStructuredRagFallback(WikiIndexTestBase):
    """Tier 2 must be able to ground answers in the 2,000-page corpus."""

    def test_search_clean_finds_corpus_pages(self) -> None:
        _seed_bullhead(self.index)
        # Seeding bypassed the builder, so self-heal the FTS mirror first.
        self.index._ensure_fts_populated()
        passages = self.index._search_clean("bullhead fish", limit=3)
        self.assertTrue(passages, "structured RAG returned nothing for an in-corpus question")
        self.assertEqual(passages[0].title, "Bullhead")

    def test_context_falls_back_to_corpus_without_legacy_pages(self) -> None:
        _seed_bullhead(self.index)
        self.index._ensure_fts_populated()
        from src import wiki_index
        context = wiki_index.get_wiki_context("bullhead fish", limit=2)
        self.assertIn("Bullhead", context)

    def test_fts_mirror_self_heals_when_empty(self) -> None:
        # Simulate a DB built before the FTS mirror existed.
        _seed_bullhead(self.index)
        self.index.connection.execute("DELETE FROM stardew_page_clean_fts")
        self.index.connection.execute("DELETE FROM stardew_corpus_meta WHERE key='fts_row_count'")
        self.index.connection.commit()
        self.index._ensure_fts_populated()
        self.assertEqual(self.index._search_clean("bullhead", limit=1)[0].title, "Bullhead")


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    sys.exit(0 if result.wasSuccessful() else 1)
