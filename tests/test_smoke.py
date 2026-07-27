"""Smoke tests for the deterministic-first Stardew pipeline.

Run with: ``python -m unittest tests.test_smoke`` from the project root.
Tests are network-free and use a fresh temp SQLite file per test class.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _json_attrs(attributes: dict) -> str:
    return json.dumps(attributes, ensure_ascii=False)


def _patch_index_db():
    """Patch ``src.wiki_index.CACHE_DB_PATH`` to point at a new temp DB."""
    from src import wiki_index

    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    wiki_index.CACHE_DB_PATH = Path(tmp.name)
    wiki_index._INDEX = None
    return tmp.name


class WikiIndexTestBase(unittest.TestCase):
    """Common setUp/tearDown that swaps the global wiki_index for a tmp DB."""

    def setUp(self) -> None:
        self.tmp_path = _patch_index_db()
        from src import wiki_index
        self.index = wiki_index.get_wiki_index()

    def tearDown(self) -> None:
        from src import wiki_index
        if getattr(self, "index", None) is not None:
            self.index.close()
        wiki_index._INDEX = None
        Path(self.tmp_path).unlink(missing_ok=True)


class TestSchemaAndInfobox(WikiIndexTestBase):
    def test_extended_schema_creates_tables(self) -> None:
        rows = self.index.connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        ).fetchall()
        names = {row[0] for row in rows}
        for table in (
            "wiki_pages", "stardew_page_clean", "stardew_entity",
            "stardew_vocabulary", "stardew_corpus_meta",
        ):
            self.assertIn(table, names)

    def test_infobox_extraction(self) -> None:
        html = """
        <html><body>
        <table class='infobox'>
          <tr><th>Growth Time</th><td>13 days</td></tr>
          <tr><th>Season</th><td>Fall</td></tr>
          <tr><th>Sell Price</th><td>320g</td></tr>
        </table>
        </body></html>
        """
        fields = self.index._extract_infobox(html)
        self.assertEqual(fields["Growth Time"], "13 days")
        self.assertEqual(fields["Season"], "Fall")
        self.assertEqual(fields["Sell Price"], "320g")

    def test_detect_kind_single_cue(self) -> None:
        # NPC infobox with only "Birthday" still classifies as npc.
        kind = self.index._detect_kind({"Birthday": "Spring 14"}, set())
        self.assertEqual(kind, "npc")
        # Fish infobox with two cues still classifies as fish.
        kind = self.index._detect_kind(
            {"Time": "Any", "Location": "Lake"}, {"fish"}
        )
        self.assertEqual(kind, "fish")

    def test_lookup_body_truncates_at_sentence(self) -> None:
        long_body = "Pike is a river fish. " + ("caught in summer. " * 200) + "The End."
        self.index.connection.execute(
            "INSERT INTO stardew_page_clean (title, page_id, url, infobox_json, body_text, categories, fetched_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("Pike", 1, "x", "{}", long_body, "[]", 0),
        )
        self.index.connection.commit()
        snippet = self.index.lookup_body("Pike", max_chars=200)
        self.assertLessEqual(len(snippet), 240)
        self.assertTrue(snippet.endswith("."), "snippet should end on a sentence boundary")


class TestLevenshteinCorrection(unittest.TestCase):
    def setUp(self) -> None:
        from src import transcribe
        transcribe._vocabulary = {
            "pike": "Pike", "shane": "Shane", "haley": "Haley",
            "penny": "Penny", "krobus": "Krobus",
        }
        transcribe._vocab_loaded = True

    def test_levenshtein_basics(self) -> None:
        from src.transcribe import _levenshtein
        self.assertEqual(_levenshtein("pike", "pik"), 1)
        self.assertEqual(_levenshtein("shame", "shane"), 1)
        self.assertEqual(_levenshtein("krobus", "krobus"), 0)

    def test_fuzzy_correction_substitutes_close_match(self) -> None:
        from src.transcribe import fuzzy_correct_transcript
        # Title-case input keeps the canonical form's capitalisation via .istitle().
        self.assertEqual(fuzzy_correct_transcript("Where can I catch Pike"), "Where can I catch Pike")
        self.assertEqual(fuzzy_correct_transcript("I like Shame"), "I like Shane")

    def test_first_char_mismatch_rejected(self) -> None:
        from src.transcribe import fuzzy_correct_transcript
        # 'Benny' looks 1-edit from 'Penny' but the first char differs.
        self.assertEqual(fuzzy_correct_transcript("where is Benny"), "where is Benny")
        # 'Shean' differs at the second char but 'Shane' is correct at distance 1 once first char matches.
        self.assertEqual(fuzzy_correct_transcript("Who is Shean"), "Who is Shane")

    def test_protected_tokens_unchanged(self) -> None:
        from src.transcribe import fuzzy_correct_transcript
        # "can" is short, "farming" is too generic (and not in vocab).
        self.assertEqual(fuzzy_correct_transcript("can I farm"), "can I farm")


class TestCorpusEmptyGuard(WikiIndexTestBase):
    def test_answer_question_refuses_when_corpus_empty(self) -> None:
        from src import wiki_index
        # tmp DB has zero entries. Override get_wiki_context to confirm we never call it.
        original_ctx = wiki_index.get_wiki_context
        wiki_index.get_wiki_context = lambda *args, **kwargs: ""
        try:
            from src import chat
            # Patch post_chat so a buggy free-form call would be detected.
            chat_calls = {"n": 0}
            original_post = chat._post_chat
            def tracker(messages, num_predict, temperature=0.1):
                chat_calls["n"] += 1
                return original_post(messages, num_predict, temperature)
            chat._post_chat = tracker
            try:
                # A question with no Stardew context and no corpus → refusal.
                answer = chat.answer_question("what is the weather in tokyo")
                self.assertIn("Stardew Valley", answer)
                # A Stardew topic but no corpus → refuse and ask to run build.
                answer = chat.answer_question("tell me about pumpkin")
                self.assertIn("build_wiki_corpus.py", answer)
                self.assertEqual(chat_calls["n"], 0, "free-form LLM must not be called when corpus is empty")
            finally:
                chat._post_chat = original_post
        finally:
            wiki_index.get_wiki_context = original_ctx


class TestBodyTextFallback(WikiIndexTestBase):
    def test_entity_found_routes_to_strict_rephrase_with_body(self) -> None:
        # Insert an empty-attributes entity so compose_answer would emit
        # the "no structured field" dead string — we want it routed into
        # tier 2 with body_text instead.
        body = "Pumpkin is a Fall crop that grows in 13 days and sells for 320g."
        self.index.connection.execute(
            "INSERT INTO stardew_page_clean (title, page_id, url, infobox_json, body_text, categories, fetched_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("Pumpkin", 1, "x", "{}", body, "[]", 0),
        )
        self.index.connection.execute(
            "INSERT INTO stardew_entity (canonical_name, display_name, kind, aliases, source_page, attributes, fetched_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("pumpkin", "Pumpkin", "crop", "[]", "Pumpkin", "{}", 0),
        )
        self.index.connection.execute(
            "INSERT INTO stardew_vocabulary (word, display, kind) VALUES ('pumpkin','Pumpkin','crop')"
        )
        self.index.connection.commit()

        from src import chat, wiki_index
        original_resolve = chat._resolved_model_name
        chat._resolved_model_name = "test-model"
        # Track prompt sent to LLM to verify body text was injected.
        captured: dict = {}
        original_post = chat._post_chat
        def fake(messages, num_predict, temperature=0.1):
            captured["messages"] = messages
            captured["num_predict"] = num_predict
            return "Pumpkin is a Fall crop that grows in 13 days."
        chat._post_chat = fake
        try:
            answer = chat.answer_question("tell me about pumpkin")
        finally:
            chat._post_chat = original_post
            chat._resolved_model_name = None
            wiki_index._INDEX = None

        self.assertEqual(answer, "Pumpkin is a Fall crop that grows in 13 days.")
        user_prompt = captured["messages"][-1]["content"]
        self.assertIn("Page Body", user_prompt)
        self.assertIn("13 days", user_prompt)
        self.assertEqual(captured["num_predict"], 120, "num_predict should be conservative")


class TestDeterministicLookupWins(WikiIndexTestBase):
    def test_tier1_handles_question_without_llm(self) -> None:
        # Pumpkin has real attributes → tier 1 compose_answer should fire.
        self.index.connection.execute(
            "INSERT INTO stardew_entity (canonical_name, display_name, kind, aliases, source_page, attributes, fetched_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "pumpkin", "Pumpkin", "crop", "[]", "Pumpkin",
                _json_attrs({"Growth Time": "13 days", "Season": "Fall", "Sell Price": "320g"}),
                0,
            ),
        )
        self.index.connection.execute(
            "INSERT INTO stardew_vocabulary (word, display, kind) VALUES ('pumpkin','Pumpkin','crop')"
        )
        self.index.connection.commit()
        from src import chat, wiki_index
        chat._resolved_model_name = "test-model"
        chat_calls = {"n": 0}
        original_post = chat._post_chat
        def tracker(messages, num_predict, temperature=0.1):
            chat_calls["n"] += 1
            return original_post(messages, num_predict, temperature)
        chat._post_chat = tracker
        try:
            answer = chat.answer_question("when is pumpkin grown in stardew valley")
        finally:
            chat._post_chat = original_post
            chat._resolved_model_name = None
            wiki_index._INDEX = None
        self.assertIn("Pumpkin", answer)
        self.assertEqual(chat_calls["n"], 0, "tier 1 deterministic answer must not call LLM")


class TestModelResolutionLock(unittest.TestCase):
    def test_resolve_name_thread_safe(self) -> None:
        from src import chat
        # Pretend nothing is cached.
        original = chat._resolved_model_name
        chat._resolved_model_name = None
        chat._model_cache = {"models": ["llama3.2"], "timestamp": chat.time.time() if hasattr(chat, "time") else 0}
        try:
            results = []
            errors: list[Exception] = []
            def worker():
                try:
                    results.append(chat._resolve_model_name())
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertFalse(errors, f"threading raised: {errors}")
            self.assertGreaterEqual(len(results), 1)
            self.assertEqual(len(set(results)), 1, "model name race condition produced inconsistent results")
        finally:
            chat._resolved_model_name = original


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    sys.exit(0 if result.wasSuccessful() else 1)
