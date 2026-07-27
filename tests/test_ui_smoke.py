"""Headless tests for the StardewBot widget.

These run with the ``offscreen`` ``QPA`` platform so the full Qt
event loop is exercised without a visible window. They cover the
state machine, signal wiring, and one round of painter calls per
state to make sure paintEvent doesn't blow up.
"""
from __future__ import annotations

import unittest

# Force the offscreen platform BEFORE PySide6 is imported anywhere
# downstream; the Qt event loop stubs out the window manager so
# frameless / translucent windows can be exercised on CI boxes.
import os
import sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ui import AnimatedSurface, AppState, STATE_LABEL, StardewWidgetWindow
from src.ui.states import is_valid
from src.ui.pipeline import PipelineThread
from src.ui.widget import (
    _BAR_COUNT,
    _DOT_RADIUS,
    _OUTER_RADIUS,
    _RING_COUNT,
)


def _ensure_qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


class AppStateTests(unittest.TestCase):
    def test_state_label_keys_are_canonical(self):
        _ensure_qapp()
        labels = set(STATE_LABEL.keys())
        self.assertEqual(labels, {s.value for s in AppState})

    def test_is_valid_accepts_only_canonical_state_names(self):
        for state in AppState:
            self.assertTrue(is_valid(state.value))
        self.assertFalse(is_valid("idle"))
        self.assertFalse(is_valid(""))
        self.assertFalse(is_valid("WAITING"))


class AnimatedSurfaceTests(unittest.TestCase):
    def test_set_state_ignores_unknown_names(self):
        _ensure_qapp()
        surface = AnimatedSurface()
        surface.set_state(AppState.WAITING.value)
        surface.set_state("frobnicated")
        self.assertEqual(surface.state(), AppState.WAITING.value)

    def test_paint_each_state_does_not_crash(self):
        _ensure_qapp()
        surface = AnimatedSurface()
        surface.resize(180, 180)
        for state in AppState:
            surface.set_state(state.value)
            surface.set_level(900.0)
            surface.tick(0.42 + 0.13 * list(AppState).index(state))
            # Force a repaint via the actual Qt event loop.
            loop = QEventLoop()
            QTimer.singleShot(0, loop.quit)
            loop.exec()
        # Run again on a smaller widget for low-res coverage.
        surface.resize(96, 96)
        for state in AppState:
            surface.set_state(state.value)
            surface.tick(0.0)
            QCoreApplication.processEvents()

    def test_constants_match_design(self):
        # Anchored so we notice accidental tweaks that affect scaling.
        self.assertEqual(_OUTER_RADIUS, 46)
        self.assertEqual(_DOT_RADIUS, 14)
        self.assertEqual(_RING_COUNT, 3)
        self.assertEqual(_BAR_COUNT, 5)


class SignalWiringTests(unittest.TestCase):
    def test_widget_relays_state_changed_to_surface(self):
        _ensure_qapp()
        widget = StardewWidgetWindow(size=160)
        try:
            widget._on_state(AppState.LISTENING.value)
            self.assertEqual(widget._surface.state(), AppState.LISTENING.value)
            widget._on_state("garbage")
            self.assertEqual(widget._surface.state(), AppState.LISTENING.value)
        finally:
            widget.close()

    def test_pipeline_thread_emits_signals_in_offline_mode(self):
        """Without a microphone, the wakeword loop should refuse to start
        cleanly when we set ``_is_running=False`` immediately.

        Wait is 15 s. ``run()`` constructs ``WakeWordDetector``
        (~1-3 s on a cold cache) and opens the PyAudio stream
        (~100 ms on a real audio backend; can hang on a CI env with
        no audio device). 15 s covers the cold-cache case with
        comfortable headroom without over-long flake budgets. If this
        ever fails on a real audio-equipped machine, the right fix
        is to mock ``pyaudio.PyAudio`` rather than to stretch the
        timeout further.
        """
        _ensure_qapp()
        thread = PipelineThread()
        thread.request_stop()
        thread.start()
        thread.wait(15000)
        self.assertFalse(thread.isRunning())

    def test_tts_heartbeat_starts_manual_loop_and_aborts_on_stop(self):
        """Mock pyttsx3.init to verify _speak_with_heartbeat:

        * attempts startLoop(False)
        * calls engine.stop() when request_stop flips _is_running
        * never invokes runAndWait (which would be a blocking fallback
          we don't want on the QThread)
        """
        import src.ui.pipeline as pipeline_mod

        class FakeEngine:
            def __init__(self):
                self.calls: list[str] = []
                self._busy = True

            def setProperty(self, *_args, **_kwargs):
                self.calls.append("setProperty")

            def say(self, _text):
                self.calls.append("say")

            def startLoop(self, blocking: bool):
                self.calls.append(f"startLoop({blocking})")

            def iterate(self):
                self.calls.append("iterate")
                # Each iterate() call costs "time" — after 5 ticks we
                # flip isBusy() off to simulate TTS completing.
                if len([c for c in self.calls if c == "iterate"]) >= 5:
                    self._busy = False

            def isBusy(self) -> bool:
                return self._busy

            def endLoop(self):
                self.calls.append("endLoop")

            def stop(self):
                self.calls.append("stop")
                self._busy = False

            def runAndWait(self):
                self.calls.append("runAndWait")

        fake = FakeEngine()
        original_init = pipeline_mod.pyttsx3.init
        pipeline_mod.pyttsx3.init = lambda: fake
        try:
            thread = PipelineThread()
            thread._is_running = True
            thread._speak_with_heartbeat("any text would do here")
            self.assertIn("startLoop(False)", fake.calls)
            self.assertNotIn("runAndWait", fake.calls)
            self.assertIn("say", fake.calls)
            self.assertTrue(fake.calls.count("iterate") >= 5)
            self.assertIn("endLoop", fake.calls)

            # Now flip _is_running and verify abort path calls stop().
            fake2 = FakeEngine()
            pipeline_mod.pyttsx3.init = lambda: fake2
            thread2 = PipelineThread()
            thread2._is_running = False
            thread2._speak_with_heartbeat("aborted")
            self.assertIn("startLoop(False)", fake2.calls)
            self.assertIn("stop", fake2.calls)
            # Abort must short-circuit BEFORE exhausting iterate() —
            # otherwise the abort would race natural completion.
            self.assertLess(
                sum(1 for c in fake2.calls if c == "iterate"),
                5,
            )
        finally:
            pipeline_mod.pyttsx3.init = original_init

    def test_set_state_resets_level_when_leaving_active_states(self):
        _ensure_qapp()
        surface = AnimatedSurface()
        surface.set_state(AppState.LISTENING.value)
        surface.set_level(1500.0)
        self.assertGreater(surface._level, 0.0)
        surface.set_state(AppState.THINKING.value)
        self.assertEqual(surface._level, 0.0)

    def test_set_state_does_not_reset_level_on_entry(self):
        """Symmetry check: entering LISTENING from WAITING must NOT zero
        the carry-over level (the reset only fires on leave)."""
        _ensure_qapp()
        surface = AnimatedSurface()
        surface.set_state(AppState.WAITING.value)
        surface.set_level(900.0)
        surface.tick(0.0)
        surface.set_state(AppState.LISTENING.value)
        surface._level = 420.0  # inject a known value
        surface.set_state(AppState.LISTENING.value)  # no-op transition
        self.assertEqual(surface._level, 420.0)


if __name__ == "__main__":
    unittest.main()
