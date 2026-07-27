"""Qt worker thread that drives the StardewBot pipeline.

Composition over inheritance: the thread owns a ``WakeWordDetector``
and uses the same audio / STT / chat / TTS primitives the CLI
``StardewAssistant`` uses — but with state emissions and a clean
shutdown flag threaded through.

Risks this module — or its companion ``run_ui.py`` entry point —
mitigates explicitly:

* ``pyttsx3`` SAPI5 on Windows requires COM initialised on the worker
  thread (``pythoncom.CoInitialize``); else ``engine.runAndWait()``
  raises ``COMError 0x80010106``.
* ``StardewAssistant.run()`` exits only on ``KeyboardInterrupt``.
  Closing the widget relies on the ``_is_running`` flag and
  ``request_stop()`` to break the wake-word loop and release the
  PyAudio stream deterministically.
* ``openwakeword.model.Model`` is observed to silently produce
  zero wake-word confidence when invoked from a thread that did
  not construct it. We build ``WakeWordDetector`` *inside*
  ``run()`` so construct-and-use stay on a single OS thread —
  ``app.py`` shares that invariant via its single-threaded
  assistant loop, which is why CLI mode has always worked.
* Qt's event loop runs mostly in C++; without a signal handler,
  SIGINT could only land at Python bytecodes — historically the
  ``_tick`` QTimer callback, leaking the worker + mic as an
  uncaught ``KeyboardInterrupt``. ``run_ui.py`` now installs
  ``SIGINT`` and ``SIGTERM`` handlers (with a 100 ms Windows
  keepalive QTimer) so the widget closes through ``closeEvent``
  directly, bypassing ``_tick`` entirely.
"""
from __future__ import annotations

import random
import time
from contextlib import suppress
from typing import Optional

import numpy as np
import pyttsx3
from PySide6.QtCore import QObject, QThread, Signal

from ..audio import open_input_stream
from ..chat import answer_question
from ..config import (
    CHUNK_SIZE,
    MAX_RECORD_SECONDS,
    POST_WAKE_TIMEOUT_SECONDS,
    QUESTION_SILENCE_SECONDS,
    SAMPLE_RATE,
    VOICE_ACTIVITY_THRESHOLD,
    TTS_RATE,
)
from ..transcribe import transcribe_audio
from ..wakeword import WakeWordDetector
from .states import AppState


class PipelineThread(QThread):
    """Background thread running wakeword -> listen -> think -> talk."""

    # State changes (waiting / listening / thinking / talking).
    state_changed = Signal(str)
    # RMS audio level emitted while recording the question so the UI
    # can pulse the listening indicator. Also used during TTS as a
    # synthetic heartbeat so the talking bars keep moving.
    level_changed = Signal(float)
    # Final transcript after STT (empty string if no speech detected).
    transcript_ready = Signal(str)
    # Final answer to be spoken by TTS.
    answer_ready = Signal(str)
    # Critical failure — UI shows it in red and falls back to waiting.
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._is_running = True
        self._com_initialized = False
        # Lazy: ``detect()`` must run on the same OS thread that
        # constructed the underlying ``openwakeword.model.Model``.
        # Earlier we built ``WakeWordDetector`` here on the main GUI
        # thread and then used it from the worker QThread, which
        # silently produced zero wake-word confidence — and because the
        # worker also printed nothing, the failure looked like a mic
        # problem. ``app.py`` works because its assistant loop lives in
        # one thread and both constructs and uses the detector there.
        # Constructing inside ``run()`` restores that single-thread
        # invariant.
        #
        # Stays ``None`` until ``run()`` succeeds; consumers should
        # treat ``None`` as "thread never reached the wake loop".
        self._detector: Optional[WakeWordDetector] = None

    # ---------------------------------------------------------------- lifecycle

    def request_stop(self) -> None:
        """Ask the wakeword loop to exit on the next poll.

        Safe to call from any thread (the worker reads it once per
        audio chunk).
        """
        self._is_running = False

    def run(self) -> None:  # noqa: D401 — Qt override, not a custom entry point
        # Build the wake-word detector on this worker thread first so
        # the onnxruntime-backed ``openwakeword.model.Model`` is created
        # in the same OS thread that ``detect()`` runs in. See
        # ``PipelineThread.__init__`` for the full rationale — building
        # on the GUI thread and using on the QThread silently produces
        # zero confidence, mimicking a microphone failure.
        try:
            self._detector = WakeWordDetector()
        except Exception as exc:
            # Bubble the failure up to the UI as a ``state_changed`` /
            # ``error`` so the operator can see it instead of staring at
            # an apparently live but silent Waiting pulse. We also flip
            # ``_is_running`` defensively so any cleanup a future
            # ``_run_pipeline`` ``finally`` block adds cannot run on a
            # thread that "thinks" it should still be listening.
            print(f"[Pipeline] Wake-word detector init failed: {exc}")
            self._is_running = False
            self.error.emit(str(exc))
            self.state_changed.emit(AppState.WAITING.value)
            return

        # COM init: required on Windows before pyttsx3/SAPI5 calls
        # succeed from this non-main thread. Track the call so the
        # matching CoUninitialize is skipped on platforms where
        # CoInitialize silently failed (Linux/macOS, missing pywin32).
        try:
            import pythoncom  # type: ignore[import-not-found]
            pythoncom.CoInitialize()
            self._com_initialized = True
        except ImportError:
            # Non-Windows or pywin32 not installed — pyttsx3 picks a
            # non-COM driver (espeak) which works without CoInitialize.
            pass
        except Exception as exc:  # pragma: no cover — defensive
            print(f"[Pipeline] CoInitialize failed: {exc}")

        # NOTE: Vosk is intentionally NOT eagerly warmed here.
        # ``transcribe._get_model`` is a singleton that loads the model
        # on first call. Eagerly calling it from ``run()`` was an
        # earlier optimisation intended to avoid first-turn latency,
        # but the cost (>5 s on a cold cache in some CI environments)
        # bled into ``request_stop`` test budgets and made the worker
        # thread appear never to exit. Letting the *first*
        # ``transcribe_audio`` call load the model instead moves the
        # cold-load cost to the first user turn, where the THINKING
        # spinner animation already hides a few seconds of latency.
        try:
            self._run_pipeline()
        except Exception as exc:  # pragma: no cover — defensive
            # Symmetric defensive flip — the exception already unwinds
            # the worker thread today, but a future ``_run_pipeline``
            # ``finally`` block (e.g. mic cleanup) should still see
            # ``_is_running=False`` rather than risk a redundant chunk
            # of audio read.
            self._is_running = False
            self.error.emit(str(exc))
            self.state_changed.emit(AppState.WAITING.value)
        finally:
            if self._com_initialized:
                with suppress(Exception):
                    import pythoncom  # type: ignore[import-not-found]
                    pythoncom.CoUninitialize()

    # ---------------------------------------------------------------- loop

    def _run_pipeline(self) -> None:
        """Drive wakeword -> record -> transcribe -> answer -> speak."""
        def _turn_over() -> None:
            if self._detector is not None:
                self._detector.reset()

        # Mirror app.py's "Passive Mode" pivot so a widget runner can
        # confirm from the terminal that the wake-word loop is alive.
        print("[Pipeline] Listening for 'Hey farmer'...")
        with open_input_stream() as (_, stream):
            while self._is_running:
                self.state_changed.emit(AppState.WAITING.value)

                if not self._wait_for_wake(stream):
                    break  # request_stop() flipped _is_running

                # Wake word fired — explicitly transition out of Waiting
                # so the widget renders the listening rings + RMS dot
                # instead of the slow pulse. Earlier this state was never
                # emitted and the user saw no animation change on capture.
                print("[Pipeline] Wake word detected — recording…")
                self.state_changed.emit(AppState.LISTENING.value)
                audio_bytes = self._capture_question(stream)
                if not audio_bytes or not self._is_running:
                    print("[Pipeline] No question captured.")
                    _turn_over()
                    continue

                print("[Pipeline] Transcribing…")
                transcript = transcribe_audio(audio_bytes, correct=True)
                self.transcript_ready.emit(transcript)
                if not transcript or not self._is_running:
                    print("[Pipeline] Empty transcript.")
                    _turn_over()
                    continue
                print(f"[Pipeline] Transcript: {transcript!r}")

                self.state_changed.emit(AppState.THINKING.value)
                print("[Pipeline] Generating answer…")
                answer = answer_question(transcript)
                self.answer_ready.emit(answer)
                if not answer or not self._is_running:
                    print("[Pipeline] Empty answer.")
                    _turn_over()
                    continue
                print(f"[Pipeline] Answer: {answer!r}")

                self.state_changed.emit(AppState.TALKING.value)
                self._speak_with_heartbeat(answer)
                print("[Pipeline] Ready for next turn.")
                _turn_over()

    def _wait_for_wake(self, stream) -> bool:
        """Block on the wake-word detector until the user says
        "hey farmer" or ``request_stop`` is called."""
        while self._is_running:
            raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            if self._detector.detect(raw):
                return True
        return False

    def _capture_question(self, stream) -> bytes:
        """Capture the user question with voice activity detection, and
        emit a per-chunk RMS level so the listening ring can pulse.

        Behaviour mirrors ``StardewAssistant.capture_question_audio``:
        we wait up to ``POST_WAKE_TIMEOUT_SECONDS`` for first voice,
        then collect until either ``QUESTION_SILENCE_SECONDS`` of quiet
        or ``MAX_RECORD_SECONDS`` is reached.
        """
        frames: list[bytes] = []
        max_chunks = int((MAX_RECORD_SECONDS * SAMPLE_RATE) / CHUNK_SIZE)
        start_timeout = max(
            1, int((POST_WAKE_TIMEOUT_SECONDS * SAMPLE_RATE) / CHUNK_SIZE)
        )
        end_silence = max(
            1, int((QUESTION_SILENCE_SECONDS * SAMPLE_RATE) / CHUNK_SIZE)
        )

        speech_started = False
        speech_wait = 0
        silence = 0

        for _ in range(max_chunks):
            if not self._is_running:
                break
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            level = self._audio_level(data)
            self.level_changed.emit(level)

            if not speech_started:
                if level >= VOICE_ACTIVITY_THRESHOLD:
                    speech_started = True
                    frames.append(data)
                else:
                    speech_wait += 1
                    if speech_wait >= start_timeout:
                        return b""
                continue

            frames.append(data)
            if level >= VOICE_ACTIVITY_THRESHOLD:
                silence = 0
            else:
                silence += 1
                if silence >= end_silence:
                    break
        return b"".join(frames)

    @staticmethod
    def _audio_level(frame: bytes) -> float:
        """Mean absolute amplitude of a 16-bit PCM frame.

        Implemented byte-for-byte identically to
        ``src.assistant.StardewAssistant._audio_level`` so the two
        pipelines share one RMS semantics. A signed-mean
        (``samples.mean()``) hovers near zero for typical speech
        because positive and negative samples cancel, which means the
        value never crosses ``VOICE_ACTIVITY_THRESHOLD`` and the
        recording loop times out instead of capturing the question.
        """
        samples = np.frombuffer(frame, dtype=np.int16)
        if samples.size == 0:
            return 0.0
        return float(np.mean(np.abs(samples)))

    def _speak_with_heartbeat(self, text: str) -> None:
        """Speak ``text`` directly on the QThread (already COM-initialised
        on Windows) using pyttsx3's non-blocking ``startLoop(False)`` /
        ``iterate()`` mode, emitting a synthetic level signal at ~20 Hz
        so the talking bars keep animating and honouring ``_is_running``
        so closing the window mid-sentence stops promptly.

        Falls back to a blocking ``engine.runAndWait()`` if the driver
        does not support manual loops (e.g., older espeak on Linux) —
        the talking animation then simply freezes for the duration of
        the speech. A cache-friendly fallback ``engine.say + runAndWait``
        is tried last to make sure we always speak at least once.
        """
        try:
            engine = pyttsx3.init()
        except Exception as exc:
            print(f"[UI] pyttsx3 init failed: {exc}")
            return

        try:
            try:
                engine.setProperty("rate", TTS_RATE)
                engine.setProperty("volume", 1.0)
            except Exception:
                pass
            engine.say(text)

            used_manual_loop = False
            try:
                engine.startLoop(False)
                used_manual_loop = True
            except Exception as exc:
                print(f"[UI] startLoop(False) unsupported: {exc}; falling back")

            if used_manual_loop:
                last_emit = 0.0
                try:
                    while engine.isBusy():
                        if not self._is_running:
                            try:
                                engine.stop()
                            except Exception:
                                pass
                            break
                        now = time.time()
                        if now - last_emit > 0.05:  # 20 Hz heartbeat
                            jitter = random.gauss(700, 220)
                            self.level_changed.emit(max(120.0, jitter))
                            last_emit = now
                        try:
                            engine.iterate()
                        except Exception:
                            break
                        time.sleep(0.01)
                finally:
                    with suppress(Exception):
                        engine.endLoop()
                    with suppress(Exception):
                        engine.stop()
                    if self._is_running:
                        # No heartbeat emitted during the loop because it
                        # was empty — push one final level so the bars
                        # collapse cleanly.
                        self.level_changed.emit(120.0)
            else:
                # Blocking fallback — UI bars freeze for the sentence.
                if not self._is_running:
                    return
                engine.runAndWait()
        except Exception as exc:
            # Without this catch an unexpected exception from pyttsx3
            # ``engine.say`` / driver init / SAPI5 backend would let the
            # turn roll through TALKING with zero audio and no log —
            # the user sees the bars animate, then Waiting resume. Make
            # the failure visible so an operator knows the driver broke.
            print(f"[Pipeline] Speaking failed: {exc}")
        finally:
            with suppress(Exception):
                del engine


# Exported for symmetry with the existing TTS module even though the UI
# doesn't actually call it — keeping the import surface area predictable.
__all__ = ["PipelineThread", "AppState"]
