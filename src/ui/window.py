"""Borderless, translucent on-screen widget window.

Stack:

* ``Qt.FramelessWindowHint`` removes the title bar / system frame.
* ``Qt.WindowStaysOnTopHint`` keeps the widget above game windows so
  it stays visible while Stardew is in focus.
* ``Qt.Tool`` keeps it out of the taskbar.
* ``WA_TranslucentBackground`` lets the painter clear to alpha=0 so
  only the orb + label are opaque.

The widget can be dragged with the mouse (custom mouse events). Right
click toggles "always-on-top" so the user can stick it in front of a
fullscreen game when they want, or move it aside for screenshots.
"""
from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import Qt, QPoint, QTimer
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import QMenu, QWidget

from .pipeline import PipelineThread
from .states import AppState
from .widget import AnimatedSurface


_DEFAULT_WIDGET_PX = 180   # square workspace reserved for the orb + label

# Animation cycle period per state (seconds). Hoisted out of ``_tick``
# so the dict is built once at import time, not 60 times per second.
_STATE_CYCLE_SECONDS: dict[str, float] = {
    AppState.WAITING.value:   4.0,
    AppState.LISTENING.value: 2.0,
    AppState.THINKING.value:  1.6,
    AppState.TALKING.value:   0.8,
}


class StardewWidgetWindow(QWidget):
    """Owns the pipeline thread, the animated surface, and the timer."""

    def __init__(
        self,
        size: int = _DEFAULT_WIDGET_PX,
        on_top: bool = True,
        always_on_top_toggle: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._size_px = max(120, size)
        self._on_top_default = on_top
        self._always_on_top_toggle = always_on_top_toggle
        self._drag_offset: Optional[QPoint] = None

        self._apply_window_flags(self._on_top_default)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(self._size_px, self._size_px)

        # Animated surface fills the whole window.
        self._surface = AnimatedSurface(self)
        self._surface.setGeometry(0, 0, self._size_px, self._size_px)

        # Pipeline worker.
        self._pipeline = PipelineThread(self)
        self._pipeline.state_changed.connect(self._on_state)
        self._pipeline.level_changed.connect(self._surface.set_level)
        self._pipeline.transcript_ready.connect(self._on_transcript)
        self._pipeline.answer_ready.connect(self._on_answer)
        self._pipeline.error.connect(self._on_error)
        self._pipeline.finished.connect(self._on_pipeline_finished)

        # 60 fps phase timer — Qt timers tick in the GUI thread so
        # they can't fight the audio loop in the worker.
        self._animation_timer = QTimer(self)
        self._animation_timer.setTimerType(Qt.PreciseTimer)
        self._animation_timer.timeout.connect(self._tick)
        self._animation_timer.start(1000 // 60)
        self._start_time = time.time()

        self._center_on_initial_screen()

    # ---------------------------------------------------------------- setup

    def _apply_window_flags(self, on_top: bool) -> None:
        flags = Qt.FramelessWindowHint | Qt.Tool
        if on_top:
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)

    def _center_on_initial_screen(self) -> None:
        """Centre on the primary screen on first show."""
        screen = self.screen()
        if screen is None:
            return
        rect = screen.availableGeometry()
        self.move(
            rect.center().x() - self._size_px // 2,
            rect.center().y() - self._size_px // 2,
        )

    # ---------------------------------------------------------------- drivers

    def start(self) -> None:
        """Begin the wakeword loop. Safe to call multiple times."""
        if not self._pipeline.isRunning():
            self._pipeline.start()

    def _tick(self) -> None:
        elapsed = time.time() - self._start_time
        # Cycle length is hand-tuned per state to feel "calm but alive".
        cycle_seconds = _STATE_CYCLE_SECONDS.get(self._surface.state(), 2.0)
        self._surface.tick((elapsed % cycle_seconds) / cycle_seconds)

    # ---------------------------------------------------------------- signals

    def _on_state(self, name: str) -> None:
        if name in {s.value for s in AppState}:
            self._surface.set_state(name)

    def _on_transcript(self, transcript: str) -> None:
        # Surface is purely visual; full transcript logging happens in
        # the pipeline thread's stdout prints. Keep this slot as the
        # extension point for future captions / text echo.
        pass

    def _on_answer(self, answer: str) -> None:
        pass

    def _on_error(self, message: str) -> None:
        print(f"[UI] Pipeline error: {message}")
        self._surface.set_state(AppState.WAITING.value)

    def _on_pipeline_finished(self) -> None:
        # Wakeword loop died (clean shutdown). Surface stays on WAITING
        # until the next start() call.
        pass

    # ---------------------------------------------------------------- drag

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt naming
        if event.button() == Qt.LeftButton:
            self._drag_offset = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()
        elif (
            event.button() == Qt.RightButton
            and self._always_on_top_toggle
        ):
            self._show_context_menu(event.globalPosition().toPoint())
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 — Qt naming
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 — Qt naming
        if event.button() == Qt.LeftButton:
            self._drag_offset = None
            event.accept()

    # ---------------------------------------------------------------- menu

    def _show_context_menu(self, global_pos: QPoint) -> None:
        menu = QMenu(self)
        toggle_action = QAction(
            "Pin to top" if not self._on_top_default else "Unpin",
            self,
        )
        toggle_action.triggered.connect(self._toggle_always_on_top)
        menu.addAction(toggle_action)
        menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        menu.addAction(quit_action)
        menu.exec(global_pos)

    def _toggle_always_on_top(self) -> None:
        new_state = not self._on_top_default
        self._on_top_default = new_state
        self._apply_window_flags(new_state)
        # Re-applying flags hides the window on some WMs; re-show.
        self.show()

    # ---------------------------------------------------------------- close

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — Qt naming
        if self._pipeline.isRunning():
            self._pipeline.request_stop()
            # Wakeword loop checks _is_running every chunk (~80 ms) and
            # the TTS heartbeat polls at ~30 ms, so 1.5 s is plenty for
            # the audio + pyttsx3 to unwind. We deliberately avoid the
            # unsafe QThread.terminate() escape hatch — Qt docs warn it
            # can leave QObjects in undefined state.
            self._pipeline.wait(1500)
        self._animation_timer.stop()
        super().closeEvent(event)
