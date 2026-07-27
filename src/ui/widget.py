"""Transparent QWidget that paints the four visual states.

All animations are driven from ``paintEvent``; nothing happens until a
60 fps ``QTimer`` in :class:`src.ui.window.StardewWidgetWindow` calls
``update()``. The painter primitives are pure functions so the unit
tests can render a single frame offscreen (via the ``offscreen``
QPA platform plugin) without spinning a real event loop.

Design notes:

* The widget is fully transparent — every pixel outside the painted
  orb has alpha=0. ``QPainter`` antialiasing gives smooth edges on a
  translucent background without needing a bitmap mask.
* ``Listening`` uses the live RMS signal emitted by the pipeline; the
  ring expansion phase is purely time-driven so the animation keeps
  flowing even if the level signal stalls momentarily.
* ``Thinking`` is a three-arc spinner with phase rotation. We avoid
  QPropertyAnimation here so the same code paths work in tests.
* ``Talking`` uses a wavetable sampled deterministically from the
  current frame index; this is robust to waveform clipping and
  doesn't require running QAudioOutput taps.
"""
from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

from .states import AppState, STATE_LABEL


# Tunable aesthetics. Kept here (not passed in) so the widget is its
# own self-contained component.
_OUTER_RADIUS = 46        # px — radius of the painted orb
_DOT_RADIUS = 14          # px — radius of the inner pulse core
_RING_COUNT = 3           # listening rings on screen at once
_BAR_COUNT = 5            # talking bars
_BAR_WIDTH = 8
_BAR_GAP = 6
_COLOR_CORE = QColor(255, 215, 100, 230)      # warm gold
_COLOR_RING = QColor(120, 220, 160, 220)      # moss green
_COLOR_ARC_PRIMARY = QColor(255, 220, 110, 255)
_COLOR_ARC_SECONDARY = QColor(110, 180, 240, 220)
_COLOR_BAR_FLOOR = QColor(255, 255, 255, 60)
_COLOR_BAR_CEIL = QColor(255, 215, 100, 240)
_COLOR_LABEL = QColor(255, 255, 255, 200)


class AnimatedSurface(QWidget):
    """Paints one of the four states; no Qt animations needed."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = AppState.WAITING.value
        self._level = 0.0           # RMS during LISTENING; synthetic during TALKING
        self._phase = 0.0           # 0..1 cycle phase driven by the 60 fps timer
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

    # ---------------------------------------------------------------- Qt props

    def set_state(self, state: str) -> None:
        """Switch state. Unknown names are silently ignored so a noisy
        signal source can't crash the UI. Resets the smoothed audio
        level whenever we leave LISTENING or TALKING so bars settle
        cleanly between turns instead of carrying residual energy."""
        if state == self._state:
            return
        if state not in {s.value for s in AppState}:
            return
        leaving = self._state in {AppState.LISTENING.value, AppState.TALKING.value}
        self._state = state
        if leaving:
            self._level = 0.0
        self.update()

    def state(self) -> str:
        return self._state

    def set_level(self, level: float) -> None:
        """Smooth level from the pipeline. Clamped to a friendly range
        so a single loud peak doesn't blow the bar layout."""
        clamped = max(0.0, min(level, 4000.0))
        # Cheap single-pole IIR smooths jittered RMS values so the
        # ripples breathe instead of popping.
        self._level = 0.65 * self._level + 0.35 * clamped
        self.update()

    def tick(self, phase: float) -> None:
        """Advance the animation phase and repaint."""
        self._phase = phase % 1.0
        self.update()

    # ---------------------------------------------------------------- paint

    def paintEvent(self, _event) -> None:  # noqa: N802 — Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            center = QPointF(self.width() / 2.0, self.height() / 2.0)
            outer = float(min(self.width(), self.height())) / 2.0 - 4.0

            if self._state == AppState.WAITING.value:
                self._paint_waiting(painter, center, outer)
            elif self._state == AppState.LISTENING.value:
                self._paint_listening(painter, center, outer)
            elif self._state == AppState.THINKING.value:
                self._paint_thinking(painter, center, outer)
            elif self._state == AppState.TALKING.value:
                self._paint_talking(painter, center, outer)

            self._paint_label(painter, center, outer)
        finally:
            painter.end()

    # ---------------------------------------------------------------- states

    def _paint_waiting(
        self, painter: QPainter, center: QPointF, outer: float
    ) -> None:
        """Slow-pulsing orb. Breath cycle is 4 s."""
        # sin gives -1..1; map to 0.78..1.0 radius scale.
        scale = 0.78 + 0.22 * (0.5 * (1.0 - math.cos(self._phase * 2.0 * math.pi)))
        alpha = 110 + int(80 * (0.5 + 0.5 * math.sin(self._phase * 2.0 * math.pi)))
        gradient = QRadialGradient(center, outer * scale)
        gradient.setColorAt(0.0, QColor(255, 215, 100, alpha))
        gradient.setColorAt(0.7, QColor(120, 220, 160, max(0, alpha - 60)))
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, outer * scale, outer * scale)

        # Inner sparkle
        painter.setBrush(QColor(255, 240, 180, 220))
        painter.drawEllipse(center, _DOT_RADIUS * 0.55, _DOT_RADIUS * 0.55)

    def _paint_listening(
        self, painter: QPainter, center: QPointF, outer: float
    ) -> None:
        """Three expanding concentric rings + a central dot whose size
        tracks the most recent RMS level."""
        # Normalise level to 0..1 using a 1500 RMS ceiling (typical
        # conversational speech with 16-bit PCM sits around 600-1200).
        normalised = min(1.0, self._level / 1500.0)
        for ring in range(_RING_COUNT):
            ring_phase = (self._phase + ring / _RING_COUNT) % 1.0
            ring_radius = outer * (0.55 + 0.40 * ring_phase)
            ring_alpha = int((1.0 - ring_phase) * 180)
            if ring_alpha <= 0:
                continue
            pen = QPen(QColor(120, 220, 160, ring_alpha))
            pen.setWidthF(3.0)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(center, ring_radius, ring_radius)

        # Inner dot — sized by RMS amplitude, with a soft floor.
        dot_radius = _DOT_RADIUS * (0.55 + 0.45 * normalised)
        painter.setBrush(QColor(255, 215, 100, 240))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, dot_radius, dot_radius)

    def _paint_thinking(
        self, painter: QPainter, center: QPointF, outer: float
    ) -> None:
        """Three rotating arcs orbing the centre, offset so the gaps
        form a slow turbine. The arc span is negative so the arcs spin
        *clockwise* — Qt's drawArc defaults to counterclockwise
        which feels reversed."""
        for arc_index in range(3):
            pen = QPen(
                _COLOR_ARC_PRIMARY if arc_index == 0
                else _COLOR_ARC_SECONDARY if arc_index == 1
                else QColor(220, 180, 250, 220)
            )
            pen.setWidthF(5.0)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            arc_radius = outer * 0.72
            spin = (self._phase * 360.0) + arc_index * 120.0
            start_angle = int((spin - 50) * 16)  # Qt uses 1/16th degree
            span = -int(100 * 16)                # negative = clockwise
            rect = QRectF(
                center.x() - arc_radius,
                center.y() - arc_radius,
                arc_radius * 2,
                arc_radius * 2,
            )
            painter.drawArc(rect, start_angle, span)

        # Bright centre pip
        painter.setBrush(QColor(255, 240, 200, 235))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, 8.0, 8.0)

    def _paint_talking(
        self, painter: QPainter, center: QPointF, outer: float
    ) -> None:
        """Five bouncing bars across the centre line. Heights derive
        from the current level signal mixed with a deterministic wave
        so the bars never go flat, even when the heart-beat pauses."""
        bar_total = _BAR_COUNT * _BAR_WIDTH + (_BAR_COUNT - 1) * _BAR_GAP
        start_x = center.x() - bar_total / 2.0
        floor_height = 10.0
        ceiling = outer * 0.85

        # Modulate by synthetic sine envelope
        envelope = 0.45 + 0.55 * abs(math.sin(self._phase * math.pi * 2.0))
        level_norm = min(1.0, self._level / 1500.0)

        for bar in range(_BAR_COUNT):
            x = start_x + bar * (_BAR_WIDTH + _BAR_GAP)
            # Per-bar phase offset gives a "running" appearance.
            offset = math.sin(self._phase * math.pi * 2.0 + bar * 0.7)
            height = floor_height + (ceiling - floor_height) * (
                0.55 * envelope + 0.45 * level_norm * (0.5 + 0.5 * offset)
            )
            top = QPointF(x + _BAR_WIDTH / 2.0, center.y() - height / 2.0)
            bottom = QPointF(x + _BAR_WIDTH / 2.0, center.y() + height / 2.0)

            # Gradient bar
            bar_gradient = QRadialGradient(
                top, max(height / 2.0, 30.0)
            )
            bar_gradient.setColorAt(0.0, _COLOR_BAR_CEIL)
            bar_gradient.setColorAt(1.0, _COLOR_BAR_FLOOR)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(bar_gradient))

            path = QPainterPath()
            rounded = QRectF(x, center.y() - height / 2.0, _BAR_WIDTH, height)
            path.addRoundedRect(rounded, _BAR_WIDTH / 2.0, _BAR_WIDTH / 2.0)
            painter.drawPath(path)

    def _paint_label(
        self, painter: QPainter, center: QPointF, outer: float
    ) -> None:
        """Tiny status label tucked beneath the orb."""
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(_COLOR_LABEL)
        label_rect = QRectF(
            center.x() - outer,
            center.y() + outer - 6,
            outer * 2,
            24,
        )
        painter.drawText(
            label_rect,
            Qt.AlignHCenter | Qt.AlignTop,
            STATE_LABEL.get(self._state, ""),
        )
