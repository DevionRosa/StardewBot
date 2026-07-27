"""StardewBot desktop UI — a borderless translucent widget.

The package wraps the existing STT/TTS pipeline (assistant.py) without
modifying it. The pipeline runs in a ``QThread`` and emits Qt signals for
each state transition; the on-screen widget renders four visual states
(waiting / listening / thinking / talking) using ``QPainter``.

Entry point: ``python run_ui.py``.
"""

from .states import AppState, STATE_LABEL
from .pipeline import PipelineThread
from .widget import AnimatedSurface
from .window import StardewWidgetWindow

__all__ = [
    "AppState",
    "STATE_LABEL",
    "PipelineThread",
    "AnimatedSurface",
    "StardewWidgetWindow",
]
