"""Visual states of the StardewBot widget.

A single source of truth shared by ``PipelineThread`` (emits the state)
and ``AnimatedSurface`` (renders it). Adding a state here automatically
wires through both ends if the widget implements ``paint_<state>``.
"""
from __future__ import annotations

from enum import Enum


class AppState(str, Enum):
    """Names match the signals emitted by ``PipelineThread``."""

    WAITING = "waiting"     # listening for "hey farmer"
    LISTENING = "listening" # recording user question
    THINKING = "thinking"   # STT + LLM working
    TALKING = "talking"     # TTS playback


# Display labels keyed by state. Used by tests + future status tooltip.
STATE_LABEL: dict[str, str] = {
    AppState.WAITING.value: "say \"hey farmer\"",
    AppState.LISTENING.value: "listening…",
    AppState.THINKING.value: "thinking…",
    AppState.TALKING.value: "speaking…",
}


def is_valid(name: str) -> bool:
    """True when ``name`` is one of the four canonical state names."""
    return name in {state.value for state in AppState}
