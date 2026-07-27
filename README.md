# StardewBot

Local-only Stardew Valley voice assistant with **deterministic, wiki-grounded
factual accuracy**. Falls back to an LLM only when no structured answer exists
in a pre-built local corpus.

Run it as a terminal-launched desktop widget:

```bash
python run_ui.py                  # default 180px, pinned on top
python run_ui.py --size 240       # larger widget
python run_ui.py --no-on-top      # does not stay above other windows
```

A borderless, translucent window pops up. It drives the full STT/TTS pipeline
and renders one of four visual states at a time:

| State | Trigger | Animation |
| --- | --- | --- |
| **Waiting** | wake-word detector idle | slow-pulsing gold orb (4 s breath) |
| **Listening** | recording the user's question | 3 expanding rings + RMS-modulated dot |
| **Thinking** | STT + LLM working | 3 clockwise-rotating arcs |
| **Talking** | TTS playback | 5 bouncing bars synced to a ~20 Hz heartbeat |

Left-click the widget to drag it. Right-click to toggle *Pin/Unpin* or quit.
Closing the window releases the microphone and exits cleanly.

## Pipeline

1. Listen for the wake word `Hey farmer` (uses `models/hey_farmer.onnx`).
2. Capture the user's spoken question with voice-activity detection.
3. Transcribe locally with Vosk, then fuzzy-correct proper nouns against the
   Stardew vocabulary (`Krobus` from a misheard `Crow bus`).
4. Look the question up in the local corpus:
   - **Tier 1 (deterministic)** — match a structured entity row (Pike,
     Abigail, Pumpkin, …) and emit a templated answer from the infobox with
     **zero LLM involvement**. This is the only path that guarantees zero
     hallucination.
   - **Tier 2 (strict rephraser)** — when an entity row exists but its
     infobox is sparse, fetch the cleaned page body and pass it to Ollama
     together with verified facts. The system prompt forbids inventing facts
     outside the supplied context.
   - **Tier 3 (free-form LLM)** — last-resort fallback only. The bot first
     refuses with a "run `build_wiki_corpus.py`" message when the local
     corpus is empty.
5. Speak the grounded response with `pyttsx3`.

## Quick start

```bash
pip install -r requirements.txt     # installs PySide6, vosk, openwakeword, …
python scripts/build_wiki_corpus.py # one-time wiki crawl (~10–15 min, ~100 MB)
python run_ui.py                    # launch the widget
```

You'll also need a Vosk model directory on disk before the first run —
either `models/vosk-model-<lang>/` next to this README, or set the
`VOSK_MODEL_DIR` environment variable. The repo already ships
`models/vosk-model-en-us-0.22/` if you cloned with the model included;
otherwise download one from <https://alphacephei.com/vosk/models> and
unzip it into `models/`. Without a model STT raises `FileNotFoundError`
on first capture.

> **Before running the builder** the bot refuses free-form LLM answers and
> instead prompts you to run it. This is intentional: an LLM with no
> ground-truth data will hallucinate.

## Optional environment variables

- `STARDEW_VOSK_GRAMMAR=strict` — restrict Vosk to the Stardew vocabulary.
  Off by default because a constrained grammar hurts follow-up coverage.
- `OLLAMA_MODEL=qwen3.5:9b` — model recommendation. The default
  (`llama3.2:latest`) works but Qwen gives the best accuracy/perf trade.
- `OLLAMA_TIMEOUT_SECONDS=60` — Ollama call timeout.
- `VOICE_ACTIVITY_THRESHOLD=450` — energy threshold for question detection
  (tune lower for quieter speech, higher for noisier rooms).

## Headless / terminal-only mode

If you don't want a window, the same loop runs in the terminal:

```bash
python app.py
```

Keyboard interrupt (`Ctrl+C`) prints the final metrics report and exits.

## Layout

- `app.py` — terminal entrypoint (wake-word loop in the terminal).
- `run_ui.py` — desktop widget entrypoint (transparent borderless window).
- `src/assistant.py` — wake-word audio pipeline used by `app.py`.
- `src/audio.py` — microphone stream management.
- `src/wakeword.py` — wake word detection (OpenWakeWord).
- `src/transcribe.py` — Vosk STT + Stardew-aware fuzzy correction.
- `src/chat.py` — three-tier answer pipeline (entity → strict rephrase → free-form).
- `src/wiki_index.py` — local wiki cache and entity-aware retrieval.
- `src/stardew_knowledge.py` — small curated facts fallback.
- `src/tts.py` — local TTS with retry logic.
- `src/config.py` — constants and model paths.
- `src/models.py` — Ollama model selection and recommendations.
- `src/performance.py` — performance metrics tracking.
- `src/wiki_mcp_server.py` — JSON-RPC server exposing the wiki corpus.
- `src/ui/`
  - `__init__.py` — package marker + re-exports (`AnimatedSurface`,
    `PipelineThread`, `StardewWidgetWindow`, `AppState`).
  - `states.py` — `AppState` enum + display labels (single source of truth).
  - `pipeline.py` — `PipelineThread(QThread)` that wraps the
    wake → record → transcribe → answer → speak loop and emits Qt signals
    for each transition.
  - `widget.py` — `AnimatedSurface` with `paintEvent` for all four states
    (waiting pulse, listening rings, thinking arcs, talking bars).
  - `window.py` — `StardewWidgetWindow` (frameless, translucent,
    always-on-top, drag-to-move, 60 fps phase timer).
- `scripts/build_wiki_corpus.py` — one-off wiki pre-crawl builder.
- `tests/test_smoke.py` — offline smoke tests for the answer pipeline.
- `tests/test_ui_smoke.py` — offscreen Qt tests for the widget + pipeline thread.

## Wiki MCP server

External agents can reach the same corpus over the Model Context Protocol:

```bash
python wiki_mcp_server.py
```

Tools: `search_wiki`, `get_wiki_context`, `answer_from_wiki`,
`refresh_wiki_cache`, `lookup_entity`, `compose_answer`, `get_vocabulary`,
`fuzzy_correct_transcript`, `get_structured_answer`, `get_corpus_stats`.

## Notes

- Keep the app offline (all processing is local; only the wiki corpus build
  and tier-2/3 chat call the network).
- The Vosk model should live under `models/vosk-model-*/` or be set via
  `VOSK_MODEL_DIR`.
- For best accuracy use Qwen 3.5; for faster responses, switch to
  `qwen:latest` or `phi`. The corpus keeps accuracy high even with smaller
  models because tier 1 never touches the LLM.
- Closing the widget (or right-click → Quit) cleanly stops the pipeline
  thread and releases the microphone within ~1.5 s. On Windows the worker
  thread COM-initialises so `pyttsx3` SAPI5 calls succeed off the main
  thread.
