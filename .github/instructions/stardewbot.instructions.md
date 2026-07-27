---
description: "Use when working on the Stardew Valley Voice Assistant project. Enforces local-only Ollama, wake-word audio flow, async safety, and Stardew Valley-only scope."
applyTo: "**"
---
# Stardew Valley Voice Assistant Instructions

- Keep the assistant strictly focused on Stardew Valley data and mechanics. If a request is unrelated, reject it gracefully and fall back to a friendly local response.
- Treat the runtime as fully local and offline. Hard-code all model access to `http://localhost:11434/v1` and do not introduce external network dependencies.
- Preserve the state pipeline: wake-word listening for "Hey farmer", capture the question, send it to the local LLM, synthesize the answer through the default audio output, then return to passive listening.
- Prefer simple, low-latency control flow over complex threading. Use async processing for microphone listening and inference so the main thread never blocks.
- Manage audio resources defensively. Wrap microphone and speaker lifecycle handling in cleanup paths so Windows devices are always released cleanly.
- Keep memory use low and avoid unnecessary buffer allocations while audio is streaming.
- Be pragmatic in refactors. Favor smaller, stable changes over broad abstractions when improving responsiveness or reliability.
- Use concise, self-documenting code. Add brief comments only where buffer sizing, audio chunking, or hardware-specific behavior is not obvious.
- When writing or updating code, keep changes aligned with the existing project layout and avoid unrelated cleanup.
