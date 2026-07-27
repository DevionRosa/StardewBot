import pyttsx3
from contextlib import suppress
import time
import threading

from .config import TTS_RATE


def _speak_impl(text: str) -> None:
    if not text:
        return

    max_retries = 2
    start_time = time.time()
    for attempt in range(max_retries):
        try:
            engine = pyttsx3.init()
            try:
                engine.setProperty("rate", TTS_RATE)
                engine.setProperty("volume", 1.0)
                engine.say(text)
                engine.runAndWait()
                elapsed = time.time() - start_time
                print(f"[TTS] Speech played in {elapsed:.2f}s")
                return
            finally:
                with suppress(Exception):
                    engine.stop()
        except Exception as exc:
            print(f"[TTS] Attempt {attempt + 1}/{max_retries} failed: {exc}")
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                print(f"[TTS] Failed after {max_retries} attempts.")


def speak(text: str, async_mode: bool = False) -> None:
    if not text:
        return

    if async_mode:
        thread = threading.Thread(target=_speak_impl, args=(text,), daemon=True)
        thread.start()
    else:
        _speak_impl(text)