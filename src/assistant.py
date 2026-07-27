import time

import numpy as np

from .audio import open_input_stream
from .config import (
    CHUNK_SIZE,
    MAX_RECORD_SECONDS,
    POST_WAKE_TIMEOUT_SECONDS,
    QUESTION_SILENCE_SECONDS,
    SAMPLE_RATE,
    SILENCE_TIMEOUT_SECONDS,
    VOICE_ACTIVITY_THRESHOLD,
)
from .chat import answer_question
from .transcribe import transcribe_audio
from .tts import speak
from .wakeword import WakeWordDetector
from .performance import get_metrics


def _warm_up_assistant():
    try:
        from .chat import _resolve_model_name

        _resolve_model_name()
    except Exception as exc:
        print(f"[Warmup] Ollama warmup skipped: {exc}")

    try:
        from .transcribe import _get_model

        _get_model()
    except Exception as exc:
        print(f"[Warmup] STT warmup skipped: {exc}")


class StardewAssistant:
    def __init__(self):
        self.detector = WakeWordDetector()

    def listen_for_wake_word(self, stream):
        print("\n[Passive Mode] Listening for 'Hey farmer'...")
        while True:
            raw_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            if self.detector.detect(raw_data):
                return True

    def _audio_level(self, frame: bytes) -> float:
        samples = np.frombuffer(frame, dtype=np.int16)
        if samples.size == 0:
            return 0.0
        return float(np.mean(np.abs(samples)))

    def _is_voice_frame(self, frame: bytes) -> bool:
        return self._audio_level(frame) >= VOICE_ACTIVITY_THRESHOLD

    def _read_question_frame(self, stream):
        return stream.read(CHUNK_SIZE, exception_on_overflow=False)

    def capture_question_audio(self, stream):
        print("[Active Mode] Listening for your question...")
        frames = []
        max_chunks = int((MAX_RECORD_SECONDS * SAMPLE_RATE) / CHUNK_SIZE)
        start_timeout_chunks = max(1, int((POST_WAKE_TIMEOUT_SECONDS * SAMPLE_RATE) / CHUNK_SIZE))
        end_silence_chunks = max(1, int((QUESTION_SILENCE_SECONDS * SAMPLE_RATE) / CHUNK_SIZE))

        speech_started = False
        speech_wait_chunks = 0
        silence_chunks = 0

        for _ in range(max_chunks):
            data = self._read_question_frame(stream)

            if not speech_started:
                if self._is_voice_frame(data):
                    speech_started = True
                    frames.append(data)
                    print("[Active Mode] Speech detected.")
                else:
                    speech_wait_chunks += 1
                    if speech_wait_chunks >= start_timeout_chunks:
                        print("[Active Mode] No speech detected after wake word timeout.")
                        break
                continue

            frames.append(data)

            if self._is_voice_frame(data):
                silence_chunks = 0
            else:
                silence_chunks += 1
                if silence_chunks >= end_silence_chunks:
                    print("[Active Mode] Question complete.")
                    break

        return b"".join(frames)

    def run(self):
        _warm_up_assistant()
        metrics = get_metrics()
        
        with open_input_stream() as (_, stream):
            try:
                while True:
                    if self.listen_for_wake_word(stream):
                        turn_start = time.time()
                        audio_bytes = self.capture_question_audio(stream)
                        
                        # STT timing
                        stt_start = time.time()
                        transcript = transcribe_audio(audio_bytes)
                        stt_elapsed = time.time() - stt_start
                        metrics.record_stt(stt_elapsed)
                        
                        if transcript:
                            print(f"[Transcript] {transcript}")
                            
                            # Chat timing
                            chat_start = time.time()
                            answer = answer_question(transcript)
                            chat_elapsed = time.time() - chat_start
                            metrics.record_chat(chat_elapsed)
                            
                            print(f"[Ollama] {answer}")
                            
                            # TTS timing
                            tts_start = time.time()
                            speak(answer, async_mode=False)
                            tts_elapsed = time.time() - tts_start
                            metrics.record_tts(tts_elapsed)
                        else:
                            print("[Transcript] No speech detected.")
                        
                        turn_elapsed = time.time() - turn_start
                        metrics.record_total(turn_elapsed)
                        print(f"[Turn] Total time: {turn_elapsed:.2f}s")
                        print(metrics.report())
                        
                        self.detector.reset()
                        time.sleep(0.2)
            except KeyboardInterrupt:
                print("\nShutting down assistant...")
                print(metrics.report())
            except Exception as exc:
                print(f"[Error] Unexpected error in main loop: {exc}")
                raise
