from contextlib import contextmanager

import pyaudio

from .config import CHANNELS, CHUNK_SIZE, SAMPLE_RATE

FORMAT = pyaudio.paInt16


@contextmanager
def open_input_stream():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )
    try:
        yield audio, stream
    finally:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        audio.terminate()
