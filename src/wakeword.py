import numpy as np
from openwakeword.model import Model

from .config import WAKE_WORD_MODEL, WAKE_WORD_NAME, WAKE_WORD_THRESHOLD


class WakeWordDetector:
    def __init__(self):
        self.model = Model(wakeword_models=[str(WAKE_WORD_MODEL)])

    def reset(self):
        self.model.reset()

    def detect(self, audio_frame: bytes) -> bool:
        frame = np.frombuffer(audio_frame, dtype=np.int16)
        self.model.predict(frame)
        confidence = self.model.prediction_buffer[WAKE_WORD_NAME][-1]
        return confidence > WAKE_WORD_THRESHOLD
