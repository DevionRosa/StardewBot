from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
WAKE_WORD_MODEL = MODEL_DIR / "hey_farmer.onnx"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
WAKE_WORD_NAME = "hey_farmer"
WAKE_WORD_THRESHOLD = 0.6
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = None
CHUNK_SIZE = 1280
POST_WAKE_TIMEOUT_SECONDS = 2.5
QUESTION_SILENCE_SECONDS = 0.9
VOICE_ACTIVITY_THRESHOLD = 450
MAX_RECORD_SECONDS = 8
SILENCE_TIMEOUT_SECONDS = 1.2
TTS_RATE = 165
ENABLE_STREAMING = True
VOSK_CHUNK_SIZE = 4096
OLLAMA_STREAM = True


def _resolve_vosk_model_dir() -> Path:
	env_value = os.getenv("VOSK_MODEL_DIR")
	if env_value:
		return Path(env_value)

	candidates = sorted(MODEL_DIR.glob("vosk-model-*"))
	if candidates:
		return candidates[0]

	return MODEL_DIR / "vosk-model-en-us-0.22"


VOSK_MODEL_DIR = _resolve_vosk_model_dir()
