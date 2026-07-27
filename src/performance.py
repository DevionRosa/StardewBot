"""
Performance monitoring and tuning utilities.
"""
import time
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class PerformanceMetrics:
    """Track performance across pipeline stages."""
    stt_time: float = 0.0
    chat_time: float = 0.0
    tts_time: float = 0.0
    total_time: float = 0.0
    turn_count: int = 0
    times: Dict[str, list] = field(default_factory=lambda: {
        "stt": [],
        "chat": [],
        "tts": [],
        "total": [],
    })

    def record_stt(self, elapsed: float):
        self.stt_time = elapsed
        self.times["stt"].append(elapsed)

    def record_chat(self, elapsed: float):
        self.chat_time = elapsed
        self.times["chat"].append(elapsed)

    def record_tts(self, elapsed: float):
        self.tts_time = elapsed
        self.times["tts"].append(elapsed)

    def record_total(self, elapsed: float):
        self.total_time = elapsed
        self.times["total"].append(elapsed)
        self.turn_count += 1

    def avg_time(self, component: str) -> float:
        times = self.times.get(component, [])
        return sum(times) / len(times) if times else 0.0

    def report(self) -> str:
        if self.turn_count == 0:
            return "[Perf] No data yet."
        
        return (
            f"[Perf] Turns: {self.turn_count} | "
            f"STT: {self.avg_time('stt'):.2f}s | "
            f"Chat: {self.avg_time('chat'):.2f}s | "
            f"TTS: {self.avg_time('tts'):.2f}s | "
            f"Total: {self.avg_time('total'):.2f}s"
        )


_metrics = PerformanceMetrics()


def get_metrics() -> PerformanceMetrics:
    return _metrics


def print_report():
    print(_metrics.report())
