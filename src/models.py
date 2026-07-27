"""
Model selection and configuration helpers for tuning performance.
"""
import requests
from typing import List, Tuple

def list_available_models(base_url: str = "http://localhost:11434") -> List[str]:
    """List all locally available Ollama models."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return sorted([model["name"] for model in data.get("models", []) if model.get("name")])
    except Exception as exc:
        print(f"Failed to list models: {exc}")
        return []


def model_speed_rank() -> List[Tuple[str, str]]:
    """
    Recommend models by speed (fastest first).
    For Stardew: Qwen and Mistral are best at instruction-following and factual accuracy.
    Tuple: (model_name, description)
    """
    return [
        ("tinyllama:latest", "Fastest (1.1B) - good for simple Q&A, less accurate"),
        ("phi:latest", "Fast (2.7B) - reasonable quality, decent speed"),
        ("qwen:latest", "Fast (7B) - RECOMMENDED for Stardew (best accuracy + speed)"),
        ("qwen3.5:9b", "Medium (9B) - Excellent accuracy for Stardew facts"),
        ("mistral:latest", "Medium (7B) - Strong instruction-following"),
        ("llama2:latest", "Medium (7B) - Baseline, may hallucinate more"),
        ("llama3.2:latest", "Medium (8B) - Current default, reasonable balance"),
        ("neural-chat:latest", "Slower (13B) - highest accuracy, slowest"),
    ]


def suggest_fastest_model(available: List[str]) -> str:
    """Suggest the fastest model from available options."""
    for model, _ in model_speed_rank():
        if model in available:
            return model
    return "llama3.2:latest"


def print_model_recommendations():
    """Print model performance recommendations."""
    available = list_available_models()
    print("\n[Models] Available models:")
    for model in available:
        print(f"  - {model}")
    
    print("\n[Models] Speed recommendations (fastest first):")
    for model, desc in model_speed_rank():
        status = "✓" if model in available else " "
        print(f"  [{status}] {model:<30} {desc}")
    
    if available:
        fastest = suggest_fastest_model(available)
        print(f"\n[Models] Fastest available: {fastest}")
        print(f"[Models] To use it: OLLAMA_MODEL={fastest} python app.py")
