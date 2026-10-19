from importlib import import_module

from .base import Backend

_BACKENDS = {
    "openai": ("openai_backend", "OpenAIBackend"),
    "claude": ("claude", "ClaudeBackend"),
    "gemini": ("gemini", "GeminiBackend"),
    "qwen": ("qwen", "QwenBackend"),
}
BACKENDS = tuple(_BACKENDS)


def get_backend_class(name: str):
    module, cls = _BACKENDS[name]
    return getattr(import_module(f".{module}", __name__), cls)
