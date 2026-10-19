import sys
import types
from types import SimpleNamespace

import pytest
from PIL import Image

from comind.backends import get_backend_class
from comind.backends.base import REFERENCE_FRAME_CAPTION, Backend

FRAME = Image.new("RGB", (64, 32), "white")
CONTEXT = [Image.new("RGB", (64, 32), "black")] * 3


class Recorder:
    def __init__(self, response):
        self.kwargs, self.response = None, response

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_openai_layout(monkeypatch):
    pytest.importorskip("openai")
    backend = get_backend_class("openai")("gpt-test", api_key="sk-test")
    rec = Recorder(SimpleNamespace(output_text="ok"))
    backend.client = SimpleNamespace(responses=SimpleNamespace(create=rec))

    assert backend.generate("PROMPT", FRAME, CONTEXT) == "ok"
    content = rec.kwargs["input"][0]["content"]
    assert [c["type"] for c in content] == ["input_text"] + ["input_image"] * 3 + ["input_text", "input_image"]
    assert content[0]["text"] == "PROMPT" and content[4]["text"] == REFERENCE_FRAME_CAPTION
    assert [c["detail"] for c in content if c["type"] == "input_image"] == ["low"] * 3 + ["high"]

    backend.generate("PROMPT", FRAME, [])
    assert [c["type"] for c in rec.kwargs["input"][0]["content"]] == ["input_text", "input_image"]


def test_claude_layout():
    pytest.importorskip("anthropic")
    backend = get_backend_class("claude")("claude-test", api_key="sk-test")
    rec = Recorder(SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")]))
    backend.client = SimpleNamespace(messages=SimpleNamespace(create=rec))

    assert backend.generate("PROMPT", FRAME, CONTEXT) == "ok"
    content = rec.kwargs["messages"][0]["content"]
    assert [c["type"] for c in content] == ["text"] + ["image"] * 3 + ["text", "image"]
    assert rec.kwargs["max_tokens"] == 8192


def test_gemini_layout(monkeypatch):
    rec = Recorder(SimpleNamespace(text="ok"))
    client = SimpleNamespace(models=SimpleNamespace(generate_content=rec))
    genai = types.ModuleType("google.genai")
    genai.Client = lambda api_key: client
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    backend = get_backend_class("gemini")("gemini-test", api_key="key")
    assert backend.generate("PROMPT", FRAME, CONTEXT) == "ok"
    assert rec.kwargs["contents"] == [*CONTEXT, FRAME, "PROMPT"]
    backend.generate("PROMPT", FRAME, CONTEXT, seed=3)
    assert rec.kwargs["config"] == {"seed": 3}


class Flaky(Backend):
    name = "flaky"

    def __init__(self, errors):
        super().__init__("m")
        self.errors = list(errors)

    def generate(self, prompt, frame, context_frames, seed=None):
        def request():
            if self.errors:
                raise self.errors.pop(0)
            return "ok"
        return self.with_retries(request)


def test_retries_on_rate_limit(monkeypatch):
    waits = []
    monkeypatch.setattr("comind.backends.base.time.sleep", waits.append)
    assert Flaky([RuntimeError("429 retry after 3 seconds")]).generate("", FRAME, []) == "ok"
    assert waits == [8]
    assert Flaky([RuntimeError("bad request")]).generate("", FRAME, []) == ""
