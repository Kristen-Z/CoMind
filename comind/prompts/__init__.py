from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent


def prompt_path(task: str) -> Path:
    return PROMPT_DIR / f"{task}.txt"


def build_prompt(task: str, transcript: str) -> str:
    template = prompt_path(task).read_text(encoding="utf-8").strip()
    return template.replace("{transcript}", transcript or "")
