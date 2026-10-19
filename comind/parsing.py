import json
import re


def _last_json_answer(text: str):
    """Last JSON object (or list of objects) in the text, for answers preceded by reasoning."""
    for close_char, open_char in (("}", "{"), ("]", "[")):
        end = text.rfind(close_char)
        while end != -1:
            depth, start = 0, -1
            for i in range(end, -1, -1):
                if text[i] == close_char:
                    depth += 1
                elif text[i] == open_char:
                    depth -= 1
                    if depth == 0:
                        start = i
                        break
            if start >= 0:
                try:
                    parsed = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return parsed[0]
            end = text.rfind(close_char, 0, end)
    return None


def parse_response(text: str):
    """Answer dict from a raw model response, or None."""
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    try:
        parsed = json.loads(match.group(1).strip() if match else text)
    except json.JSONDecodeError:
        return _last_json_answer(text)
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else None
    return parsed if isinstance(parsed, dict) else None
