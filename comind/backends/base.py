import base64
import io
import logging
import re
import time
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

# JPEG quality for base64 images, as in the paper runs (OpenCV default / PIL default)
CONTEXT_JPEG_QUALITY = 95
REFERENCE_JPEG_QUALITY = 75
REFERENCE_FRAME_CAPTION = "This is the reference frame (prediction frame):"


def encode_jpeg_b64(img, quality: int) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class Backend(ABC):
    name = "base"
    default_model = None
    # Box format the model answers in, for each format the prompt can ask for, where the two differ.
    bbox_format_map = {}
    max_retries = 5

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, frame, context_frames: list, seed: int | None = None) -> str:
        """Raw text answer for one sample (reference frame + sampled context frames).

        `seed` makes sampling reproducible where the model supports it (Qwen, Gemini).
        """

    def is_rate_limit(self, err: Exception) -> bool:
        msg = str(err).lower()
        return any(s in msg for s in ("429", "rate limit", "quota", "resource_exhausted"))

    def with_retries(self, request):
        """Retry on rate limits with backoff. Other errors return "" so the sample is retried next run."""
        for attempt in range(self.max_retries):
            try:
                return request()
            except Exception as err:
                if not self.is_rate_limit(err):
                    log.error("%s request failed: %s", self.name, err)
                    return ""
                m = re.search(r"(?:after|seconds:)\s*(\d+)", str(err))
                wait = int(m.group(1)) + 5 if m else min(10 * 2 ** attempt, 120)
                log.warning("Rate limited, waiting %ds", wait)
                time.sleep(wait)
        return ""
