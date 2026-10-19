import os

from .base import CONTEXT_JPEG_QUALITY, REFERENCE_FRAME_CAPTION, REFERENCE_JPEG_QUALITY, Backend, encode_jpeg_b64


def _image(img, quality):
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": encode_jpeg_b64(img, quality)}}


class ClaudeBackend(Backend):
    name = "claude"
    default_model = "claude-opus-4-6"

    def __init__(self, model, api_key=None):
        super().__init__(model)
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._rate_limit_error = anthropic.RateLimitError

    def is_rate_limit(self, err):
        return isinstance(err, self._rate_limit_error)

    def generate(self, prompt, frame, context_frames, seed=None):
        content = [{"type": "text", "text": prompt}]
        if context_frames:
            content += [_image(f, CONTEXT_JPEG_QUALITY) for f in context_frames]
            content.append({"type": "text", "text": REFERENCE_FRAME_CAPTION})
        content.append(_image(frame, REFERENCE_JPEG_QUALITY))

        def request():
            response = self.client.messages.create(
                model=self.model, max_tokens=8192, messages=[{"role": "user", "content": content}])
            return "".join(b.text for b in response.content if b.type == "text")

        return self.with_retries(request)
