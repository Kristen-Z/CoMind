import os

from .base import CONTEXT_JPEG_QUALITY, REFERENCE_FRAME_CAPTION, REFERENCE_JPEG_QUALITY, Backend, encode_jpeg_b64


def _image(img, quality, detail):
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encode_jpeg_b64(img, quality)}",
            "detail": detail}


class OpenAIBackend(Backend):
    """Context frames at low detail, reference frame at high detail."""

    name = "openai"
    default_model = "gpt-5.1"

    def __init__(self, model, api_key=None):
        super().__init__(model)
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def generate(self, prompt, frame, context_frames, seed=None):
        content = [{"type": "input_text", "text": prompt}]
        if context_frames:
            content += [_image(f, CONTEXT_JPEG_QUALITY, "low") for f in context_frames]
            content.append({"type": "input_text", "text": REFERENCE_FRAME_CAPTION})
        content.append(_image(frame, REFERENCE_JPEG_QUALITY, "high"))
        return self.with_retries(lambda: self.client.responses.create(
            model=self.model, input=[{"role": "user", "content": content}]).output_text)
