import os

from .base import Backend


class GeminiBackend(Backend):
    name = "gemini"
    default_model = "gemini-3-flash-preview"
    bbox_format_map = {"xyxy_pixels": "xyxy_1000"}  # answers normalised when asked for pixels

    def __init__(self, model, api_key=None):
        super().__init__(model)
        from google import genai

        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt, frame, context_frames, seed=None):
        contents = [*context_frames, frame, prompt]
        config = None if seed is None else {"seed": seed}
        return self.with_retries(lambda: self.client.models.generate_content(
            model=self.model, contents=contents, config=config).text) or ""
