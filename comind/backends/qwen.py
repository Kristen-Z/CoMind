import logging

from ..data.video import uniform_indices
from .base import Backend

log = logging.getLogger(__name__)


CLIP_FPS = 30
CLIP_FRAMES = 10 * CLIP_FPS


class QwenBackend(Backend):
    """Local Qwen-VL with Hugging Face transformers. Context frames are passed as one video."""

    name = "qwen"
    default_model = "Qwen/Qwen3-VL-8B-Instruct"
    bbox_format_map = {"xyxy_pixels": "xyxy_1000", "yxyx_1000": "xyxy_1000"}  # always its native format

    def __init__(self, model, cache_dir=None, dtype="auto", max_new_tokens=8192):
        super().__init__(model)
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        try:
            import flash_attn  # noqa: F401
            attn = "flash_attention_2"
        except ImportError:
            attn = "sdpa"
        self.torch = torch
        self.net = AutoModelForImageTextToText.from_pretrained(
            model, dtype=dtype if dtype == "auto" else getattr(torch, dtype), device_map="auto",
            cache_dir=cache_dir, attn_implementation=attn).eval()
        self.processor = AutoProcessor.from_pretrained(model, cache_dir=cache_dir)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt, frame, context_frames, seed=None):
        content = [{"type": "video", "video": context_frames}] if context_frames else []
        content += [{"type": "image", "image": frame}, {"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        video_kwargs = {}
        if context_frames:
            # Use exactly the given frames (no re-sampling) and their real timestamps in the 10 s clip.
            meta = {"total_num_frames": CLIP_FRAMES, "fps": CLIP_FPS,
                    "frames_indices": uniform_indices(CLIP_FRAMES, len(context_frames))}
            video_kwargs = {"videos": [context_frames], "video_metadata": [meta], "do_sample_frames": False}
        inputs = self.processor(text=[text], images=[frame], padding=True, return_tensors="pt",
                                **video_kwargs).to(self.net.device)
        if seed is not None:
            self.torch.manual_seed(seed)
        with self.torch.inference_mode():
            out = self.net.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
