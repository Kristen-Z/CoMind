"""Predict with a Qwen3-VL model fine-tuned by finetune/train.sh.

    python finetune/infer.py --task scoia --adapter checkpoints/qwen3_vl_8b_scoia

Writes the files of `python -m comind.infer` (one raw answer per sample, <recording_id>_<frame_id>.txt, and
run_config.json), so the output can be evaluated with `python -m comind.evaluate` or submitted to the
leaderboard. The model gets the inputs it was trained on: the prompts of comind.infer, the context frames
as separate images before the reference frame, and the image budget of train.sh. Its boxes are in the format
the prompts ask for, [ymin, xmin, ymax, xmax] normalised to [0, 1000]. Without --adapter, the base model gets
the same inputs.
"""

import argparse
import json
import logging
import zlib
from pathlib import Path

from comind import TASKS
from comind.data import CoMindDataset
from comind.infer import model_dirname
from comind.prompts import build_prompt
from comind.tasks import get_task

log = logging.getLogger("finetune.infer")

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


class FinetunedQwen:
    """Qwen3-VL with the LoRA adapter merged into its weights."""

    def __init__(self, model, adapter=None, cache_dir=None, min_pixels=784, max_pixels=401408):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        try:
            import flash_attn  # noqa: F401
            attn = "flash_attention_2"
        except ImportError:
            attn = "sdpa"
        self.torch = torch
        net = AutoModelForImageTextToText.from_pretrained(model, dtype="auto", device_map="auto",
                                                          cache_dir=cache_dir, attn_implementation=attn)
        if adapter:
            from peft import PeftModel
            net = PeftModel.from_pretrained(net, adapter).merge_and_unload()
        self.net = net.eval()
        self.processor = AutoProcessor.from_pretrained(model, cache_dir=cache_dir)
        # image budget of training, set as by update_processor_pixels in qwenvl/data/data_processor.py
        ip = self.processor.image_processor
        ip.min_pixels, ip.max_pixels = min_pixels, max_pixels
        ip.size["shortest_edge"], ip.size["longest_edge"] = min_pixels, max_pixels

    def generate(self, prompt, frame, context_frames, seed=None):
        images = [*context_frames, frame]
        content = [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt").to(self.net.device)
        if seed is not None:
            self.torch.manual_seed(seed)
        with self.torch.inference_mode():
            out = self.net.generate(**inputs, max_new_tokens=8192)
        return self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--adapter", help="output or checkpoint directory of train.sh")
    p.add_argument("--model", help=f"base model, default: the adapter's, or {DEFAULT_MODEL} without an adapter")
    p.add_argument("--data_root", help="default: $COMIND_DATA_ROOT")
    p.add_argument("--split", default="test", choices=("train", "test", "all"))
    p.add_argument("--output_dir", help="default: results/<task>/<adapter name>")
    p.add_argument("--limit", type=int, help="only run the first N samples")
    p.add_argument("--seed", type=int, default=0, help="sampling seed")
    p.add_argument("--cache_dir", help="Hugging Face cache for the base model")
    p.add_argument("--min_pixels", type=int, default=784, help="per image, as in training")
    p.add_argument("--max_pixels", type=int, default=401408, help="per image, as in training")
    return p.parse_args(argv)


def run_name(adapter, model) -> str:
    """checkpoints/qwen3_vl_8b_scoia/checkpoint-500 -> qwen3_vl_8b_scoia_checkpoint_500"""
    if not adapter:
        return model_dirname(model) + "_base"
    path = Path(adapter).resolve()
    return model_dirname(f"{path.parent.name}_{path.name}" if path.name.startswith("checkpoint-") else path.name)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)
    if not args.model:
        args.model = (json.loads((Path(args.adapter) / "adapter_config.json").read_text())["base_model_name_or_path"]
                      if args.adapter else DEFAULT_MODEL)
    spec = get_task(args.task)

    dataset = CoMindDataset(args.task, split=args.split, root=args.data_root)
    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    out_dir = Path(args.output_dir or Path("results") / args.task / run_name(args.adapter, args.model))
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {"task": args.task, "backend": "qwen_finetuned", "model": args.model, "adapter": args.adapter,
              "num_frames": spec.num_frames, "bbox_format": spec.bbox_format, "seed": args.seed,
              "min_pixels": args.min_pixels, "max_pixels": args.max_pixels}
    config_path = out_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit(f"{out_dir} contains a run with a different configuration, use another --output_dir.")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    log.info("%r -> %s", dataset, out_dir)

    model = FinetunedQwen(args.model, args.adapter, cache_dir=args.cache_dir, min_pixels=args.min_pixels,
                          max_pixels=args.max_pixels)
    missing = 0
    for i in range(n):
        sample = dataset[i]
        out_path = out_dir / f"{sample.key}.txt"
        if out_path.exists():
            continue
        if not sample.frame_path.exists() or (spec.num_frames and not sample.clip_path.exists()):
            missing += 1
            continue

        # per-sample seed, as in comind.infer
        seed = (args.seed + zlib.crc32(sample.key.encode())) % 2**31
        answer = model.generate(build_prompt(args.task, sample.transcript), sample.load_frame(),
                                sample.load_context_frames(spec.num_frames), seed=seed)
        if not answer.strip():
            log.warning("[%d/%d] %s: empty answer", i + 1, n, sample.key)
            continue
        out_path.write_text(answer, encoding="utf-8")
        log.info("[%d/%d] %s", i + 1, n, sample.key)
    if missing:
        log.warning("%d samples skipped: inputs not found (download the recordings and run comind.prepare)", missing)


if __name__ == "__main__":
    main()
