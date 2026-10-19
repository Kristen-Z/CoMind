"""Run a VLM baseline on one CoMind task.

    python -m comind.infer --task handover --backend openai --model gpt-5.1

Writes one raw answer per sample to <output_dir>/<video_id>_<frame_id>.txt. Existing answers
are skipped, so an interrupted run can be resumed with the same command.
"""

import argparse
import json
import logging
import re
import zlib
from pathlib import Path

from . import TASKS
from .backends import BACKENDS, get_backend_class
from .data import CoMindDataset
from .parsing import parse_response
from .prompts import build_prompt
from .tasks import get_task
from .visualize import draw_prediction

log = logging.getLogger("comind.infer")


def model_dirname(model: str) -> str:
    """Qwen/Qwen3-VL-8B-Instruct -> qwen3_vl_8b_instruct"""
    return re.sub(r"[^a-z0-9]+", "_", model.rstrip("/").split("/")[-1].lower()).strip("_")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--backend", required=True, choices=BACKENDS)
    p.add_argument("--model", help="default depends on the backend")
    p.add_argument("--data_root", help="default: $COMIND_DATA_ROOT")
    p.add_argument("--split", default="test", choices=("train", "test", "all"))
    p.add_argument("--output_dir", help="default: results/<task>/<model>")
    p.add_argument("--limit", type=int, help="only run the first N samples")
    p.add_argument("--save_vis", action="store_true", help="save frames with drawn boxes to <output_dir>_vis")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (Qwen, Gemini; the OpenAI and Anthropic APIs "
                                                        "do not support seeding)")
    p.add_argument("--api_key", help="default: OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY")
    p.add_argument("--cache_dir", help="Hugging Face cache for Qwen weights")
    p.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float16"), help="Qwen weights dtype")
    return p.parse_args(argv)


def build_backend(args):
    cls = get_backend_class(args.backend)
    if args.backend == "qwen":
        return cls(args.model, cache_dir=args.cache_dir, dtype=args.dtype)
    return cls(args.model, api_key=args.api_key)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for name in ("httpx", "google_genai"):  # per-request logs of the API clients
        logging.getLogger(name).setLevel(logging.WARNING)
    args = parse_args(argv)
    spec = get_task(args.task)
    backend_cls = get_backend_class(args.backend)
    args.model = args.model or backend_cls.default_model
    bbox_format = backend_cls.bbox_format_map.get(spec.bbox_format, spec.bbox_format)

    dataset = CoMindDataset(args.task, split=args.split, root=args.data_root)
    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    out_dir = Path(args.output_dir or Path("results") / args.task / model_dirname(args.model))
    vis_dir = out_dir.with_name(out_dir.name + "_vis")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(exist_ok=True)

    config = {"task": args.task, "backend": args.backend, "model": args.model,
              "num_frames": spec.num_frames, "bbox_format": bbox_format, "seed": args.seed}
    config_path = out_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit(f"{out_dir} contains a run with a different configuration, use another --output_dir.")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    log.info("%r -> %s", dataset, out_dir)

    backend = build_backend(args)
    missing = 0
    for i in range(n):
        sample = dataset[i]
        out_path = out_dir / f"{sample.key}.txt"
        if out_path.exists():
            continue
        if not sample.frame_path.exists() or (spec.num_frames and not sample.clip_path.exists()):
            missing += 1
            continue

        # per-sample seed: results do not depend on run order or on resuming
        seed = (args.seed + zlib.crc32(sample.key.encode())) % 2**31
        answer = backend.generate(build_prompt(args.task, sample.transcript), sample.load_frame(),
                                  sample.load_context_frames(spec.num_frames), seed=seed)
        if not answer.strip():
            log.warning("[%d/%d] %s: empty answer", i + 1, n, sample.key)
            continue
        out_path.write_text(answer, encoding="utf-8")
        log.info("[%d/%d] %s", i + 1, n, sample.key)

        if args.save_vis and (prediction := parse_response(answer)):
            draw_prediction(sample.load_frame(), prediction, spec, bbox_format,
                            sample.gt_boxes()).save(vis_dir / f"{sample.key}.jpg")
    if missing:
        log.warning("%d samples skipped: inputs not found (download the recordings and run comind.prepare)", missing)


if __name__ == "__main__":
    main()
