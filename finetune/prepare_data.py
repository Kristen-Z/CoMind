"""Write the Qwen3-VL fine-tuning data: one conversation per annotated sample of the train split.

    python finetune/prepare_data.py --data_root data/comind

Run `python -m comind.prepare --split train` first. The model gets the inputs of `comind.infer` (the reference
frame, for handover and SCOIA after the 5 context frames, and the transcript) with the same prompts, and
the answer is the annotation in the JSON format the prompt asks for. Writes
<data_root>/finetune/<task>_train.json and the context frames next to it, as JPEGs; image paths in the JSON
are relative to it.
"""

import argparse
import json
import logging
import os
from pathlib import Path

from comind import TASKS
from comind.data import CoMindDataset, resolve_root
from comind.prompts import build_prompt
from comind.tasks import get_task

log = logging.getLogger("finetune.prepare_data")

JPEG_QUALITY = 95  # context frames, as in the paper's training data (OpenCV default)

JA_CUES = {"gaze": "<gaze>", "action": "<action>", "gesture": "<gestural>", "communication": "<verbal>"}
SCOIA_CUES = {"A": "<verbal>", "G": "<gestural>", "AG": "<verbal and gestural>", "O": "<other>"}
FLOWS = {"ltr": "<left to right>", "rtl": "<right to left>"}
INITIATION_TYPES = {  # any other combination: <implicit>
    frozenset({"verbal"}): "<verbal>",
    frozenset({"gestural"}): "<gestural>",
    frozenset({"verbal", "gestural"}): "<verbal and gestural>",
}


def to_yxyx_1000(bbox, width, height, x_lo=0, x_hi=1000) -> list:
    """GT [x, y, w, h] in pixels -> [ymin, xmin, ymax, xmax] in [0, 1000], with x clamped to [x_lo, x_hi]."""
    x, y, w, h = bbox
    x0, x1 = sorted((x, x + w))
    y0, y1 = sorted((y, y + h))
    xs = [max(x_lo, min(x_hi, round(v / width * 1000))) for v in (x0, x1)]
    ys = [max(0, min(1000, round(v / height * 1000))) for v in (y0, y1)]
    return [ys[0], xs[0], ys[1], xs[1]]


def object_category(ann) -> str:
    return (ann.get("object_category_level_1") or ann.get("object_category_level_2")
            or ann.get("object_category_level_3") or "")


def joint_attention_answer(ann, spec):
    cues = [JA_CUES[c] for c in ann.get("cue_types") or [] if c in JA_CUES]
    boxes = ann.get("bounding_boxes") or []
    left = [b for b in boxes if b[0] + b[2] / 2 < spec.frame_width / 2]
    right = [b for b in boxes if b[0] + b[2] / 2 >= spec.frame_width / 2]
    if not cues or not left or not right:
        return None

    def largest(bs):
        return max(bs, key=lambda b: abs(b[2] * b[3]))

    # each box stays within its own view: x in [0, 500] for the left one, [500, 1000] for the right one
    return {
        "cue type": cues,
        "object category": object_category(ann),
        "left object bounding box": to_yxyx_1000(largest(left), spec.frame_width, spec.frame_height, x_hi=500),
        "right object bounding box": to_yxyx_1000(largest(right), spec.frame_width, spec.frame_height, x_lo=500),
    }


def handover_answer(ann, spec):
    if ann.get("skip") or not ann.get("bbox"):
        return None
    return {
        "handover happen offset": round(ann["end_time"] - ann["start_time"]),
        "delivering flow": FLOWS.get(ann["delivering_flow"], ann["delivering_flow"]),
        "initiator": f"<{ann['initiator']}>",
        "initiation type": INITIATION_TYPES.get(frozenset(ann["initiation_type"]), "<implicit>"),
        "object category": object_category(ann),
        "object bounding box": to_yxyx_1000(ann["bbox"], spec.frame_width, spec.frame_height),
    }


def scoia_answer(ann, spec):
    if ann.get("annot_type") is None or not ann.get("bbox"):
        return None
    verbs, nouns = ann.get("verbs") or [], ann.get("nouns") or []
    return {
        "cue type": SCOIA_CUES.get(ann["annot_type"], "<other>"),
        "action verb": verbs[0] if verbs else "",
        "action noun": nouns[0] if nouns else "",
        "object bounding box": to_yxyx_1000(ann["bbox"], spec.frame_width, spec.frame_height),
    }


ANSWERS = {"joint_attention": joint_attention_answer, "handover": handover_answer, "scoia": scoia_answer}


def conversation(sample, out_dir: Path):
    """Training sample in the format of qwenvl/data, or None if the annotation lacks what the answer needs.
    Writes the context frames to <out_dir>/<task>/<video_id>/<frame_id>/<i>.jpg."""
    spec = get_task(sample.task)
    answer = ANSWERS[sample.task](sample.annotation or {}, spec)
    if answer is None:
        return None
    images = []
    if spec.num_frames:
        frame_dir = out_dir / sample.task / sample.video_id / sample.frame_id
        images = [frame_dir / f"{i}.jpg" for i in range(spec.num_frames)]
        if not all(p.exists() for p in images):
            frames = sample.load_context_frames(spec.num_frames)
            if len(frames) < spec.num_frames:
                log.warning("%s: %d context frames in %s, skipped", sample.key, len(frames), sample.clip_path)
                return None
            frame_dir.mkdir(parents=True, exist_ok=True)
            for frame, path in zip(frames, images):
                frame.save(path, quality=JPEG_QUALITY)
    images.append(sample.frame_path)
    return {
        "image": [os.path.relpath(p, out_dir) for p in images],
        "conversations": [
            {"from": "human", "value": "<image>\n" * len(images) + build_prompt(sample.task, sample.transcript)},
            {"from": "gpt", "value": json.dumps([answer], indent=2)},
        ],
    }


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", help="default: $COMIND_DATA_ROOT")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    p.add_argument("--output_dir", type=Path, help="default: <data_root>/finetune")
    args = p.parse_args(argv)
    root = resolve_root(args.data_root)
    out_dir = args.output_dir or root / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        spec, dataset = get_task(task), CoMindDataset(task, split="train", root=root)
        samples, missing = [], 0
        for i in range(len(dataset)):
            sample = dataset[i]
            if not sample.frame_path.exists() or (spec.num_frames and not sample.clip_path.exists()):
                missing += 1
                continue
            if (conv := conversation(sample, out_dir)) is not None:
                samples.append(conv)
            if (i + 1) % 500 == 0:
                log.info("%s: %d/%d", task, i + 1, len(dataset))
        path = out_dir / f"{task}_train.json"
        path.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
        log.info("%s: %d of %d samples -> %s", task, len(samples), len(dataset), path)
        if missing:
            log.warning("%s: %d samples skipped, inputs not found (run comind.prepare --split train)", task, missing)


if __name__ == "__main__":
    main()
