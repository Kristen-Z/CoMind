"""Visualize benchmark samples: context frames, reference frame with GT (green) and predicted
(red / blue) boxes, annotation and transcript.

    python -m comind.visualize --task handover --split train --num 4 --out vis/
    python -m comind.visualize --task handover --split train --pred_dir results/handover/gpt_5_1 --out vis/
"""

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import TASKS
from .bbox import parse_pred_box
from .data import CoMindDataset
from .evaluation import load_predictions
from .evaluation.scoia import CUE_TYPE_MAP
from .tasks import get_task

GT_FIELDS = {
    "joint_attention": (("object_category_level_1", "Object"), ("cue_types", "Cue")),
    "handover": (("object_category_level_1", "Object"), ("delivering_flow", "Flow"), ("initiator", "Initiator"),
                 ("initiation_type", "Init type")),
    "scoia": (("verbs", "Verb"), ("nouns", "Noun"), ("annot_type", "Cue")),
}
WIDTH = 1408
MAX_FRAME_HEIGHT = 704


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _fmt(value):
    return ", ".join(map(str, value)) if isinstance(value, list) else str(value)


def draw_boxes(frame, spec, gt_boxes=(), prediction=None, bbox_format=None):
    img = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for box in gt_boxes:
        draw.rectangle(box, outline="lime", width=4)
    for color, key in zip(("red", "blue"), spec.bbox_keys if prediction else ()):
        box = parse_pred_box(prediction.get(key), bbox_format, spec.frame_width, spec.frame_height)
        if box is not None:
            draw.rectangle(box, outline=color, width=4)
    return img


def draw_prediction(frame, prediction: dict, spec, bbox_format: str, gt_boxes=()):
    """Reference frame with GT and predicted boxes and the predicted labels."""
    img = draw_boxes(frame, spec, gt_boxes, prediction, bbox_format)
    draw, font, y = ImageDraw.Draw(img), _font(18), 8
    for key, label in spec.display_fields:
        if key in prediction:
            line = f"{label}: {_fmt(prediction[key])}"
            draw.rectangle(draw.textbbox((8, y), line, font=font), fill="black")
            draw.text((8, y), line, fill="yellow", font=font)
            y += 24
    return img


def render_sample(sample, prediction=None, bbox_format=None, num_frames=5):
    """One figure: context frames, reference frame with boxes, annotation and transcript."""
    spec = get_task(sample.task)
    ref = draw_boxes(sample.load_frame(), spec, sample.gt_boxes(), prediction, bbox_format)
    scale = min(WIDTH / ref.width, MAX_FRAME_HEIGHT / ref.height)
    ref = ref.resize((round(ref.width * scale), round(ref.height * scale)))

    context = sample.load_context_frames(num_frames)
    thumb_w = WIDTH // num_frames
    thumbs = [f.resize((thumb_w, round(f.height * thumb_w / f.width))) for f in context]
    strip_h = thumbs[0].height if thumbs else 0

    font = _font(24)
    lines = [f"{sample.task}  {sample.video_id}  frame {sample.frame_id}"]
    ann = sample.annotation or {}
    if sample.task == "scoia" and ann.get("annot_type"):
        ann = {**ann, "annot_type": CUE_TYPE_MAP.get(ann["annot_type"], ann["annot_type"])}
    gt = "  |  ".join(f"{label}: {_fmt(ann[k])}" for k, label in GT_FIELDS[sample.task] if ann.get(k) is not None)
    if gt:
        lines.append("GT  " + gt)
    if prediction:
        lines.append("Pred  " + "  |  ".join(
            f"{label}: {_fmt(prediction[k])}" for k, label in spec.display_fields if k in prediction))
    if sample.transcript:
        lines += textwrap.wrap(f'Transcript: "{sample.transcript}"', 110)
    text_h = 34 * len(lines) + 16

    canvas = Image.new("RGB", (WIDTH, strip_h + ref.height + text_h), "white")
    for i, t in enumerate(thumbs):
        canvas.paste(t, (i * thumb_w, 0))
    canvas.paste(ref, ((WIDTH - ref.width) // 2, strip_h))
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((12, strip_h + ref.height + 8 + 34 * i), line, fill="black", font=font)
    return canvas


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--data_root", help="default: $COMIND_DATA_ROOT")
    p.add_argument("--split", default="train", choices=("train", "test", "all"))
    p.add_argument("--pred_dir", type=Path, help="also draw the predictions of this run")
    p.add_argument("--keys", nargs="+", help="samples to draw, as <video_id>_<frame_id> (default: the first --num)")
    p.add_argument("--num", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("vis"))
    args = p.parse_args(argv)

    spec = get_task(args.task)
    ds = CoMindDataset(args.task, split=args.split, root=args.data_root)
    preds, bbox_format = {}, None
    if args.pred_dir:
        preds, _ = load_predictions(args.pred_dir)
        config = args.pred_dir / "run_config.json"
        bbox_format = json.loads(config.read_text())["bbox_format"] if config.exists() else spec.bbox_format

    samples = [s for s in ds if s.frame_path.exists() and (not args.keys or s.key in args.keys)]
    if not args.keys:
        samples = samples[:args.num]
    args.out.mkdir(parents=True, exist_ok=True)
    for s in samples:
        fig = render_sample(s, preds.get((s.video_id, s.frame_id)), bbox_format, max(spec.num_frames, 5))
        fig.save(args.out / f"{args.task}_{s.key}.jpg", quality=85)
        print(args.out / f"{args.task}_{s.key}.jpg")


if __name__ == "__main__":
    main()
