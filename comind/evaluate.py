"""Evaluate predictions written by comind.infer.

    python -m comind.evaluate --task handover --pred_dir results/handover/gpt_5_1

Writes <pred_dir>/evaluation_results.json with overall and per-session metrics.
"""

import argparse
import json
import os
from pathlib import Path

from . import TASKS
from .bbox import BBOX_FORMATS, canonical_bbox_format
from .data.dataset import ENV_VAR, annotation_path, load_annotations, load_split
from .evaluation import get_evaluator, load_predictions, lookup
from .evaluation.handover import DEFAULT_TIME_TOLERANCE
from .tasks import get_task


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--pred_dir", required=True, type=Path)
    p.add_argument("--data_root", help="default: $COMIND_DATA_ROOT")
    p.add_argument("--gt", type=Path, help="default: <data_root>/annotations/dataset_<task>_consolidated.json")
    p.add_argument("--split", default="test", choices=("train", "test", "all"))
    p.add_argument("--bbox_format", type=canonical_bbox_format, metavar="{" + ",".join(BBOX_FORMATS) + "}",
                   help="default: from run_config.json")
    p.add_argument("--time_tolerance", type=float, default=DEFAULT_TIME_TOLERANCE, help="handover only, seconds")
    p.add_argument("--output", type=Path, help="default: <pred_dir>/evaluation_results.json")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.data_root or os.environ.get(ENV_VAR) or (args.gt.parent.parent if args.gt else None)
    if root is None:
        raise SystemExit(f"Pass --data_root or --gt, or set ${ENV_VAR}.")
    gt_path = args.gt or annotation_path(root, args.task)

    bbox_format = args.bbox_format
    if bbox_format is None:
        config = args.pred_dir / "run_config.json"
        bbox_format = (json.loads(config.read_text())["bbox_format"] if config.exists()
                       else get_task(args.task).bbox_format)

    evaluator = get_evaluator(args.task, **({"time_tolerance": args.time_tolerance} if args.task == "handover" else {}))
    gt = {k: v for k, v in load_annotations(gt_path, load_split(root, args.split)).items() if v is not None}
    if not gt:
        raise SystemExit(f"No {args.split} annotations in {gt_path}. Test annotations are held out: "
                         "submit test predictions to the leaderboard, or evaluate on --split train.")
    preds, unparsable = load_predictions(args.pred_dir)
    results = evaluator.evaluate(gt, preds, bbox_format)
    results["config"] = {"gt": str(gt_path), "split": args.split, "bbox_format": bbox_format,
                         "unparsable_files": unparsable}

    output = args.output or args.pred_dir / "evaluation_results.json"
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.pred_dir}: {results['num_evaluated']}/{results['num_gt']} samples, {len(unparsable)} unparsable")
    print("  ".join(f"{label}={lookup(results['overall'], path):.4f}" for label, path in evaluator.headline))


if __name__ == "__main__":
    main()
