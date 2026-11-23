from collections import defaultdict
from pathlib import Path

from ..bbox import iou, parse_pred_box
from ..parsing import parse_response
from ..tasks import TaskSpec

IOU_THRESHOLDS = (0.25, 0.50)


def load_predictions(pred_dir, strict: bool = False) -> tuple[dict, list[str]]:
    """({(video_id, frame_id): answer}, [unparsable files]) for all <video_id>_<frame_id>.txt.
    strict: parse as in the paper's evaluation (see parse_response)."""
    preds, unparsable = {}, []
    for path in sorted(Path(pred_dir).glob("*.txt")):
        video_id, sep, frame_id = path.stem.rpartition("_")
        if not sep:
            continue
        answer = parse_response(path.read_text(encoding="utf-8"), strict=strict)
        if answer is None:
            unparsable.append(path.name)
        else:
            preds[(video_id, frame_id)] = answer
    return preds, unparsable


def as_text(value) -> str:
    """Model answers are not always strings (null, lists)."""
    if isinstance(value, list):
        return " ".join(map(str, value))
    return "" if value is None else str(value)


def strip_tags(s) -> str:
    return as_text(s).strip().lower().replace("<", "").replace(">", "").strip()


def _same_word(a: str, b: str) -> bool:
    """Equal up to an English plural ending: cup/cups, box/boxes, cherry/cherries."""
    a, b = sorted((a, b), key=len)
    return b in (a, a + "s", a + "es") or (a.endswith("y") and b == a[:-1] + "ies")


def name_tokens(name) -> tuple:
    """"<Paper_Towels>" -> ("paper", "towels")"""
    text = strip_tags(name).replace("_", " ")
    return tuple("".join(c if c.isalnum() else " " for c in text).split())


def _tokens_match(pred: tuple, gt: tuple) -> bool:
    if not pred or not gt:
        return False
    if "".join(pred) == "".join(gt):  # "cutting board" == "cuttingboard"
        return True
    short, long = sorted((pred, gt), key=len)
    # whole words: "board" ~ "cutting board", but "pot" !~ "potato"
    return all(any(_same_word(w, v) for v in long) for w in short)


def names_match(pred_name, gt_name) -> bool:
    """Object, noun or verb names: equal up to case, spacing and plurals, or the words of one name all
    contained in the other. A GT name "X_or_Y" matches either alternative. Empty names match nothing."""
    pred = name_tokens(pred_name)
    return any(_tokens_match(pred, name_tokens(alt)) for alt in as_text(gt_name).split("_or_"))


def accuracy(correct: int, total: int) -> dict:
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


def evaluate_bbox(samples: list[dict], key: str, iou_threshold: float) -> dict:
    """Share of samples with a GT box whose predicted box overlaps one of them by IoU >= iou_threshold.
    A missing or malformed box is a miss. Also per GT class and its mean."""
    hits, class_hits = [], defaultdict(list)
    for s in samples:
        gt_boxes = s["gt_boxes"][key]
        if not gt_boxes:
            continue
        pred_box = s["pred_boxes"][key]
        hit = pred_box is not None and max(iou(pred_box, gt) for gt in gt_boxes) >= iou_threshold
        hits.append(hit)
        class_hits[s["gt_class"]].append(hit)
    per_class = {cls: sum(h) / len(h) for cls, h in sorted(class_hits.items())}
    return {
        **accuracy(sum(hits), len(hits)),
        "mean_class_accuracy": sum(per_class.values()) / len(per_class) if per_class else 0.0,
        "per_class_accuracy": per_class,
    }


class TaskEvaluator:
    """Every included GT sample is scored; a sample without a (parsable) prediction counts as wrong."""

    spec: TaskSpec
    bbox_results: dict     # answer key -> result name ("<name>_IoU_0.50")
    headline: tuple = ()   # (column label, "path:in:results") for the results table

    def include(self, ann: dict) -> bool:
        return True

    def gt_boxes(self, ann: dict, key: str) -> list:
        """GT boxes, as pixel (x1, y1, x2, y2), that the predicted box `key` is compared with."""
        raise NotImplementedError

    def gt_class(self, ann: dict) -> str:
        raise NotImplementedError

    def attribute_metrics(self, samples: list[dict]) -> dict:
        raise NotImplementedError

    def pair(self, gt: dict, preds: dict, bbox_format: str) -> list[dict]:
        samples = []
        for (vid, fid), ann in gt.items():
            if not self.include(ann):
                continue
            pred = preds.get((vid, fid)) or {}
            samples.append({
                "video_id": vid,
                "frame_id": fid,
                "gt": ann,
                "pred": pred,
                "answered": (vid, fid) in preds,
                "gt_boxes": {key: self.gt_boxes(ann, key) for key in self.bbox_results},
                "pred_boxes": {
                    key: parse_pred_box(pred.get(key), bbox_format, self.spec.frame_width,
                                        self.spec.frame_height)
                    for key in self.bbox_results
                },
                "gt_class": self.gt_class(ann),
            })
        return samples

    def metrics(self, samples: list[dict]) -> dict:
        results = {}
        for key, name in self.bbox_results.items():
            for thr in IOU_THRESHOLDS:
                results[f"{name}_IoU_{thr:.2f}"] = evaluate_bbox(samples, key, thr)
        results.update(self.attribute_metrics(samples))
        return results

    def evaluate(self, gt: dict, preds: dict, bbox_format: str) -> dict:
        samples = self.pair(gt, preds, bbox_format)
        by_session = defaultdict(list)
        for s in samples:
            by_session[s["video_id"]].append(s)
        return {
            "overall": self.metrics(samples),
            "per_session": {vid: {"num_samples": len(ss), **self.metrics(ss)} for vid, ss in sorted(by_session.items())},
            "num_gt": len(samples),
            "num_answered": sum(s["answered"] for s in samples),
        }
