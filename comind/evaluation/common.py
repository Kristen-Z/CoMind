from collections import defaultdict
from pathlib import Path

from ..bbox import iou, parse_pred_box
from ..parsing import parse_response
from ..tasks import TaskSpec

IOU_THRESHOLDS = (0.25, 0.50)


def load_predictions(pred_dir) -> tuple[dict, list[str]]:
    """({(video_id, frame_id): answer}, [unparsable files]) for all <video_id>_<frame_id>.txt."""
    preds, unparsable = {}, []
    for path in sorted(Path(pred_dir).glob("*.txt")):
        video_id, sep, frame_id = path.stem.rpartition("_")
        if not sep:
            continue
        answer = parse_response(path.read_text(encoding="utf-8"))
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


def normalise_category(name) -> str:
    return as_text(name).strip().lower().replace(" ", "_")


def category_matches(pred_name: str, gt_name: str) -> bool:
    """Exact, substring (either direction) or any alternative of an "X_or_Y" GT name."""
    pred, gt = normalise_category(pred_name), normalise_category(gt_name)
    if pred == gt:
        return True
    if "_or_" in gt:
        return any(alt == pred or alt in pred or pred in alt for alt in (a.strip() for a in gt.split("_or_")))
    return pred in gt or gt in pred


def accuracy(correct: int, total: int) -> dict:
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


def compute_ap(is_tp: list[bool]) -> float:
    """11-point interpolated AP; all predictions have score 1 (VLMs give no confidence)."""
    if not is_tp:
        return 0.0
    tp = fp = 0
    precisions, recalls = [], []
    for hit in is_tp:
        tp += hit
        fp += not hit
        precisions.append(tp / (tp + fp))
        recalls.append(tp / len(is_tp))
    return sum(
        max((p for p, r in zip(precisions, recalls) if r >= t / 10.0), default=0.0) for t in range(11)
    ) / 11.0


def compute_prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_bbox(samples: list[dict], side: str, iou_threshold: float, n_unpaired: int) -> dict:
    """A box is a TP if it overlaps any GT box of the sample by >= iou_threshold. A wrong box
    counts as FP and FN; a missing box, or a GT sample without prediction, as FN."""
    hits, class_hits = [], defaultdict(list)
    tp = fp = fn = 0
    for s in samples:
        if not s["gt_boxes"]:
            continue
        pred_box = s["pred_boxes"][side]
        if pred_box is None:
            hit = False
            fn += 1
        else:
            hit = max(iou(pred_box, gt) for gt in s["gt_boxes"]) >= iou_threshold
            if hit:
                tp += 1
            else:
                fp += 1
                fn += 1
        hits.append(hit)
        class_hits[s["gt_class"]].append(hit)
    fn += n_unpaired

    per_class = {cls: compute_ap(h) for cls, h in sorted(class_hits.items())}
    return {
        "AP": compute_ap(hits),
        "mAP": sum(per_class.values()) / len(per_class) if per_class else 0.0,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        **compute_prf(tp, fp, fn),
        "per_class_AP": per_class,
        "num_evaluated": len(hits),
    }


class TaskEvaluator:
    spec: TaskSpec
    bbox_results: dict     # answer key -> result name ("<name>_IoU_0.50")
    headline: tuple = ()   # (column label, "path:in:results") for the results table

    def include(self, ann: dict) -> bool:
        return True

    def has_gt_box(self, ann: dict) -> bool:
        raise NotImplementedError

    def gt_boxes(self, ann: dict) -> list:
        raise NotImplementedError

    def gt_class(self, ann: dict) -> str:
        raise NotImplementedError

    def attribute_metrics(self, samples: list[dict]) -> dict:
        raise NotImplementedError

    def pair(self, gt: dict, preds: dict, bbox_format: str) -> list[dict]:
        samples = []
        for (vid, fid), ann in gt.items():
            if not self.include(ann) or (vid, fid) not in preds:
                continue
            pred = preds[(vid, fid)]
            samples.append({
                "video_id": vid,
                "frame_id": fid,
                "gt": ann,
                "pred": pred,
                "gt_boxes": self.gt_boxes(ann),
                "pred_boxes": {
                    key: parse_pred_box(pred.get(key), bbox_format, self.spec.frame_width,
                                        self.spec.frame_height)
                    for key in self.bbox_results
                },
                "gt_class": self.gt_class(ann),
            })
        return samples

    def metrics(self, samples: list[dict], gt: dict) -> dict:
        paired = {(s["video_id"], s["frame_id"]) for s in samples}
        n_unpaired = sum(
            1 for key, ann in gt.items() if self.include(ann) and self.has_gt_box(ann) and key not in paired
        )
        results = {}
        for key, name in self.bbox_results.items():
            for thr in IOU_THRESHOLDS:
                results[f"{name}_IoU_{thr:.2f}"] = evaluate_bbox(samples, key, thr, n_unpaired)
        results.update(self.attribute_metrics(samples))
        return results

    def evaluate(self, gt: dict, preds: dict, bbox_format: str) -> dict:
        samples = self.pair(gt, preds, bbox_format)
        by_session = defaultdict(list)
        for s in samples:
            by_session[s["video_id"]].append(s)
        per_session = {}
        for vid in sorted(by_session):
            session_gt = {k: v for k, v in gt.items() if k[0] == vid}
            per_session[vid] = {"num_samples": len(by_session[vid]),
                                **self.metrics(by_session[vid], session_gt)}
        return {
            "overall": self.metrics(samples, gt),
            "per_session": per_session,
            "num_gt": sum(1 for ann in gt.values() if self.include(ann)),
            "num_evaluated": len(samples),
        }
