from ..bbox import gt_to_xyxy
from ..tasks import HANDOVER
from .common import TaskEvaluator, accuracy, strip_tags
from .joint_attention import object_category_metrics

DEFAULT_TIME_TOLERANCE = 0.25  # seconds


def normalise_flow(raw: str) -> str:
    s = strip_tags(raw)
    if "left to right" in s or s == "ltr":
        return "ltr"
    if "right to left" in s or s == "rtl":
        return "rtl"
    return s


def normalise_initiator(raw: str) -> str:
    s = strip_tags(raw)
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    return s


def normalise_initiation_type(raw) -> frozenset:
    if isinstance(raw, list):
        return frozenset(t.strip().lower() for t in raw)
    s = strip_tags(raw)
    if "verbal" in s and "gestural" in s:
        return frozenset(["verbal", "gestural"])
    for t in ("implicit", "gestural", "verbal"):
        if t in s:
            return frozenset([t])
    return frozenset([s])


class HandoverEvaluator(TaskEvaluator):
    spec = HANDOVER
    bbox_results = {"object bounding box": "bbox"}
    headline = (
        ("IoU@0.5", "bbox_IoU_0.50:AP"),
        ("Del. Flow", "multiple_choice:delivering_flow:accuracy"),
        ("Initiator", "multiple_choice:initiator:accuracy"),
        ("Init. Type", "multiple_choice:initiation_type:accuracy"),
        ("Cat. (L1)", "object_category:level_1:accuracy"),
        ("TTH", "handover_offset:accuracy"),
    )

    def __init__(self, time_tolerance: float = DEFAULT_TIME_TOLERANCE):
        self.time_tolerance = time_tolerance

    def include(self, ann):
        return not ann.get("skip", False)

    def has_gt_box(self, ann):
        return bool(ann.get("bbox"))

    def gt_boxes(self, ann):
        return [gt_to_xyxy(ann["bbox"])] if ann.get("bbox") else []

    def gt_class(self, ann):
        return ann.get("object_category_level_1", "unknown")

    def offset_metrics(self, samples) -> dict:
        """Time to handover, correct if within the tolerance of end_time - start_time."""
        errors = []
        for s in samples:
            try:
                pred = float(s["pred"].get("handover happen offset"))
            except (TypeError, ValueError):
                continue
            errors.append(abs(pred - (s["gt"]["end_time"] - s["gt"]["start_time"])))
        return {
            **accuracy(sum(e <= self.time_tolerance for e in errors), len(errors)),
            "mean_abs_error": sum(errors) / len(errors) if errors else 0.0,
            "time_tolerance": self.time_tolerance,
        }

    def attribute_metrics(self, samples):
        choices = {
            "delivering_flow": ("delivering flow", normalise_flow),
            "initiator": ("initiator", normalise_initiator),
            "initiation_type": ("initiation type", normalise_initiation_type),
        }
        multiple_choice = {
            name: accuracy(sum(norm(s["gt"][name]) == norm(s["pred"].get(key, "")) for s in samples), len(samples))
            for name, (key, norm) in choices.items()
        }
        return {
            "multiple_choice": multiple_choice,
            "object_category": object_category_metrics(samples),
            "handover_offset": self.offset_metrics(samples),
        }
