from ..bbox import gt_to_xyxy
from ..tasks import JOINT_ATTENTION
from .common import TaskEvaluator, accuracy, category_matches, strip_tags

GT_CUE_LABELS = {"action": "action", "communication": "verbal", "gaze": "gaze", "gesture": "gestural"}

CATEGORY_LEVELS = {
    "level_1": "object_category_level_1",
    "level_2": "object_category_level_2",
    "level_3": "object_category_level_3",
}


def cue_set(raw, is_gt: bool) -> frozenset:
    if not raw:
        return frozenset()
    if isinstance(raw, str):
        raw = [raw]
    labels = (strip_tags(c) for c in raw)
    return frozenset(GT_CUE_LABELS.get(c, c) for c in labels) if is_gt else frozenset(labels)


def object_category_metrics(samples, levels=CATEGORY_LEVELS, pred_key="object category") -> dict:
    results = {}
    for level, field in levels.items():
        correct = total = 0
        for s in samples:
            gt_cat = s["gt"].get(field, "")
            if not gt_cat:
                continue
            total += 1
            correct += category_matches(s["pred"].get(pred_key, ""), gt_cat)
        results[level] = accuracy(correct, total)
    return results


class JointAttentionEvaluator(TaskEvaluator):
    spec = JOINT_ATTENTION
    bbox_results = {"left object bounding box": "left_bbox", "right object bounding box": "right_bbox"}
    headline = (
        ("L-IoU@0.5", "left_bbox_IoU_0.50:AP"),
        ("R-IoU@0.5", "right_bbox_IoU_0.50:AP"),
        ("Cue Type", "cue_type:accuracy"),
        ("Cat. (L1)", "object_category:level_1:accuracy"),
    )

    def has_gt_box(self, ann):
        return bool(ann.get("bounding_boxes"))

    def gt_boxes(self, ann):
        return [gt_to_xyxy(b) for b in ann.get("bounding_boxes") or [] if b and len(b) == 4]

    def gt_class(self, ann):
        return ann.get("object_category_level_1", "unknown")

    def attribute_metrics(self, samples):
        cue_correct = sum(
            cue_set(s["gt"].get("cue_types", []), True) == cue_set(s["pred"].get("cue type", []), False)
            for s in samples
        )
        return {
            "cue_type": accuracy(cue_correct, len(samples)),
            "object_category": object_category_metrics(samples),
        }
