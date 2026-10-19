from ..bbox import gt_to_xyxy
from ..tasks import SCOIA
from .common import TaskEvaluator, accuracy, category_matches, strip_tags

CUE_TYPE_MAP = {"A": "verbal", "G": "gestural", "AG": "verbal and gestural", "O": "other"}

NOUN_LEVELS = {"level_1": "nouns", "level_2": "nouns_level2", "level_3": "nouns_level3"}


def verb_matches(pred_verb: str, gt_verbs: list) -> bool:
    pred = strip_tags(pred_verb)
    return bool(pred) and any(pred == g or pred in g or g in pred for g in map(strip_tags, gt_verbs))


class ScoiaEvaluator(TaskEvaluator):
    spec = SCOIA
    bbox_results = {"object bounding box": "bbox"}
    headline = (
        ("IoU@0.5", "bbox_IoU_0.50:AP"),
        ("Cue Type", "cue_type:accuracy"),
        ("Act. Verb", "action_verb:accuracy"),
        ("Act. Noun (L1)", "action_noun:level_1:accuracy"),
    )

    def has_gt_box(self, ann):
        return bool(ann.get("bbox"))

    def gt_boxes(self, ann):
        return [gt_to_xyxy(ann["bbox"])] if ann.get("bbox") else []

    def gt_class(self, ann):
        nouns = ann.get("nouns", [])
        return nouns[0] if nouns else "unknown"

    def attribute_metrics(self, samples):
        labelled = [s for s in samples if s["gt"].get("annot_type")]  # a few annotations have no cue type
        cue_correct = sum(
            CUE_TYPE_MAP.get(s["gt"]["annot_type"].strip(), s["gt"]["annot_type"].strip().lower())
            == strip_tags(s["pred"].get("cue type", ""))
            for s in labelled
        )
        verb_correct = sum(verb_matches(s["pred"].get("action verb", ""), s["gt"].get("verbs", [])) for s in samples)

        nouns = {}
        for level, field in NOUN_LEVELS.items():
            correct = total = 0
            for s in samples:
                gt_nouns = s["gt"].get(field, [])
                if not gt_nouns:
                    continue
                total += 1
                pred_noun = s["pred"].get("action noun", "")
                correct += any(category_matches(pred_noun, g) for g in gt_nouns)
            nouns[level] = accuracy(correct, total)

        return {
            "cue_type": accuracy(cue_correct, len(labelled)),
            "action_verb": accuracy(verb_correct, len(samples)),
            "action_noun": nouns,
        }
