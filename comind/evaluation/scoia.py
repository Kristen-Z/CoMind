from ..bbox import gt_to_xyxy
from ..tasks import SCOIA
from .common import TaskEvaluator, accuracy, names_match, strip_tags

CUE_TYPE_MAP = {"A": "verbal", "G": "gestural", "AG": "verbal and gestural", "O": "other"}

NOUN_LEVELS = {"level_1": "nouns", "level_2": "nouns_level2", "level_3": "nouns_level3"}


def verb_matches(pred_verb: str, gt_verbs: list) -> bool:
    return any(names_match(pred_verb, g) for g in gt_verbs)


class ScoiaEvaluator(TaskEvaluator):
    spec = SCOIA
    bbox_results = {"object bounding box": "bbox"}
    headline = (
        ("IoU@0.5", "bbox_IoU_0.50:accuracy"),
        ("Cue Type", "cue_type:accuracy"),
        ("Act. Verb", "action_verb:accuracy"),
        ("Act. Noun (L1)", "action_noun:level_1:accuracy"),
    )

    def gt_boxes(self, ann, key):
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
        with_verb = [s for s in samples if s["gt"].get("verbs")]
        verb_correct = sum(verb_matches(s["pred"].get("action verb", ""), s["gt"]["verbs"]) for s in with_verb)

        nouns = {}
        for level, field in NOUN_LEVELS.items():
            correct = total = 0
            for s in samples:
                gt_nouns = s["gt"].get(field, [])
                if not gt_nouns:
                    continue
                total += 1
                pred_noun = s["pred"].get("action noun", "")
                correct += any(names_match(pred_noun, g) for g in gt_nouns)
            nouns[level] = accuracy(correct, total)

        return {
            "cue_type": accuracy(cue_correct, len(labelled)),
            "action_verb": accuracy(verb_correct, len(with_verb)),
            "action_noun": nouns,
        }
