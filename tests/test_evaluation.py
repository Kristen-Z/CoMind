import pytest

from comind.evaluation import get_evaluator
from comind.evaluation.common import category_matches, compute_ap

from .conftest import ANNOTATIONS, TEST


def gt_of(task):
    return {(vid, fid): ann for vid, frames in ANNOTATIONS[task].items() for fid, ann in frames.items()}


def test_compute_ap():
    assert compute_ap([]) == 0.0
    assert compute_ap([True, True]) == 1.0
    assert compute_ap([False, False]) == 0.0
    # precision 1.0 up to recall 0.5, then 0.5 at recall 0.5 -> 6 of 11 points at 1.0
    assert compute_ap([True, False]) == pytest.approx(6 / 11)


@pytest.mark.parametrize("pred,gt,ok", [
    ("onion", "onions", True),
    ("Paper Towel", "paper_towel", True),
    ("fruit", "vegetable_or_fruit", True),
    ("meat", "vegetable_or_fruit", False),
])
def test_category_matches(pred, gt, ok):
    assert category_matches(pred, gt) is ok


def test_joint_attention():
    gt = gt_of("joint_attention")
    preds = {
        # left box matches GT box 1 exactly; right box is in the wrong place
        (TEST, "000100"): {"left object bounding box": [100, 100, 200, 200],
                           "right object bounding box": [0, 600, 10, 700],
                           "cue type": ["<verbal>", "<gaze>"], "object category": "onion"},
    }
    res = get_evaluator("joint_attention").evaluate(gt, preds, "xyxy_pixels")
    o = res["overall"]
    assert o["left_bbox_IoU_0.50"]["TP"] == 1 and o["left_bbox_IoU_0.50"]["AP"] == 1.0
    assert o["right_bbox_IoU_0.50"]["TP"] == 0 and o["right_bbox_IoU_0.50"]["FP"] == 1
    # FN: the unmatched right box + two GT samples with boxes but no prediction
    assert o["right_bbox_IoU_0.50"]["FN"] == 3
    assert o["cue_type"]["accuracy"] == 1.0  # communication == verbal
    assert o["object_category"]["level_3"]["accuracy"] == 0.0
    assert list(res["per_session"]) == [TEST]


def test_handover_skips_flagged_entries():
    gt = gt_of("handover")
    preds = {
        (TEST, "000300"): {"object bounding box": [100, 200, 200, 300], "delivering flow": "<left to right>",
                           "initiator": "<right person>", "initiation type": "<verbal and gestural>",
                           "object category": "salt", "handover happen offset": 1.2},
        (TEST, "000400"): {"object bounding box": [0, 0, 1, 1]},
    }
    res = get_evaluator("handover").evaluate(gt, preds, "xyxy_pixels")
    o = res["overall"]
    assert res["num_evaluated"] == 1
    assert o["bbox_IoU_0.50"]["TP"] == 1
    assert all(v["accuracy"] == 1.0 for v in o["multiple_choice"].values())
    assert o["handover_offset"]["correct"] == 1  # |1.2 - 1.0| <= 0.25
    assert get_evaluator("handover", time_tolerance=0.1).evaluate(gt, preds, "xyxy_pixels")[
        "overall"]["handover_offset"]["correct"] == 0


def test_scoia():
    gt = gt_of("scoia")
    preds = {(TEST, "000500"): {"object bounding box": [100, 100, 200, 200], "cue type": "<verbal and gestural>",
                                "action verb": "grab", "action noun": "packages"}}
    o = get_evaluator("scoia").evaluate(gt, preds, "yxyx_1000")["overall"]
    assert o["bbox_IoU_0.50"]["TP"] == 1  # 100/1000 * 1408 = 140.8
    assert o["cue_type"]["accuracy"] == 1.0
    assert o["action_verb"]["accuracy"] == 1.0
    assert o["action_noun"]["level_1"]["accuracy"] == 1.0


def test_missing_gt_fields():
    gt = {("v", "1"): {"bbox": [0, 0, 10, 10], "annot_type": None, "verbs": ["cut"], "nouns": ["onion"]},
          ("v", "2"): {"bbox": [0, 0, 10, 10], "annot_type": "A", "verbs": ["cut"], "nouns": ["onion"]}}
    preds = {k: {"cue type": "<verbal>", "action verb": ["cut"], "action noun": None} for k in gt}
    o = get_evaluator("scoia").evaluate(gt, preds, "xyxy_pixels")["overall"]
    assert o["cue_type"]["total"] == 1 and o["cue_type"]["accuracy"] == 1.0
    assert o["action_verb"]["accuracy"] == 1.0

    gt = {("v", "1"): {"bounding_boxes": None, "cue_types": None, "object_category_level_1": "pan"}}
    res = get_evaluator("joint_attention").evaluate(gt, {("v", "1"): {"cue type": []}}, "xyxy_pixels")
    assert res["overall"]["cue_type"]["accuracy"] == 1.0
