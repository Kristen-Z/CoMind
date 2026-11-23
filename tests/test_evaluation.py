import pytest

from comind.evaluation import get_evaluator, load_predictions, lookup
from comind.evaluation.common import names_match

from .conftest import ANNOTATIONS, TEST


def gt_of(task):
    return {(vid, fid): ann for vid, frames in ANNOTATIONS[task].items() for fid, ann in frames.items()}


@pytest.mark.parametrize("pred,gt,ok", [
    ("onion", "onions", True),
    ("Paper Towel", "paper_towel", True),
    ("fruit", "vegetable_or_fruit", True),
    ("meat", "vegetable_or_fruit", False),
    ("", "onion", False),
    (None, "onion", False),
    (" ", "vegetable_or_fruit", False),
    ("onion", "", False),
    ("<Knife>", "knife", True),
    ("board", "cutting board", True),
    ("cutting board", "cuttingboard", True),
    ("glasses", "glass", True),
    ("cherries", "cherry", True),
    ("boxes", "box", True),
    ("pot", "potato", False),  # whole words only
    ("cup", "cupboard", False),
    ("pan", "pancake", False),
])
def test_names_match(pred, gt, ok):
    assert names_match(pred, gt) is ok


def test_load_predictions(tmp_path):
    (tmp_path / f"{TEST}_000100.txt").write_text('Box [1, 2] first.\n[{"object category": "onion"}]')
    (tmp_path / f"{TEST}_000200.txt").write_text('[{"object category": "knife"}]')
    preds, unparsable = load_predictions(tmp_path)
    assert len(preds) == 2 and not unparsable
    preds, unparsable = load_predictions(tmp_path, strict=True)
    assert list(preds) == [(TEST, "000200")] and unparsable == [f"{TEST}_000100.txt"]


def test_joint_attention():
    gt = gt_of("joint_attention")
    preds = {
        # left box matches the left GT box; the "right" box is the left GT box, i.e. in the wrong view
        (TEST, "000100"): {"left object bounding box": [100, 100, 200, 200],
                           "right object bounding box": [100, 100, 200, 200],
                           "cue type": ["<verbal>", "<gaze>"], "object category": "onion"},
    }
    res = get_evaluator("joint_attention").evaluate(gt, preds, "xyxy_pixels")
    o = res["overall"]
    assert res["num_gt"] == 3 and res["num_answered"] == 1
    # unanswered samples count as wrong: 3 samples with a left GT box, 1 with a right one
    assert (o["left_bbox_IoU_0.50"]["correct"], o["left_bbox_IoU_0.50"]["total"]) == (1, 3)
    assert (o["right_bbox_IoU_0.50"]["correct"], o["right_bbox_IoU_0.50"]["total"]) == (0, 1)
    assert (o["cue_type"]["correct"], o["cue_type"]["total"]) == (1, 3)  # communication == verbal
    assert o["object_category"]["level_3"]["accuracy"] == 0.0
    assert sorted(res["per_session"]) == sorted({TEST, "11111111-train"})


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
    assert res["num_gt"] == 1 and res["num_answered"] == 1
    assert o["bbox_IoU_0.50"]["accuracy"] == 1.0
    assert all(v["accuracy"] == 1.0 for v in o["multiple_choice"].values())
    assert o["handover_offset"]["correct"] == 1  # |1.2 - 1.0| <= 0.25
    assert get_evaluator("handover", time_tolerance=0.1).evaluate(gt, preds, "xyxy_pixels")[
        "overall"]["handover_offset"]["correct"] == 0

    preds[(TEST, "000300")].update({"handover happen offset": "soon", "initiation type": ["<verbal>", "<gestural>"]})
    o = get_evaluator("handover").evaluate(gt, preds, "xyxy_pixels")["overall"]
    assert o["handover_offset"]["total"] == 1 and o["handover_offset"]["accuracy"] == 0.0  # non-numeric is wrong
    assert o["handover_offset"]["mean_abs_error"] is None
    assert o["multiple_choice"]["initiation_type"]["accuracy"] == 1.0  # a list with tags


def test_scoia():
    gt = gt_of("scoia")
    preds = {(TEST, "000500"): {"object bounding box": [100, 100, 200, 200], "cue type": "<verbal and gestural>",
                                "action verb": "grab", "action noun": "packages"}}
    o = get_evaluator("scoia").evaluate(gt, preds, "yxyx_1000")["overall"]
    assert o["bbox_IoU_0.50"]["accuracy"] == 1.0  # 100/1000 * 1408 = 140.8
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
    assert o["action_noun"]["level_1"]["accuracy"] == 0.0  # a missing noun is not correct

    gt = {("v", "1"): {"bounding_boxes": None, "cue_types": None, "object_category_level_1": "pan"}}
    res = get_evaluator("joint_attention").evaluate(gt, {("v", "1"): {"cue type": []}}, "xyxy_pixels")
    assert res["overall"]["cue_type"]["total"] == 0  # no GT cue types: not scored
    assert res["overall"]["left_bbox_IoU_0.50"]["total"] == 0


def test_unanswered_samples_are_wrong():
    for task in ("joint_attention", "handover", "scoia"):
        evaluator = get_evaluator(task)
        o = evaluator.evaluate(gt_of(task), {}, "xyxy_pixels")["overall"]
        for _, path in evaluator.headline:
            metric = lookup(o, path.rpartition(":")[0])
            assert metric["total"] > 0 and metric["accuracy"] == 0.0, (task, path)
