import pytest

from comind.bbox import canonical_bbox_format, gt_to_xyxy, iou, parse_pred_box, pred_to_xyxy
from comind.parsing import parse_response
from comind.prompts import build_prompt, prompt_path
from comind.tasks import TASK_SPECS


def test_bbox_formats():
    expected = (140.8, 70.4, 281.6, 140.8)
    assert pred_to_xyxy([100, 100, 200, 200], "yxyx_1000", 1408, 704) == pytest.approx(expected)
    assert pred_to_xyxy([100, 100, 200, 200], "xyxy_1000", 1408, 704) == pytest.approx(expected)
    assert pred_to_xyxy(list(expected), "xyxy_pixels", 1408, 704) == pytest.approx(expected)
    assert pred_to_xyxy([0, 500, 1000, 1000], "yxyx_1000", 1000, 1000) == pytest.approx((500, 0, 1000, 1000))


@pytest.mark.parametrize("box", [[10, 10, 5, 20], [10, 10, 20, 10], [1, 2, 3]])
def test_invalid_boxes(box):
    assert pred_to_xyxy(box, "xyxy_pixels", 100, 100) is None


def test_bbox_helpers():
    assert parse_pred_box(["1", "2", "3", "4"], "xyxy_pixels", 10, 10) == (1, 2, 3, 4)
    assert parse_pred_box(["a", 2, 3, 4], "xyxy_pixels", 10, 10) is None
    assert canonical_bbox_format("xyxy_pixels") == "xyxy_pixels"
    with pytest.raises(ValueError):
        canonical_bbox_format("xywh")
    assert gt_to_xyxy([1, 2, 3, 4]) == (1, 2, 4, 6)
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_parse_response():
    assert parse_response('```json\n[{"cue type": "<gaze>"}]\n```') == {"cue type": "<gaze>"}
    text = 'Roughly [280, 420, 420, 540]. Answer:\n[{"object bounding box": [1, 2, 3, 4]}]'
    assert parse_response(text) == {"object bounding box": [1, 2, 3, 4]}
    for bad in ["", "no json", "[1, 2, 3]", "[]"]:
        assert parse_response(bad) is None


def test_prompts():
    for task in TASK_SPECS:
        assert prompt_path(task).exists()
        prompt = build_prompt(task, "hello there")
        assert "hello there" in prompt and "{transcript}" not in prompt
