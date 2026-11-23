import json
import shutil
import sys
from pathlib import Path

import pytest

from comind import evaluate
from comind.bbox import gt_to_xyxy, pred_to_xyxy
from comind.prompts import build_prompt
from comind.tasks import TASK_SPECS

from .conftest import ANNOTATIONS, TEST

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune"))
import infer  # noqa: E402
import prepare_data  # noqa: E402
from qwenvl.data import data_list  # noqa: E402


@pytest.fixture
def train_root(data_root, tmp_path):
    """The test fixture with its test recording as the train split."""
    root = tmp_path / "root"
    shutil.copytree(data_root, root)
    (root / "split" / "train.txt").write_text(TEST + "\n")
    return root


def decode(box, task):
    spec = TASK_SPECS[task]
    return pred_to_xyxy(box, spec.bbox_format, spec.frame_width, spec.frame_height)


def test_prepare_data(train_root):
    prepare_data.main(["--data_root", str(train_root)])
    out = train_root / "finetune"
    data = {task: json.loads((out / f"{task}_train.json").read_text()) for task in TASK_SPECS}
    assert {task: len(d) for task, d in data.items()} == {"joint_attention": 1, "handover": 1, "scoia": 1}

    for task, (sample,) in data.items():
        human, gpt = (turn["value"] for turn in sample["conversations"])
        n = TASK_SPECS[task].num_frames + 1
        assert len(sample["image"]) == n and human.startswith("<image>\n" * n) and human.count("<image>") == n
        assert all((out / p).resolve().exists() for p in sample["image"])
        assert sample["image"][-1].endswith(".jpg") and "benchmark" in sample["image"][-1]
        assert len(json.loads(gpt)) == 1 and gpt.startswith("[\n  {\n")

    ja = json.loads(data["joint_attention"][0]["conversations"][1]["value"])[0]
    assert ja["cue type"] == ["<gaze>", "<verbal>"] and ja["object category"] == "onion"
    gt = [gt_to_xyxy(b) for b in ANNOTATIONS["joint_attention"][TEST]["000100"]["bounding_boxes"]]
    for key, box in zip(("left object bounding box", "right object bounding box"), gt):
        assert decode(ja[key], "joint_attention") == pytest.approx(box, abs=1.5)
    assert ja["left object bounding box"][3] <= 500 <= ja["right object bounding box"][1]

    handover = json.loads(data["handover"][0]["conversations"][1]["value"])[0]
    assert handover == {"handover happen offset": 1, "delivering flow": "<left to right>", "initiator": "<right>",
                        "initiation type": "<verbal and gestural>", "object category": "salt",
                        "object bounding box": handover["object bounding box"]}
    assert decode(handover["object bounding box"], "handover") == pytest.approx((100, 200, 200, 300), abs=1.5)
    assert "salt please" in data["handover"][0]["conversations"][0]["value"]

    scoia = json.loads(data["scoia"][0]["conversations"][1]["value"])[0]
    assert scoia == {"cue type": "<verbal and gestural>", "action verb": "grab", "action noun": "package",
                     "object bounding box": [100, 100, 200, 200]}

    (config,) = data_list([str(out / "scoia_train.json") + "%50"])
    assert config["data_path"] == str(out) and config["sampling_rate"] == 0.5
    with pytest.raises(ValueError):
        data_list([str(out / "missing.json")])


def test_to_yxyx_1000():
    assert prepare_data.to_yxyx_1000([140.8, 70.4, 140.8, 70.4], 1408, 704) == [100, 100, 200, 200]
    assert prepare_data.to_yxyx_1000([281.6, 140.8, -140.8, -70.4], 1408, 704) == [100, 100, 200, 200]
    assert prepare_data.to_yxyx_1000([600, 0, 300, 704], 1408, 704, x_hi=500) == [0, 426, 1000, 500]


class FakeModel:
    calls = []

    def __init__(self, model, adapter=None, **kwargs):
        self.model, self.adapter = model, adapter

    def generate(self, prompt, frame, context_frames, seed=None):
        FakeModel.calls.append((prompt, len(context_frames)))
        return ('[{"cue type": "<verbal and gestural>", "action verb": "grab", "action noun": "package", '
                '"object bounding box": [100, 100, 200, 200]}]')


def test_infer_and_evaluate(data_root, tmp_path, monkeypatch):
    monkeypatch.setattr(infer, "FinetunedQwen", FakeModel)
    adapter = tmp_path / "qwen3_vl_32b_scoia" / "checkpoint-500"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3-VL-32B-Instruct"}))
    out = tmp_path / "run"
    infer.main(["--task", "scoia", "--adapter", str(adapter), "--data_root", str(data_root), "--output_dir", str(out)])
    assert FakeModel.calls == [(build_prompt("scoia", "hand me that"), 5)]
    config = json.loads((out / "run_config.json").read_text())
    assert config["model"] == "Qwen/Qwen3-VL-32B-Instruct" and config["bbox_format"] == "yxyx_1000"

    evaluate.main(["--task", "scoia", "--pred_dir", str(out), "--data_root", str(data_root)])
    res = json.loads((out / "evaluation_results.json").read_text())
    assert res["overall"]["bbox_IoU_0.50"]["accuracy"] == 1.0

    assert infer.run_name(str(adapter), None) == "qwen3_vl_32b_scoia_checkpoint_500"
    assert infer.run_name(str(adapter.parent), None) == "qwen3_vl_32b_scoia"
    assert infer.run_name(None, "Qwen/Qwen3-VL-8B-Instruct") == "qwen3_vl_8b_instruct_base"
