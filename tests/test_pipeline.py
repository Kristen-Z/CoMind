import json

import pytest

from comind import evaluate, infer
from comind.data import CoMindDataset, sample_frames
from comind.data.video import count_frames, uniform_indices

from .conftest import TEST


def test_dataset(data_root, monkeypatch):
    ds = CoMindDataset("joint_attention", split="test", root=data_root)
    assert len(ds) == 2 and ds.video_ids == [TEST]
    s = ds[0]
    assert s.key == f"{TEST}_000100" and s.transcript == "take the onion"
    assert s.gt_boxes() == [(100, 100, 200, 200), (800, 100, 900, 200)]
    assert s.frame_path.exists() and s.clip_path.exists()
    assert s.load_frame().size == (1408, 704)
    assert len(CoMindDataset("joint_attention", split="all", root=data_root)) == 3
    monkeypatch.setenv("COMIND_DATA_ROOT", str(data_root))
    assert len(CoMindDataset("scoia")) == 1


def test_frame_sampling(data_root):
    clip = data_root / "benchmark" / "handover" / "clips" / TEST / "000300.mp4"
    assert uniform_indices(12, 5) == [0, 2, 5, 8, 11]
    assert uniform_indices(3, 5) == [0, 1, 2]
    assert len(sample_frames(clip, 5)) == 5
    assert sample_frames(clip, 0) == []


class FakeBackend:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, frame, context_frames, seed=None):
        self.calls.append((prompt, len(context_frames)))
        # GT box (100, 200, 200, 300) px on the 1408 x 704 frame, as [ymin, xmin, ymax, xmax] in [0, 1000]
        return ('[{"object bounding box": [284, 71, 426, 142], "delivering flow": "<left to right>", '
                '"initiator": "<right>", "initiation type": "<verbal and gestural>", '
                '"object category": "salt", "handover happen offset": 1.0}]')


def test_infer_and_evaluate(data_root, tmp_path, monkeypatch):
    fake = FakeBackend()
    monkeypatch.setattr(infer, "build_backend", lambda args: fake)
    out = tmp_path / "run"
    argv = ["--task", "handover", "--backend", "openai", "--data_root", str(data_root),
            "--output_dir", str(out), "--save_vis"]
    infer.main(argv)
    assert len(fake.calls) == 2
    assert "salt please" in fake.calls[0][0] and fake.calls[0][1] == 5
    assert (tmp_path / "run_vis" / f"{TEST}_000300.jpg").exists()

    infer.main(argv)  # cached
    assert len(fake.calls) == 2
    with pytest.raises(SystemExit):
        infer.main([*argv, "--model", "other"])

    evaluate.main(["--task", "handover", "--pred_dir", str(out), "--data_root", str(data_root)])
    res = json.loads((out / "evaluation_results.json").read_text())
    assert res["config"]["bbox_format"] == "yxyx_1000"
    assert res["overall"]["bbox_IoU_0.50"]["accuracy"] == 1.0
    assert res["overall"]["handover_offset"]["accuracy"] == 1.0


def test_prepare(data_root, tmp_path):
    import shutil

    import cv2
    import numpy as np

    from comind import prepare

    root = tmp_path / "root"
    shutil.copytree(data_root / "annotations", root / "annotations")
    shutil.copytree(data_root / "split", root / "split")
    rec = root / "recordings" / TEST
    (rec / "mp4s").mkdir(parents=True)
    (rec / "transcripts").mkdir()
    writer = cv2.VideoWriter(str(rec / "mp4s" / "helper_trimmed_sync.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                             (64, 64))
    for i in range(520):
        writer.write(np.full((64, 64, 3), 200 if i == 500 else 20, np.uint8))
    writer.release()
    words = [{"word": "hand", "start": 10.0, "end": 10.3}, {"word": "late", "start": 17.0, "end": 17.2}]
    (rec / "transcripts" / "helper_trimmed_sync_transcript.json").write_text(
        json.dumps({"segments": [{"start": 10.0, "end": 17.2, "words": words}]}))

    prepare.main(["--data_root", str(root), "--tasks", "scoia", "--split", "test"])
    sample = CoMindDataset("scoia", root=root)[0]  # frame 500
    assert abs(np.asarray(sample.load_frame()).mean() - 200) < 5  # reference frame is frame 500
    assert count_frames(sample.clip_path) == 300  # frames 200..499
    assert np.asarray(sample.load_context_frames(5)[-1]).mean() < 50
    assert sample.transcript == "hand"  # 10 s before frame 500 (16.7 s)
    assert prepare.transcript_window(json.loads(
        (rec / "transcripts" / "helper_trimmed_sync_transcript.json").read_text())["segments"], 16.0) == "hand"


def test_prepare_without_side_by_side_video(data_root, tmp_path):
    """A recording with only the leader and helper videos: the two views are placed side by side."""
    import shutil

    import cv2
    import numpy as np

    from comind import prepare

    root = tmp_path / "root"
    shutil.copytree(data_root / "annotations", root / "annotations")
    shutil.copytree(data_root / "split", root / "split")
    mp4s = root / "recordings" / TEST / "mp4s"
    mp4s.mkdir(parents=True)
    for name, value in (("leader_trimmed_sync.mp4", 200), ("helper_trimmed_sync.mp4", 40)):
        writer = cv2.VideoWriter(str(mp4s / name), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
        for i in range(120):
            writer.write(np.full((64, 64, 3), value if i == 100 else 0, np.uint8))
        writer.release()

    prepare.main(["--data_root", str(root), "--tasks", "joint_attention", "--split", "test"])
    sample = CoMindDataset("joint_attention", root=root)[0]  # frame 100
    frame = np.asarray(sample.load_frame())
    assert frame.shape[:2] == (704, 1408)
    assert abs(frame[:, :600].mean() - 200) < 5 and abs(frame[:, 808:].mean() - 40) < 5  # leader left, helper right
    assert count_frames(sample.clip_path) == 100  # frames 0..99
    assert not CoMindDataset("joint_attention", root=root)[1].frame_path.exists()  # frame 200: past the video

    (mp4s / "leader_trimmed_sync.mp4").unlink()  # one view alone is not enough
    assert prepare.source_videos(mp4s, "leader_helper_trimmed_sync.mp4") is None


@pytest.mark.parametrize("backend,task,expected", [
    ("openai", "handover", "yxyx_1000"), ("claude", "scoia", "yxyx_1000"),
    ("gemini", "joint_attention", "yxyx_1000"), ("gemini", "handover", "yxyx_1000"),
    ("qwen", "handover", "xyxy_1000"), ("qwen", "scoia", "xyxy_1000"),
])
def test_bbox_format_per_backend(data_root, tmp_path, monkeypatch, backend, task, expected):
    monkeypatch.setattr(infer, "build_backend", lambda args: FakeBackend())
    infer.main(["--task", task, "--backend", backend, "--data_root", str(data_root), "--output_dir", str(tmp_path)])
    assert json.loads((tmp_path / "run_config.json").read_text())["bbox_format"] == expected
