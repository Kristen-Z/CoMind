import json

import cv2
import numpy as np
import pytest
from PIL import Image

TRAIN, TEST = "11111111-train", "22222222-test"

ANNOTATIONS = {
    "joint_attention": {
        TEST: {
            "000100": {"bounding_boxes": [[100, 100, 100, 100], [800, 100, 100, 100]],
                       "cue_types": ["gaze", "communication"], "object_category_level_1": "onion",
                       "object_category_level_2": "vegetable", "object_category_level_3": "vegetable_or_fruit",
                       "transcript_10s": "take the onion", "start_time": 3.3, "end_time": 4.0},
            "000200": {"bounding_boxes": [[300, 300, 50, 50]], "cue_types": ["action"],
                       "object_category_level_1": "knife", "transcript_10s": ""},
        },
        TRAIN: {"000050": {"bounding_boxes": [[0, 0, 10, 10]], "cue_types": ["gaze"],
                           "object_category_level_1": "pan"}},
    },
    "handover": {
        TEST: {
            "000300": {"bbox": [100, 200, 100, 100], "delivering_flow": "ltr", "initiator": "right",
                       "initiation_type": ["verbal", "gestural"], "skip": False, "start_time": 10.0,
                       "end_time": 11.0, "object_category_level_1": "salt", "transcript_10s": "salt please"},
            "000400": {"bbox": [500, 200, 50, 50], "delivering_flow": "rtl", "initiator": "left",
                       "initiation_type": ["implicit"], "skip": True, "start_time": 0.0, "end_time": 2.0,
                       "object_category_level_1": "plate"},
        },
    },
    "scoia": {
        TEST: {"000500": {"bbox": [140.8, 140.8, 140.8, 140.8], "annot_type": "AG", "verbs": ["grab"],
                          "nouns": ["package"], "nouns_level2": ["container"],
                          "nouns_level3": ["container_packaging"], "transcript_10s": "hand me that"}},
    },
}

FRAME_SIZE = {"joint_attention": (1408, 704), "handover": (1408, 704), "scoia": (1408, 1408)}


def _write_clip(path, n_frames=12, size=(176, 88)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), i * 20, np.uint8))
    writer.release()


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("comind_data")
    (root / "split").mkdir()
    (root / "split" / "train.txt").write_text(TRAIN + "\n")
    (root / "split" / "test.txt").write_text(TEST + "\n")
    for task, data in ANNOTATIONS.items():
        (root / "annotations").mkdir(exist_ok=True)
        (root / "annotations" / f"dataset_{task}_consolidated.json").write_text(
            json.dumps({"dataset_version": "test", "data": data}))
        for vid, frames in data.items():
            tdir = root / "benchmark" / task / "transcripts"
            tdir.mkdir(parents=True, exist_ok=True)
            (tdir / f"{vid}.json").write_text(json.dumps({f: a.get("transcript_10s", "") for f, a in frames.items()}))
            fdir = root / "benchmark" / task / "frames" / vid
            cdir = root / "benchmark" / task / "clips" / vid
            fdir.mkdir(parents=True)
            cdir.mkdir(parents=True)
            for fid in frames:
                Image.new("RGB", FRAME_SIZE[task], "gray").save(fdir / f"{fid}.jpg")
                _write_clip(cdir / f"{fid}.mp4")
    return root
