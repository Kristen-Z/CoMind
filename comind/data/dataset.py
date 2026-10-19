"""Dataset layout (as downloaded with comind_download.py, plus `python -m comind.prepare`):

    <root>/annotations/dataset_<task>_consolidated.json
    <root>/split/{train,test}.txt
    <root>/recordings/<video_id>/mp4s/*.mp4
    <root>/benchmark/<task>/frames/<video_id>/<frame_id>.jpg    reference frames
    <root>/benchmark/<task>/clips/<video_id>/<frame_id>.mp4     10 s before each reference frame
    <root>/benchmark/<task>/transcripts/<video_id>.json          speech in the same 10 s

Test annotations are held out: their entries are null.
"""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ..bbox import gt_to_xyxy
from ..tasks import get_task
from .video import sample_frames

ENV_VAR = "COMIND_DATA_ROOT"
SPLITS = ("train", "test", "all")


def resolve_root(root=None) -> Path:
    root = root or os.environ.get(ENV_VAR)
    if not root:
        raise ValueError(f"Pass --data_root or set ${ENV_VAR}.")
    return Path(root).expanduser()


def annotation_path(root, task: str) -> Path:
    return Path(root) / "annotations" / f"dataset_{task}_consolidated.json"


def benchmark_dir(root, task: str) -> Path:
    return Path(root) / "benchmark" / task


def load_split(root, split: str):
    """Video ids of a split, or None for "all"."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}', choose from {SPLITS}")
    if split == "all":
        return None
    lines = (Path(root) / "split" / f"{split}.txt").read_text().splitlines()
    return {line.strip() for line in lines if line.strip()}


def load_annotations(path, video_ids=None) -> dict:
    """{(video_id, frame_id): annotation or None}, in file order."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)["data"]
    return {
        (vid, fid): ann
        for vid, frames in data.items()
        if video_ids is None or vid in video_ids
        for fid, ann in frames.items()
    }


@dataclass
class Sample:
    task: str
    video_id: str
    frame_id: str
    annotation: dict | None  # None for held-out test samples
    frame_path: Path
    clip_path: Path
    transcript: str = field(default="", repr=False)

    @property
    def key(self) -> str:
        return f"{self.video_id}_{self.frame_id}"

    def load_frame(self) -> Image.Image:
        return Image.open(self.frame_path).convert("RGB")

    def load_context_frames(self, num_frames: int) -> list:
        return sample_frames(self.clip_path, num_frames)

    def gt_boxes(self) -> list:
        """GT boxes as pixel (x1, y1, x2, y2) on the reference frame."""
        ann = self.annotation or {}
        boxes = (ann.get("bounding_boxes") or []) if self.task == "joint_attention" else [ann.get("bbox")]
        return [gt_to_xyxy(b) for b in boxes if b and len(b) == 4]


class CoMindDataset(Sequence):
    """Samples of one task and split, sorted by (video_id, frame_id)."""

    def __init__(self, task: str, split: str = "test", root=None):
        get_task(task)
        self.task = task
        self.split = split
        self.root = resolve_root(root)
        self.annotations = load_annotations(annotation_path(self.root, task), load_split(self.root, split))
        self.keys = sorted(self.annotations)
        self.dir = benchmark_dir(self.root, task)
        self.transcripts = {p.stem: json.loads(p.read_text()) for p in (self.dir / "transcripts").glob("*.json")}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx) -> Sample:
        vid, fid = self.keys[idx]
        ann = self.annotations[(vid, fid)]
        return Sample(
            task=self.task,
            video_id=vid,
            frame_id=fid,
            annotation=ann,
            frame_path=self.dir / "frames" / vid / f"{fid}.jpg",
            clip_path=self.dir / "clips" / vid / f"{fid}.mp4",
            transcript=self.transcripts.get(vid, {}).get(fid, ""),
        )

    @property
    def video_ids(self) -> list:
        return sorted({vid for vid, _ in self.keys})

    def __repr__(self):
        return f"CoMindDataset({self.task!r}, split={self.split!r}, samples={len(self)})"
