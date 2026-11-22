"""Cut the benchmark inputs from the downloaded recordings.

    python -m comind.prepare --data_root data/comind --split test

For every annotated moment this writes the reference frame, the 10 s clip leading up to it
and the speech transcript of those 10 s into <data_root>/benchmark/<task>/. Recordings that
have not been downloaded are skipped. Where a recording has no side-by-side video, the leader
and helper views are read in step and placed next to each other.
"""

import argparse
import json
import logging
import os
from collections import defaultdict, deque

import cv2
from PIL import Image

from . import TASKS
from .data.dataset import annotation_path, benchmark_dir, load_annotations, load_split, resolve_root
from .tasks import get_task

log = logging.getLogger("comind.prepare")

FPS = 30
CONTEXT_FRAMES = 10 * FPS
TRANSCRIPT_FILE = "transcripts/helper_trimmed_sync_transcript.json"
# The views a side-by-side video shows, left to right, for recordings that come without it.
SIDE_BY_SIDE_VIEWS = {"leader_helper_trimmed_sync.mp4": ("leader_trimmed_sync.mp4", "helper_trimmed_sync.mp4")}


def transcript_window(segments: list, end_time: float) -> str:
    """Words spoken in [end_time - 10 s, end_time]."""
    start_time = max(0.0, end_time - CONTEXT_FRAMES / FPS)
    by_end = {seg["end"]: seg for seg in segments}
    words = []
    for seg_end in sorted(by_end):
        if seg_end < start_time:
            continue
        seg = by_end[seg_end]
        if seg["start"] > end_time:
            break
        for w in seg.get("words", []):
            if start_time <= w.get("start", 0) <= end_time and start_time <= w.get("end", 0) <= end_time:
                words.append(w.get("word", "").strip())
    return " ".join(words).strip()


def _write(frame, clip_frames, size, frame_path, clip_path):
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(frame_path, quality=100)
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    for f in clip_frames:
        writer.write(f)
    writer.release()


def source_videos(mp4_dir, name: str):
    """Paths of the video(s) to read for the source video `name`: the video itself, or the views it shows
    side by side if the recording has only those. None if neither has been downloaded."""
    if (mp4_dir / name).exists():
        return (mp4_dir / name,)
    views = tuple(mp4_dir / v for v in SIDE_BY_SIDE_VIEWS.get(name, ()))
    return views if views and all(v.exists() for v in views) else None


class _Video:
    """One video, or several of equal height read in step with their frames placed side by side."""

    def __init__(self, paths):
        self.caps = [cv2.VideoCapture(str(p), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, os.cpu_count() or 1])
                     for p in paths]

    def grab(self) -> bool:
        return all([cap.grab() for cap in self.caps])  # a list, so that every video advances

    def retrieve(self):
        frames = []
        for cap in self.caps:
            ok, frame = cap.retrieve()
            if not ok:
                return False, None
            frames.append(frame)
        return True, frames[0] if len(frames) == 1 else cv2.hconcat(frames)

    def release(self):
        for cap in self.caps:
            cap.release()


def extract(video_paths, jobs: list):
    """jobs: (frame_index, (width, height), frame_path, clip_path). Writes each reference frame and the
    clip of the CONTEXT_FRAMES frames before it, in a single pass over the video (see source_videos)."""
    jobs = sorted(j for j in jobs if not (j[2].exists() and j[3].exists()))
    if not jobs:
        return
    needed = set()
    for k, *_ in jobs:
        needed.update(range(max(0, k - CONTEXT_FRAMES), k + 1))
    sizes = {j[1] for j in jobs}
    buffers = {size: deque(maxlen=CONTEXT_FRAMES) for size in sizes}

    cap = _Video(video_paths)
    pending = deque(jobs)
    idx, frames = 0, None
    while pending and cap.grab():
        if idx in needed:
            ok, raw = cap.retrieve()
            if not ok:
                break
            frames = {s: raw if raw.shape[1::-1] == s else cv2.resize(raw, s, interpolation=cv2.INTER_AREA)
                      for s in sizes}
            while pending and pending[0][0] == idx:
                k, size, frame_path, clip_path = pending.popleft()
                _write(frames[size], [f for i, f in buffers[size] if i >= k - CONTEXT_FRAMES], size,
                       frame_path, clip_path)
            for size, f in frames.items():
                buffers[size].append((idx, f))
        idx += 1
    cap.release()
    # A few annotations point just past the last frame: use the last frame, as the original extraction did.
    while pending and frames is not None and pending[0][0] - idx < FPS:
        k, size, frame_path, clip_path = pending.popleft()
        _write(frames[size], [f for i, f in buffers[size] if i >= k - CONTEXT_FRAMES], size, frame_path, clip_path)
    if pending:
        log.warning("%s ended before frame %d", video_paths[0], pending[0][0])


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", help="default: $COMIND_DATA_ROOT")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    p.add_argument("--split", default="test", choices=("train", "test", "all"))
    p.add_argument("--video_ids", nargs="+", help="only these recordings (e.g. to run several jobs in parallel)")
    args = p.parse_args(argv)
    root = resolve_root(args.data_root)
    video_ids = load_split(root, args.split)
    if args.video_ids:
        video_ids = set(args.video_ids) if video_ids is None else video_ids & set(args.video_ids)

    annotations = {task: load_annotations(annotation_path(root, task), video_ids) for task in args.tasks}
    recordings = sorted({vid for ann in annotations.values() for vid, _ in ann})
    for n, vid in enumerate(recordings, 1):
        rec = root / "recordings" / vid
        seg_file = rec / TRANSCRIPT_FILE
        segments = json.loads(seg_file.read_text())["segments"] if seg_file.exists() else []
        if not seg_file.exists() and (rec / "mp4s").exists():
            log.warning("%s: no transcript found, transcripts will be empty", vid)
        jobs = defaultdict(list)  # source video(s) -> jobs of all tasks
        for task in args.tasks:
            spec, out = get_task(task), benchmark_dir(root, task)
            fids = [fid for v, fid in annotations[task] if v == vid]
            videos = source_videos(rec / "mp4s", spec.source_video)
            if not fids or videos is None:
                continue
            for fid in fids:
                jobs[videos].append((int(fid), (spec.frame_width, spec.frame_height),
                                                out / "frames" / vid / f"{fid}.jpg",
                                                out / "clips" / vid / f"{fid}.mp4"))
            (out / "transcripts").mkdir(parents=True, exist_ok=True)
            transcripts = {fid: transcript_window(segments, int(fid) / FPS) for fid in fids}
            (out / "transcripts" / f"{vid}.json").write_text(json.dumps(transcripts, indent=1, ensure_ascii=False))
        if not jobs:
            continue
        log.info("[%d/%d] %s: %d samples", n, len(recordings), vid, sum(map(len, jobs.values())))
        for videos, source_jobs in jobs.items():
            extract(videos, source_jobs)


if __name__ == "__main__":
    main()
