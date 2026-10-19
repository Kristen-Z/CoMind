import cv2
import numpy as np
from PIL import Image


def uniform_indices(total: int, num: int) -> list:
    if total <= num:
        return list(range(total))
    return np.linspace(0, total - 1, num, dtype=int).tolist()


def count_frames(video_path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def sample_frames(video_path, num_frames: int) -> list:
    """num_frames RGB frames sampled uniformly from a clip (first and last included)."""
    if num_frames <= 0:
        return []
    wanted = set(uniform_indices(count_frames(video_path), num_frames))
    frames, idx = [], 0
    cap = cv2.VideoCapture(str(video_path))
    while wanted and cap.grab():
        if idx in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            wanted.discard(idx)
        idx += 1
    cap.release()
    return frames
