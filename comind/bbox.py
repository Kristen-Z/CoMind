"""GT boxes are [x, y, w, h] in pixels of the reference frame. Predicted boxes come in one of:

    yxyx_1000:   [ymin, xmin, ymax, xmax] normalised to [0, 1000]   (SCOIA prompt)
    xyxy_1000:   [xmin, ymin, xmax, ymax] normalised to [0, 1000]   (Qwen-VL)
    xyxy_pixels: [xmin, ymin, xmax, ymax] in pixels                 (joint attention and handover prompts)
"""

BBOX_FORMATS = ("yxyx_1000", "xyxy_1000", "xyxy_pixels")


def canonical_bbox_format(name: str) -> str:
    if name not in BBOX_FORMATS:
        raise ValueError(f"Unknown bbox format '{name}', choose from {BBOX_FORMATS}")
    return name


def gt_to_xyxy(bbox) -> tuple:
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def pred_to_xyxy(bbox, bbox_format: str, frame_width: float, frame_height: float):
    """Predicted box -> pixel (x1, y1, x2, y2), or None if malformed."""
    bbox_format = canonical_bbox_format(bbox_format)
    if len(bbox) != 4:
        return None
    if bbox_format == "yxyx_1000":
        y1, x1, y2, x2 = bbox
    else:
        x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        return None
    # via [x, y, w, h], same float arithmetic as the paper's evaluation
    if bbox_format == "xyxy_pixels":
        x, y, w, h = float(x1), float(y1), float(x2 - x1), float(y2 - y1)
    else:
        x1, y1, x2, y2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
        x, y = frame_width * x1, frame_height * y1
        w, h = frame_width * (x2 - x1), frame_height * (y2 - y1)
    return (x, y, x + w, y + h)


def parse_pred_box(raw, bbox_format: str, frame_width: float, frame_height: float):
    if not (raw and isinstance(raw, list) and len(raw) == 4):
        return None
    try:
        values = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    return pred_to_xyxy(values, bbox_format, frame_width, frame_height)


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
