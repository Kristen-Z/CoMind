from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    source_video: str      # recording video the reference frames and clips are cut from
    frame_width: int
    frame_height: int
    bbox_keys: tuple       # keys of the predicted boxes in the model answer
    bbox_format: str       # box format the prompt asks for (see comind.bbox)
    num_frames: int        # context frames given to the model (0: reference frame only)
    display_fields: tuple  # (answer key, label) shown on visualizations


JOINT_ATTENTION = TaskSpec(
    name="joint_attention",
    source_video="leader_helper_trimmed_sync.mp4",
    frame_width=1408,
    frame_height=704,
    bbox_keys=("left object bounding box", "right object bounding box"),
    bbox_format="xyxy_pixels",
    num_frames=0,
    display_fields=(("object category", "Object"), ("cue type", "Cue")),
)

HANDOVER = TaskSpec(
    name="handover",
    source_video="leader_helper_trimmed_sync.mp4",
    frame_width=1408,
    frame_height=704,
    bbox_keys=("object bounding box",),
    bbox_format="xyxy_pixels",
    num_frames=5,
    display_fields=(("object category", "Object"), ("delivering flow", "Flow"), ("initiator", "Initiator"),
                    ("initiation type", "Init type"), ("handover happen offset", "Offset")),
)

SCOIA = TaskSpec(
    name="scoia",
    source_video="helper_trimmed_sync.mp4",
    frame_width=1408,
    frame_height=1408,
    bbox_keys=("object bounding box",),
    bbox_format="yxyx_1000",
    num_frames=5,
    display_fields=(("action verb", "Verb"), ("action noun", "Noun"), ("cue type", "Cue")),
)

TASK_SPECS = {t.name: t for t in (JOINT_ATTENTION, HANDOVER, SCOIA)}


def get_task(name: str) -> TaskSpec:
    if name not in TASK_SPECS:
        raise ValueError(f"Unknown task '{name}', choose from {list(TASK_SPECS)}")
    return TASK_SPECS[name]
