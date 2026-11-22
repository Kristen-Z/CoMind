# Modified for CoMind: --dataset_use takes paths of annotation files instead of names of registered datasets.

import re
from pathlib import Path


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    """Each name is the path of a .json or .jsonl annotation file whose image paths are relative to the file,
    optionally followed by %<percentage of the samples to use>, e.g. data/comind/finetune/scoia_train.json%50."""
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        path = Path(re.sub(r"%(\d+)$", "", dataset_name))
        if not path.is_file():
            raise ValueError(f"do not find {path}")
        config_list.append({"annotation_path": str(path), "data_path": str(path.parent),
                            "sampling_rate": sampling_rate})
    return config_list
