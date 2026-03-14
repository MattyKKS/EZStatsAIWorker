from __future__ import annotations

from pathlib import Path


DATA_DIRS = [
    Path("data/raw"),
    Path("data/interim"),
    Path("data/processed"),
    Path("data/datasets/roboflow/detector"),
    Path("data/datasets/roboflow/keypoints"),
    Path("data/datasets/kaggle"),
    Path("data/datasets/soccernet"),
]


def init_data_layout(root: Path) -> list[Path]:
    created: list[Path] = []
    for relative_path in DATA_DIRS:
        target = root / relative_path
        target.mkdir(parents=True, exist_ok=True)
        created.append(target)
    return created
