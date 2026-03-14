from __future__ import annotations

from pathlib import Path

import yaml


def prepare_roboflow_dataset_yaml(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.resolve()
    source_yaml = dataset_dir / "data.yaml"
    if not source_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {source_yaml}")

    payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    payload["train"] = str((dataset_dir / "train" / "images").resolve())
    payload["val"] = str((dataset_dir / "valid" / "images").resolve())
    payload["test"] = str((dataset_dir / "test" / "images").resolve())

    prepared_yaml = dataset_dir / "data.local.yaml"
    prepared_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return prepared_yaml
