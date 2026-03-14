from __future__ import annotations

from pathlib import Path

from ez_worker.dataset_utils import prepare_roboflow_dataset_yaml


def train_detector(
    dataset_dir: Path,
    model_name: str,
    epochs: int,
    image_size: int,
    batch_size: int,
    device: str,
    project_dir: Path,
    run_name: str,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run 'pip install -e .[ml]' before training."
        ) from exc

    prepared_yaml = prepare_roboflow_dataset_yaml(dataset_dir)
    project_dir = project_dir.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)
    results = model.train(
        data=str(prepared_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )
    return Path(results.save_dir)
