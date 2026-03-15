from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2


def prepare_appearance_inputs(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    crops_dir = run_dir / "player_crops"
    if not crops_dir.exists():
        raise FileNotFoundError(
            f"Player crops folder not found: {crops_dir}. "
            "Run analyze with --export-player-crops first."
        )

    track_entries = _collect_track_entries(crops_dir)
    manifest = {
        "run_dir": str(run_dir),
        "crops_dir": str(crops_dir),
        "track_count": len(track_entries),
        "tracks": track_entries,
        "next_stage": {
            "embedding_model": "SigLIP",
            "reduction": "UMAP",
            "clustering": "KMeans",
        },
    }

    output_path = run_dir / "appearance_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def _collect_track_entries(crops_dir: Path) -> list[dict]:
    track_to_files: dict[int, list[Path]] = defaultdict(list)
    for crop_path in sorted(crops_dir.rglob("*.jpg")):
        track_id = _parse_track_id(crop_path.parent.name)
        if track_id is None:
            continue
        track_to_files[track_id].append(crop_path)

    entries: list[dict] = []
    for track_id in sorted(track_to_files):
        crop_paths = track_to_files[track_id]
        samples = crop_paths[: min(5, len(crop_paths))]
        entries.append(
            {
                "track_id": track_id,
                "crop_count": len(crop_paths),
                "sample_crops": [str(path) for path in samples],
                "upper_body_mean_bgr": _mean_upper_body_color(samples),
            }
        )
    return entries


def _parse_track_id(dirname: str) -> int | None:
    if not dirname.startswith("track_"):
        return None
    suffix = dirname.removeprefix("track_")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _mean_upper_body_color(crop_paths: list[Path]) -> list[float] | None:
    colors: list[list[float]] = []
    for crop_path in crop_paths:
        image = cv2.imread(str(crop_path))
        if image is None or image.size == 0:
            continue
        height = image.shape[0]
        top_half = image[: max(1, height // 2), :]
        if top_half.size == 0:
            continue
        mean_bgr = top_half.mean(axis=(0, 1)).tolist()
        colors.append([round(float(value), 2) for value in mean_bgr])

    if not colors:
        return None

    channel_count = len(colors[0])
    averaged = [
        round(sum(color[channel] for color in colors) / len(colors), 2)
        for channel in range(channel_count)
    ]
    return averaged
