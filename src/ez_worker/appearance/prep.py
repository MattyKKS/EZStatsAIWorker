from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


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
        samples = crop_paths[: min(20, len(crop_paths))]
        entries.append(
            {
                "track_id": track_id,
                "crop_count": len(crop_paths),
                "sample_crops": [str(path) for path in samples],
                "upper_body_hsv": _mean_upper_body_color(samples),
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
    """
    Returns the dominant jersey color as [H, S, V] using pixel-level KMeans.
    KMeans(k=2) on jersey pixels — take the larger cluster center as the jersey color.
    This is more robust than a mean because background bleed doesn't skew the result.
    """
    from sklearn.cluster import KMeans

    all_pixels: list[np.ndarray] = []
    for crop_path in crop_paths:
        image = cv2.imread(str(crop_path))
        if image is None or image.size == 0:
            continue
        h, w = image.shape[:2]
        # Jersey region: top 55% height, center 70% width
        y_end = max(1, int(h * 0.55))
        x_start = int(w * 0.15)
        x_end = max(x_start + 1, int(w * 0.85))
        jersey = image[:y_end, x_start:x_end]
        if jersey.size == 0:
            continue

        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)

        # Mask: exclude green pitch (H 35-90, S>40) and very dark pixels (V<40)
        green_mask = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 39))
        exclude = cv2.bitwise_or(green_mask, dark_mask)
        valid_mask = cv2.bitwise_not(exclude)

        valid_pixels = hsv[valid_mask > 0]
        if len(valid_pixels) >= 10:
            all_pixels.append(valid_pixels)

    if not all_pixels:
        return None

    pixels = np.concatenate(all_pixels, axis=0).astype(np.float32)

    if len(pixels) < 4:
        center = pixels.mean(axis=0)
        return [round(float(v), 2) for v in center]

    # KMeans(2) — dominant cluster is the jersey color
    km = KMeans(n_clusters=2, random_state=42, n_init=5)
    labels = km.fit_predict(pixels)
    counts = np.bincount(labels)
    dominant_idx = int(np.argmax(counts))
    center = km.cluster_centers_[dominant_idx]
    return [round(float(v), 2) for v in center]
