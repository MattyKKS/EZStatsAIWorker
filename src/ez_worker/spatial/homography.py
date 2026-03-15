from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def apply_pitch_mapping(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    template_path = run_dir / "pitch_mapping" / "pitch_mapping_template.json"
    tracks_path = run_dir / "tracks_with_teams.json"
    if not template_path.exists():
        raise FileNotFoundError(
            f"Pitch mapping template not found: {template_path}. "
            "Run 'ez-worker prepare-pitch-mapping --run-dir ...' first."
        )
    if not tracks_path.exists():
        raise FileNotFoundError(
            f"Team-aware track file not found: {tracks_path}. "
            "Run 'ez-worker apply-team-clusters --run-dir ...' first."
        )

    template = json.loads(template_path.read_text(encoding="utf-8"))
    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))

    image_points: list[list[float]] = []
    pitch_points: list[list[float]] = []
    for point in template.get("points", []):
        image_xy = point.get("image_xy_px")
        pitch_xy = point.get("pitch_xy_m")
        if image_xy is None or pitch_xy is None:
            continue
        image_points.append([float(image_xy[0]), float(image_xy[1])])
        pitch_points.append([float(pitch_xy[0]), float(pitch_xy[1])])

    if len(image_points) < 4:
        raise ValueError(
            "At least 4 visible pitch points are required before homography can be applied."
        )

    homography, inlier_mask = cv2.findHomography(
        np.asarray(image_points, dtype=np.float32),
        np.asarray(pitch_points, dtype=np.float32),
        method=0,
    )
    if homography is None:
        raise RuntimeError("OpenCV could not compute a homography from the provided points.")

    video_width = float(template["video_width"])
    video_height = float(template["video_height"])
    projected_tracks = []
    for track in tracks:
        if track.get("label") not in {"player", "ball"}:
            continue
        bbox = track["bbox"]
        bbox_cx = (float(bbox["x1"]) + float(bbox["x2"])) / 2.0
        bbox_y2 = float(bbox["y2"])
        image_point = np.asarray(
            [[[bbox_cx * video_width, bbox_y2 * video_height]]],
            dtype=np.float32,
        )
        pitch_point = cv2.perspectiveTransform(image_point, homography)[0][0]
        projected_tracks.append(
            {
                "frame_index": track["frame_index"],
                "track_id": track["track_id"],
                "label": track["label"],
                "team_id": track.get("team_id"),
                "pitch_xy_m": [round(float(pitch_point[0]), 3), round(float(pitch_point[1]), 3)],
            }
        )

    output = {
        "run_dir": str(run_dir),
        "reference_frame_path": template.get("reference_frame_path"),
        "used_point_count": len(image_points),
        "inlier_count": int(inlier_mask.sum()) if inlier_mask is not None else len(image_points),
        "homography_matrix": [[round(float(value), 6) for value in row] for row in homography.tolist()],
        "tracks_pitch_path": str(run_dir / "tracks_pitch.json"),
    }

    (run_dir / "tracks_pitch.json").write_text(json.dumps(projected_tracks, indent=2), encoding="utf-8")
    output_path = run_dir / "pitch_mapping" / "homography_result.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output_path
