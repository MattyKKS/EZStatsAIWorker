from __future__ import annotations

import json
from pathlib import Path

import cv2

from ez_worker.io.video import load_video_meta


STANDARD_PITCH_POINTS = [
    {"key": "top_halfway_touchline", "pitch_xy_m": [52.5, 0.0]},
    {"key": "bottom_halfway_touchline", "pitch_xy_m": [52.5, 68.0]},
    {"key": "center_spot", "pitch_xy_m": [52.5, 34.0]},
    {"key": "center_circle_left", "pitch_xy_m": [43.35, 34.0]},
    {"key": "center_circle_right", "pitch_xy_m": [61.65, 34.0]},
    {"key": "center_circle_top", "pitch_xy_m": [52.5, 24.85]},
    {"key": "center_circle_bottom", "pitch_xy_m": [52.5, 43.15]},
]


def prepare_pitch_mapping(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    video_path = Path(summary["video_path"])
    video = load_video_meta(video_path)

    prep_dir = run_dir / "pitch_mapping"
    prep_dir.mkdir(parents=True, exist_ok=True)
    frame_path = prep_dir / "reference_frame.jpg"
    reference_frame_index = _export_reference_frame(video.path, frame_path)

    payload = {
        "run_dir": str(run_dir),
        "video_path": str(video.path),
        "reference_frame_path": str(frame_path),
        "reference_frame_index": reference_frame_index,
        "video_width": video.width,
        "video_height": video.height,
        "pitch_size_m": {"length": 105.0, "width": 68.0},
        "status": "template_only",
        "instructions": [
            "Open reference_frame.jpg and mark visible pitch points.",
            "For midfield broadcast views, start with halfway-line and center-circle points.",
            "Fill image_xy_px for every point you can clearly see.",
            "Leave null for points that are not visible in this frame.",
            "This template is for later homography work and does not change V1 tracking.",
        ],
        "points": [
            {
                "key": item["key"],
                "pitch_xy_m": item["pitch_xy_m"],
                "image_xy_px": None,
            }
            for item in STANDARD_PITCH_POINTS
        ],
    }

    output_path = prep_dir / "pitch_mapping_template.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _export_reference_frame(video_path: Path, output_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for pitch mapping prep: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frame = max(0, frame_count // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to read reference frame from: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    return target_frame
