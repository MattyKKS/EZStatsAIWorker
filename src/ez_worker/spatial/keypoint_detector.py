from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

KEYPOINT_LABELS = [
    "top_left_corner",
    "top_right_corner",
    "bottom_left_corner",
    "bottom_right_corner",
    "centre_circle_top",
    "centre_circle_bottom",
    "centre_spot",
    "penalty_spot_top",
    "penalty_spot_bottom",
    "top_left_penalty_area_top_left",
    "top_left_penalty_area_top_right",
    "top_left_penalty_area_bottom_left",
    "top_left_penalty_area_bottom_right",
    "top_right_penalty_area_top_left",
    "top_right_penalty_area_top_right",
    "top_right_penalty_area_bottom_left",
    "top_right_penalty_area_bottom_right",
    "bottom_left_penalty_area_top_left",
    "bottom_left_penalty_area_top_right",
    "bottom_left_penalty_area_bottom_left",
    "bottom_left_penalty_area_bottom_right",
    "bottom_right_penalty_area_top_left",
    "bottom_right_penalty_area_top_right",
    "bottom_right_penalty_area_bottom_left",
    "bottom_right_penalty_area_bottom_right",
    "halfway_top",
    "halfway_bottom",
    "halfway_centre",
]

KEYPOINT_PITCH_XY_M: dict[str, tuple[float, float]] = {
    "top_left_corner": (0.0, 0.0),
    "top_right_corner": (105.0, 0.0),
    "bottom_left_corner": (0.0, 68.0),
    "bottom_right_corner": (105.0, 68.0),
    "centre_circle_top": (52.5, 25.85),
    "centre_circle_bottom": (52.5, 42.15),
    "centre_spot": (52.5, 34.0),
    "penalty_spot_top": (11.0, 34.0),
    "penalty_spot_bottom": (94.0, 34.0),
    "top_left_penalty_area_top_left": (0.0, 13.84),
    "top_left_penalty_area_top_right": (16.5, 13.84),
    "top_left_penalty_area_bottom_left": (0.0, 54.16),
    "top_left_penalty_area_bottom_right": (16.5, 54.16),
    "top_right_penalty_area_top_left": (88.5, 13.84),
    "top_right_penalty_area_top_right": (105.0, 13.84),
    "top_right_penalty_area_bottom_left": (88.5, 54.16),
    "top_right_penalty_area_bottom_right": (105.0, 54.16),
    "bottom_left_penalty_area_top_left": (0.0, 13.84),
    "bottom_left_penalty_area_top_right": (16.5, 13.84),
    "bottom_left_penalty_area_bottom_left": (0.0, 54.16),
    "bottom_left_penalty_area_bottom_right": (16.5, 54.16),
    "bottom_right_penalty_area_top_left": (88.5, 13.84),
    "bottom_right_penalty_area_top_right": (105.0, 13.84),
    "bottom_right_penalty_area_bottom_left": (88.5, 54.16),
    "bottom_right_penalty_area_bottom_right": (105.0, 54.16),
    "halfway_top": (52.5, 0.0),
    "halfway_bottom": (52.5, 68.0),
    "halfway_centre": (52.5, 34.0),
}

STRIDE = 60
ROBOFLOW_MODEL_ID = "football-field-detection-f07vi/15"


def detect_pitch_keypoints(
    run_dir: Path,
    model_path: Path | None = None,
    api_key: str | None = None,
) -> Path:
    run_dir = run_dir.resolve()

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    video_path = Path(summary["video_path"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    use_api = api_key and (model_path is None or not Path(model_path).exists())

    if use_api:
        print(f"  Using Roboflow inference API (model: {ROBOFLOW_MODEL_ID})")
        result_kps, best_frame_idx = _detect_via_api(cap, api_key, total, w, h)
    else:
        model_path = Path(model_path).resolve()
        if not model_path.exists():
            cap.release()
            raise FileNotFoundError(
                f"Pitch keypoint model not found: {model_path}\n"
                "Pass --api-key YOUR_ROBOFLOW_KEY to use the cloud API instead."
            )
        print(f"  Using local model: {model_path}")
        result_kps, best_frame_idx = _detect_via_local(cap, model_path, total, w, h)

    cap.release()

    best_count = len(result_kps)
    output = {
        "run_dir": str(run_dir),
        "video_wh": [w, h],
        "best_frame": best_frame_idx,
        "visible_keypoint_count": best_count,
        "keypoints": result_kps,
        "keypoint_pitch_xy_m": {
            k: list(KEYPOINT_PITCH_XY_M[k])
            for k in result_kps
            if k in KEYPOINT_PITCH_XY_M
        },
    }
    output_path = run_dir / "pitch_keypoints.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"  Detected {best_count} pitch keypoints → {output_path}")
    return output_path


def _detect_via_local(cap, model_path: Path, total: int, w: int, h: int):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed.") from exc

    model = YOLO(str(model_path))
    best_kps: dict[str, list[float]] = {}
    best_count = 0
    best_frame_idx = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % STRIDE == 0:
            result = model(frame, verbose=False)[0]
            kps = _parse_local_keypoints(result, w, h)
            if len(kps) > best_count:
                best_count = len(kps)
                best_kps = kps
                best_frame_idx = frame_idx
            print(f"    frame {frame_idx}/{total}: {len(kps)} keypoints", end="\r")
        frame_idx += 1

    print()
    return best_kps, best_frame_idx


def _detect_via_api(cap, api_key: str, total: int, w: int, h: int):
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError as exc:
        raise RuntimeError(
            "inference-sdk is not installed. Run: pip install inference-sdk"
        ) from exc

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )

    best_kps: dict[str, list[float]] = {}
    best_count = 0
    best_frame_idx = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % STRIDE == 0:
            kps = _infer_frame_api(client, frame, w, h)
            if len(kps) > best_count:
                best_count = len(kps)
                best_kps = kps
                best_frame_idx = frame_idx
            print(f"    frame {frame_idx}/{total}: {len(kps)} keypoints", end="\r")
        frame_idx += 1

    print()
    return best_kps, best_frame_idx


def _infer_frame_api(client, frame: np.ndarray, w: int, h: int) -> dict[str, list[float]]:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    image_str = f"data:image/jpeg;base64,{b64}"

    try:
        result = client.infer(image_str, model_id=ROBOFLOW_MODEL_ID)
    except Exception as e:
        print(f"\n    API error: {e}")
        return {}

    return _parse_api_keypoints(result, w, h)


def _parse_api_keypoints(result: dict, w: int, h: int) -> dict[str, list[float]]:
    kps: dict[str, list[float]] = {}
    predictions = result.get("predictions", [])
    if not predictions:
        return kps

    # Pick prediction with most keypoints
    best_pred = max(predictions, key=lambda p: len(p.get("keypoints", [])))
    keypoints = best_pred.get("keypoints", [])

    for kp in keypoints:
        class_id = kp.get("class_id", -1)
        confidence = kp.get("confidence", 0.0)
        x = kp.get("x", 0.0)
        y = kp.get("y", 0.0)

        if confidence < 0.5 or (x < 1 and y < 1):
            continue
        if class_id < 0 or class_id >= len(KEYPOINT_LABELS):
            continue

        label = KEYPOINT_LABELS[class_id]
        kps[label] = [float(x), float(y)]

    return kps


def _parse_local_keypoints(result, w: int, h: int) -> dict[str, list[float]]:
    kps: dict[str, list[float]] = {}
    if result.keypoints is None:
        return kps

    xy = result.keypoints.xy
    conf = result.keypoints.conf

    if xy is None or len(xy) == 0:
        return kps

    best_det = 0
    if len(xy) > 1 and conf is not None:
        best_det = int(conf[0].sum().argmax()) if conf.dim() > 1 else 0

    pts = xy[best_det].cpu().numpy()
    confs = conf[best_det].cpu().numpy() if conf is not None else np.ones(len(pts))

    for i, (pt, c) in enumerate(zip(pts, confs)):
        if i >= len(KEYPOINT_LABELS):
            break
        if c < 0.5 or (pt[0] < 1 and pt[1] < 1):
            continue
        label = KEYPOINT_LABELS[i]
        kps[label] = [float(pt[0]), float(pt[1])]

    return kps
