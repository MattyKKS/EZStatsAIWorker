from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from sports.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()
VERTICES = CONFIG.vertices  # 32 (x_cm, y_cm) tuples, index matches model output

STRIDE = 60
ROBOFLOW_MODEL_ID = "football-field-detection-f07vi/15"


def detect_pitch_keypoints(
    run_dir: Path,
    model_path: Path | None = None,
    api_key: str | None = None,
    per_frame_stride: int | None = None,
) -> Path:
    """Detect pitch keypoints and write `pitch_keypoints.json`.

    When `per_frame_stride` is given, ALSO writes `pitch_keypoints_per_frame.json`
    holding the keypoints of every sampled frame, so downstream code can build a
    homography that follows the camera instead of one fixed matrix for the whole
    clip (RC-1). The single-best-frame output is byte-identical either way: the
    "best" frame is still chosen only from multiples of STRIDE, so enabling this
    cannot change any existing result.
    """
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

    per_frame: dict[str, dict] = {}
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
        result_kps, best_frame_idx, per_frame = _detect_via_local(
            cap, model_path, total, w, h, per_frame_stride
        )

    cap.release()

    best_count = len(result_kps)
    output = {
        "video_wh": [w, h],
        "best_frame": best_frame_idx,
        "visible_keypoint_count": best_count,
        "keypoints": result_kps,
        "keypoint_pitch_xy_cm": {
            k: list(VERTICES[int(k)])
            for k in result_kps
            if int(k) < len(VERTICES)
        },
    }
    output_path = run_dir / "pitch_keypoints.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"  Detected {best_count} pitch keypoints → {output_path}")

    if per_frame:
        usable = sum(1 for v in per_frame.values() if len(v["keypoints"]) >= 6)
        pf = {
            "video_wh": [w, h],
            "stride": per_frame_stride,
            "frames": per_frame,
        }
        pf_path = run_dir / "pitch_keypoints_per_frame.json"
        pf_path.write_text(json.dumps(pf), encoding="utf-8")
        print(f"  Per-frame keypoints: {len(per_frame)} sampled, "
              f"{usable} with >=6 points (homography-capable) → {pf_path.name}")
    return output_path


def _detect_via_local(cap, model_path: Path, total: int, w: int, h: int,
                      per_frame_stride: int | None = None):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed.") from exc

    model = YOLO(str(model_path))
    best_kps: dict[str, list[float]] = {}
    best_count = 0
    best_frame_idx = 0
    frame_idx = 0
    per_frame: dict[str, dict] = {}

    # Sample at the finer of the two strides, but only let multiples of STRIDE
    # compete for "best frame" — so turning per-frame capture on never changes
    # the single-homography result that existing runs depend on.
    step = min(STRIDE, per_frame_stride) if per_frame_stride else STRIDE

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            result = model(frame, verbose=False)[0]
            kps = _parse_local_keypoints(result)
            if per_frame_stride and frame_idx % per_frame_stride == 0 and kps:
                per_frame[str(frame_idx)] = {"keypoints": kps}
            if frame_idx % STRIDE == 0 and len(kps) > best_count:
                best_count = len(kps)
                best_kps = kps
                best_frame_idx = frame_idx
            print(f"    frame {frame_idx}/{total}: {len(kps)} keypoints", end="\r")
        frame_idx += 1

    print()
    return best_kps, best_frame_idx, per_frame


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
            kps = _infer_frame_api(client, frame)
            if len(kps) > best_count:
                best_count = len(kps)
                best_kps = kps
                best_frame_idx = frame_idx
            print(f"    frame {frame_idx}/{total}: {len(kps)} keypoints", end="\r")
        frame_idx += 1

    print()
    return best_kps, best_frame_idx


def _infer_frame_api(client, frame: np.ndarray) -> dict[str, list[float]]:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    image_str = f"data:image/jpeg;base64,{b64}"

    try:
        result = client.infer(image_str, model_id=ROBOFLOW_MODEL_ID)
    except Exception as e:
        print(f"\n    API error: {e}")
        return {}

    return _parse_api_keypoints(result)


def _parse_api_keypoints(result: dict) -> dict[str, list[float]]:
    kps: dict[str, list[float]] = {}
    predictions = result.get("predictions", [])
    if not predictions:
        return kps

    best_pred = max(predictions, key=lambda p: len(p.get("keypoints", [])))
    for kp in best_pred.get("keypoints", []):
        class_id = kp.get("class_id", -1)
        confidence = kp.get("confidence", 0.0)
        x = kp.get("x", 0.0)
        y = kp.get("y", 0.0)

        if confidence < 0.5 or (x <= 1 and y <= 1):
            continue
        if class_id < 0 or class_id >= len(VERTICES):
            continue

        kps[str(class_id)] = [float(x), float(y)]

    return kps


def _parse_local_keypoints(result) -> dict[str, list[float]]:
    kps: dict[str, list[float]] = {}
    if result.keypoints is None:
        return kps

    xy = result.keypoints.xy
    conf = result.keypoints.conf

    if xy is None or len(xy) == 0:
        return kps

    pts = xy[0].cpu().numpy()
    confs = conf[0].cpu().numpy() if conf is not None else np.ones(len(pts))

    for i, (pt, c) in enumerate(zip(pts, confs)):
        if i >= len(VERTICES):
            break
        if c < 0.5 or (pt[0] <= 1 and pt[1] <= 1):
            continue
        kps[str(i)] = [float(pt[0]), float(pt[1])]

    return kps
