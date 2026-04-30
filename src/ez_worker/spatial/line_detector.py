from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def detect_penalty_marks(
    video_path: Path,
    sample_second: float = 8.0,
) -> list[tuple[float, float]]:
    """
    Detect penalty mark locations from a broadcast football video.

    Samples one frame and finds small white circular blobs on the green pitch.
    Returns a list of (cx_norm, cy_norm) in [0, 1] range — typically 0–3 marks
    (center spot + up to two penalty spots visible in broadcast view).

    Returns empty list if detection fails or no marks are found.
    """
    frame = _extract_frame(video_path, sample_second)
    if frame is None:
        return []
    marks = _detect_marks_in_frame(frame)
    if not marks:
        # Retry with a different frame position if first attempt finds nothing
        frame2 = _extract_frame(video_path, max(2.0, sample_second - 5.0))
        if frame2 is not None:
            marks = _detect_marks_in_frame(frame2)
    return marks


def _extract_frame(video_path: Path, target_second: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * target_second))
    ok, frame = cap.read()
    cap.release()
    return frame if ok and frame is not None else None


def _detect_marks_in_frame(frame: np.ndarray) -> list[tuple[float, float]]:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Isolate green pitch area
    pitch_mask = cv2.inRange(hsv, np.array([30, 35, 35]), np.array([90, 255, 255]))
    pitch_mask = cv2.dilate(pitch_mask, np.ones((7, 7), np.uint8), iterations=2)

    # Isolate white markings (low saturation, high value)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([180, 45, 255]))

    # White markings that sit on the pitch
    white_on_pitch = cv2.bitwise_and(white_mask, pitch_mask)

    # Mild blur to merge adjacent pixels of the same mark
    white_on_pitch = cv2.GaussianBlur(white_on_pitch, (3, 3), 0)
    _, white_on_pitch = cv2.threshold(white_on_pitch, 100, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(white_on_pitch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, float, float]] = []  # (cx_norm, cy_norm, score)

    for contour in contours:
        area = cv2.contourArea(contour)
        # Penalty marks are small painted dots — filter by area range
        # At 1280px wide, a ~0.3m mark ≈ 3-5px diameter → 7–400 px²
        if area < 6 or area > 600:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter < 1.0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        # Must be reasonably circular (not a line segment)
        if circularity < 0.35:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Reject detections very close to frame borders (artifacts)
        margin_x = w * 0.04
        margin_y = h * 0.06
        if cx < margin_x or cx > w - margin_x or cy < margin_y or cy > h - margin_y:
            continue

        # Score: prefer more circular and appropriately-sized blobs
        score = circularity * min(area / 50.0, 1.0)
        candidates.append((cx / w, cy / h, score))

    # Sort by score descending and deduplicate nearby marks
    candidates.sort(key=lambda item: item[2], reverse=True)
    marks = _deduplicate_marks([(cx, cy) for cx, cy, _ in candidates], min_separation=0.04)
    return marks[:5]


def _deduplicate_marks(
    marks: list[tuple[float, float]],
    min_separation: float,
) -> list[tuple[float, float]]:
    """Remove marks that are too close together (keep the first/highest-scored one)."""
    kept: list[tuple[float, float]] = []
    for cx, cy in marks:
        too_close = False
        for kx, ky in kept:
            if ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5 < min_separation:
                too_close = True
                break
        if not too_close:
            kept.append((cx, cy))
    return kept
