"""Bounded, time-varying ground-plane projection for event analysis."""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import cv2
import numpy as np


def project_tracks(run_dir, tracks, video):
    path = Path(run_dir) / "pitch_keypoints_per_frame.json"
    if not path.exists():
        return None, None
    from sports.configs.soccer import SoccerPitchConfiguration

    vertices = SoccerPitchConfiguration().vertices
    data = json.loads(path.read_text(encoding="utf-8"))
    matrices = {}
    for frame, entry in data.get("frames", {}).items():
        pairs = [(xy, vertices[int(i)]) for i, xy in entry.get("keypoints", {}).items()
                 if 0 <= int(i) < len(vertices)]
        if len(pairs) < 6:
            continue
        src, dst = (np.asarray(a, dtype=np.float32) for a in zip(*pairs))
        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 50.0)
        if matrix is not None and mask is not None and int(mask.sum()) >= 5:
            matrices[int(frame)] = matrix
    if not matrices:
        return None, None
    keys = sorted(matrices)
    max_gap = max(1, int(data.get("stride", 10))) * 3
    balls, players = {}, {}
    for obs in tracks:
        frame = obs.frame_index
        j = bisect.bisect_left(keys, frame)
        left, right = keys[max(0, j - 1)], keys[min(j, len(keys) - 1)]
        if min(abs(left - frame), abs(right - frame)) > max_gap:
            continue
        point = np.array([[[(obs.bbox.x1 + obs.bbox.x2) * video.width / 2,
                            obs.bbox.y2 * video.height]]], dtype=np.float32)
        if left != right and right - left <= max_gap:
            alpha = (frame - left) / (right - left)
            # Interpolate projected points, not homography coefficients. Nearest
            # sample switching creates artificial ball-speed spikes at boundaries.
            xy = ((1 - alpha) * cv2.perspectiveTransform(point, matrices[left])
                  + alpha * cv2.perspectiveTransform(point, matrices[right]))[0, 0]
        else:
            nearest = min((left, right), key=lambda k: abs(k - frame))
            xy = cv2.perspectiveTransform(point, matrices[nearest])[0, 0]
        if not np.isfinite(xy).all():
            continue
        if obs.label == "ball":
            balls[frame] = tuple(float(x) for x in xy)
        elif obs.label == "player":
            players.setdefault(frame, {})[obs.track_id] = tuple(float(x) for x in xy)
    # Event thresholds must never compare pixel fallback distances with cm.
    ball_frames = {o.frame_index for o in tracks if o.label == "ball"}
    if not ball_frames or any(f not in balls or f not in players for f in ball_frames):
        print("Incomplete pitch coverage: using pixel units for the entire event pass.", flush=True)
        return None, None
    return balls, players
