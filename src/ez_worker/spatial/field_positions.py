"""Partial, quality-gated ground-plane coordinates from saved pitch landmarks."""
from __future__ import annotations

import bisect

import cv2
import numpy as np


class FieldProjector:
    def __init__(self, data, vertices, *, cut_frames=(), max_error_px=4.0):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.cuts = sorted(set(cut_frames))
        self.max_gap = max(1, int(data.get("stride", 10))) * 3
        self.matrices = {}
        self.quality = {}
        for key, entry in data.get("frames", {}).items():
            f = int(key)
            pairs = []
            for index, point in entry.get("keypoints", {}).items():
                i = int(index)
                if 0 <= i < len(self.vertices) and len(point) == 2 and np.isfinite(point).all():
                    pairs.append((point, self.vertices[i]))
            record = dict(valid=False, landmarks=len(pairs), reason="insufficient_landmarks")
            self.quality[f] = record
            if len(pairs) < 6:
                continue
            image, field = (np.asarray(p, dtype=np.float64) for p in zip(*pairs))
            # Fit in the image domain so the RANSAC threshold is explicitly pixels,
            # independent of whether the field template uses cm or metres.
            inverse, mask = cv2.findHomography(field, image, cv2.RANSAC, max_error_px)
            record["reason"] = "degenerate_or_inconsistent_landmarks"
            if inverse is None or mask is None or int(mask.sum()) < 6:
                continue
            try:
                matrix = np.linalg.inv(inverse)
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(matrix).all():
                continue
            inliers = mask.ravel().astype(bool)
            reprojection = cv2.perspectiveTransform(field.reshape(-1, 1, 2), inverse).reshape(-1, 2)
            error = np.linalg.norm(reprojection[inliers] - image[inliers], axis=1)
            if not np.isfinite(error).all() or np.percentile(error, 90) > max_error_px:
                continue
            self.matrices[f] = matrix
            record.update(valid=True, reason=None, inliers=int(inliers.sum()),
                          reprojection_p90_px=float(np.percentile(error, 90)))
        self.keys = sorted(self.matrices)

    def point(self, frame, x, y):
        if not self.keys or not np.isfinite([x, y]).all():
            return None
        j = bisect.bisect_left(self.keys, frame)
        neighbours = sorted(set(self.keys[max(0, j - 1):min(len(self.keys), j + 1)]))
        if frame in self.matrices:
            neighbours = [frame]
        segment = bisect.bisect_right(self.cuts, frame)
        neighbours = [k for k in neighbours if abs(k - frame) <= self.max_gap
                      and bisect.bisect_right(self.cuts, k) == segment]
        if not neighbours:
            return None

        def transform(k):
            v = self.matrices[k] @ np.array([x, y, 1.0])
            if abs(v[2]) < 1e-8 or not np.isfinite(v).all():
                return None
            return v[:2] / v[2]

        if len(neighbours) == 2 and neighbours[1] - neighbours[0] <= self.max_gap:
            a, b = neighbours
            pa, pb = transform(a), transform(b)
            if pa is None or pb is None:
                return None
            alpha = (frame - a) / (b - a)
            xy = (1 - alpha) * pa + alpha * pb
        else:
            xy = transform(min(neighbours, key=lambda k: abs(k - frame)))
        if xy is None or not np.isfinite(xy).all():
            return None
        lower, upper = self.vertices.min(axis=0), self.vertices.max(axis=0)
        margin = (upper - lower) * .05
        if np.any(xy < lower - margin) or np.any(xy > upper + margin):
            return None
        return [float(v) for v in xy]


def export_field_positions(tracks, video, data, vertices, *, cut_frames=()):
    projector = FieldProjector(data, vertices, cut_frames=cut_frames)
    if list(data.get("video_wh", [video.width, video.height])) != [video.width, video.height]:
        raise ValueError("Pitch landmark image dimensions do not match the source video")
    rows = []
    for o in tracks:
        if o.label not in ("player", "goalkeeper", "ball") or o.source_label == "referee":
            continue
        point = projector.point(o.frame_index, o.bbox.cx * video.width, o.bbox.y2 * video.height)
        rows.append(dict(frame_index=o.frame_index, track_id=o.track_id, label=o.label,
                         team_id=o.team_id, xy_template=point, available=point is not None,
                         ground_assumption=o.label == "ball", is_interpolated=o.is_interpolated))
    players = [r for r in rows if r["label"] != "ball"]
    return dict(coordinate_system="pitch_template_units", actual_field_dimensions_verified=False,
                template_bounds=[projector.vertices.min(axis=0).tolist(), projector.vertices.max(axis=0).tolist()],
                warning="Ball coordinates assume ground contact, not 3D position. Template coordinates are not verified physical distance.",
                player_projection_coverage=sum(r["available"] for r in players) / len(players) if players else None,
                calibration_samples=projector.quality, positions=rows)
