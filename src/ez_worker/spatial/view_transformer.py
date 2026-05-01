from __future__ import annotations

import cv2
import numpy as np


class ViewTransformer:
    """
    Homography-based transformer from camera pixel coords to pitch metre coords.
    Matches tutorial's ViewTransformer in roboflow/sports exactly.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.m, _ = cv2.findHomography(
            source.astype(np.float32),
            target.astype(np.float32),
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2)
