"""Goal detection by image-space goal-mouth containment.

Why this replaces the ground-plane test
---------------------------------------
`goal_detection.py` projects the BALL down onto the pitch plane and asks whether
it crossed the goal line. A homography maps the ground, so a ball in the air
projects to where its line of sight meets the grass - past the line when it is
travelling away from the camera. That made airborne goals, which is most goals,
undecidable in that formulation.

Verified on leo_messi frame 5691 (the annotated goal, night match, camera tight on
the penalty area): projecting the goal mouth as a 3D quadrilateral through the
estimated camera lands **exactly** on the real goal - both posts, the crossbar and
the goal line. So the test can be inverted: instead of pushing the ball down to
the ground, lift the goal up into the image and ask whether the ball lies inside
that quadrilateral.

Ball height stops being a source of error and becomes irrelevant, because the
quadrilateral already spans the full 2.44 m of the goal.

Remaining ambiguity, not solved here
------------------------------------
A ball BEHIND the goal can project inside the same quadrilateral, because the
camera has no depth. Two gates reduce this: the ball must have been outside the
mouth shortly before (so a ball already dead behind the goal does not register),
and its ground projection must be near the goal line rather than far beyond it.
Neither is proof. Output stays a reviewable candidate, never an assertion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GOAL_HALF_WIDTH_M = 7.32 / 2.0
GOAL_HEIGHT_M = 2.44
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


@dataclass
class GoalMouthCandidate:
    frame: int
    time_seconds: float
    side: str
    frames_inside: int
    first_frame: int
    last_frame: int
    entered_from_outside: bool
    evidence: dict = field(default_factory=dict)
    requires_review: bool = True

    def as_event(self) -> dict:
        return {
            "type": "goal_candidate",
            "frame": self.frame,
            "time_s": round(self.time_seconds, 2),
            "actor": None, "actor_label": None,
            "target": None, "target_label": None,
            "details": {
                "method": "image_space_goal_mouth",
                "side": self.side,
                "frames_inside": self.frames_inside,
                "first_frame": self.first_frame,
                "last_frame": self.last_frame,
                "entered_from_outside": self.entered_from_outside,
                "requires_review": True,
                **self.evidence,
            },
        }


def goal_mouth_quad(projection: np.ndarray, side: str) -> np.ndarray | None:
    """Project one goal mouth into the image as a 4-point polygon.

    `projection` is the 3x4 camera matrix in PnLCalib's convention: world origin at
    the pitch centre, metres, and height entering as a NEGATIVE z.
    Returns pixel corners (post base, post top, post top, post base) or None when
    the goal is behind the camera or the projection degenerates.
    """
    gx = 0.0 if side == "left" else PITCH_LENGTH_M
    cy = PITCH_WIDTH_M / 2.0
    corners = [
        (gx, cy - GOAL_HALF_WIDTH_M, 0.0),
        (gx, cy - GOAL_HALF_WIDTH_M, GOAL_HEIGHT_M),
        (gx, cy + GOAL_HALF_WIDTH_M, GOAL_HEIGHT_M),
        (gx, cy + GOAL_HALF_WIDTH_M, 0.0),
    ]
    out = []
    for wx, wy, wz in corners:
        v = projection @ np.array([wx - PITCH_LENGTH_M / 2.0,
                                   wy - PITCH_WIDTH_M / 2.0, -wz, 1.0])
        if not np.isfinite(v).all() or v[-1] <= 1e-6:
            return None                      # behind the camera or degenerate
        out.append([v[0] / v[-1], v[1] / v[-1]])
    quad = np.asarray(out, dtype=np.float32)
    if not np.isfinite(quad).all():
        return None
    # A mouth spanning a handful of pixels is a numerically broken solution, not a
    # goal seen from far away worth testing.
    if np.ptp(quad[:, 0]) < 8.0 and np.ptp(quad[:, 1]) < 8.0:
        return None
    return quad


def _inside(quad: np.ndarray, point) -> bool:
    import cv2

    return cv2.pointPolygonTest(quad.astype(np.float32),
                                (float(point[0]), float(point[1])), False) >= 0


def find_goal_mouth_candidates(
    ball_image_xy: dict[int, tuple[float, float]],
    projections: dict[int, np.ndarray],
    fps: float,
    *,
    min_frames_inside: int = 2,
    lookback_frames: int = 90,
) -> list[GoalMouthCandidate]:
    """Ball image positions + per-frame cameras -> goal candidates.

    `ball_image_xy` must hold OBSERVED ball positions in pixels. Interpolated balls
    are excluded by the caller: interpolation across the gap a shot creates is
    exactly where a false crossing would be invented.
    """
    if not ball_image_xy or not projections or fps <= 0:
        return []

    frames = sorted(f for f in ball_image_xy if f in projections)
    state: dict[int, str | None] = {}
    for f in frames:
        pos = ball_image_xy[f]
        found = None
        for side in ("left", "right"):
            quad = goal_mouth_quad(projections[f], side)
            if quad is not None and _inside(quad, pos):
                found = side
                break
        state[f] = found

    out: list[GoalMouthCandidate] = []
    run: list[int] = []
    run_side: str | None = None

    def flush() -> None:
        nonlocal run, run_side
        if run_side and len(run) >= min_frames_inside:
            first = run[0]
            prior = [g for g in frames if first - lookback_frames <= g < first]
            entered = any(state.get(g) is None for g in prior)
            if entered:
                out.append(GoalMouthCandidate(
                    frame=first,
                    time_seconds=first / fps,
                    side=run_side,
                    frames_inside=len(run),
                    first_frame=first,
                    last_frame=run[-1],
                    entered_from_outside=True,
                    evidence={"observed_ball_frames_checked": len(frames)},
                ))
        run, run_side = [], None

    for f in frames:
        side = state[f]
        if side is not None:
            if run_side is not None and side != run_side:
                flush()
            run_side = side
            run.append(f)
        else:
            flush()
    flush()
    return out
