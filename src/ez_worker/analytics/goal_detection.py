"""Geometric goal detection: did the ball cross a goal line between the posts?

Why this exists
---------------
Measured on the Messi development windows, the event output scored precision 1.00
and recall 0.50 — every event it reported was correct, but it found half of them.
The entire deficit was `shot_attempt` and `goal`, because no component produced
those classes at all. As the v3 handoff puts it: "no goal confirmation stage:
threshold tuning or detector retraining cannot supply that missing capability."

A goal is the most rule-shaped event in football — the ball wholly crosses the
goal line between the posts and under the bar — so it needs geometry, not a model.
What it needs is a trustworthy ball position on the field, which only became
available with per-frame, quality-gated projection (`FieldProjector`). The earlier
single-homography path could not supply it.

Honest limitation, stated up front
----------------------------------
A homography maps the GROUND PLANE. A ball in the air projects to where its line
of sight meets the grass, which is *beyond* its true position when it travels away
from the camera. So an airborne ball near the goal can project past the goal line
without having crossed it, and a ball rolling in can be projected short.

This detector therefore does not claim to decide goals. It emits *candidates* with
the evidence attached (how far past the line, for how many frames, how central,
whether the ball approached from the field) so a human can confirm quickly, and it
prefers to miss a goal over inventing one. `requires_review` is always True.

It also cannot see the net, so a ball passing behind the goal at the same ground
projection looks identical. The "approached from inside the field" gate is what
separates most of those cases, not certainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Goal mouth, from the Laws of the Game: 7.32 m between the posts.
GOAL_WIDTH_CM = 732.0


@dataclass
class GoalCandidate:
    frame: int
    time_seconds: float
    side: str                      # "left" (x<0) or "right" (x>length)
    depth_past_line_cm: float      # how far beyond the line, at its deepest
    frames_past_line: int
    offset_from_centre_cm: float   # |y - goal centre| at crossing
    approached_from_field: bool
    evidence: dict = field(default_factory=dict)
    requires_review: bool = True

    def as_event(self) -> dict:
        return {
            "type": "goal_candidate",
            "frame": self.frame,
            "time_s": round(self.time_seconds, 2),
            "actor": None,
            "actor_label": None,
            "target": None,
            "target_label": None,
            "details": {
                "side": self.side,
                "depth_past_line_cm": round(self.depth_past_line_cm, 1),
                "frames_past_line": self.frames_past_line,
                "offset_from_centre_cm": round(self.offset_from_centre_cm, 1),
                "approached_from_field": self.approached_from_field,
                "requires_review": True,
                **self.evidence,
            },
        }


def _pitch_bounds(vertices) -> tuple[float, float]:
    xs = [float(v[0]) for v in vertices]
    ys = [float(v[1]) for v in vertices]
    return max(xs) - min(xs), max(ys) - min(ys)


def find_goal_candidates(
    ball_field_positions: dict[int, tuple[float, float]],
    vertices,
    fps: float,
    *,
    min_frames_past_line: int = 2,
    max_depth_cm: float = 1500.0,
    approach_lookback_frames: int = 50,
) -> list[GoalCandidate]:
    """Ball positions already projected to the field -> goal candidates.

    `ball_field_positions` must contain ONLY frames whose projection passed the
    quality gate (FieldProjector returns None otherwise) and whose ball position
    was actually observed — an interpolated ball must not be allowed to invent a
    goal, because interpolation across the very gap a shot creates is exactly
    where a false crossing would come from.

    Gates:
      * `min_frames_past_line` — one frame past the line is projection noise on a
        ball travelling at speed. Requiring persistence costs little, since a ball
        in the net stays there.
      * `max_depth_cm` — a projection hundreds of metres past the line is a broken
        homography, not a goal.
      * the crossing must be between the posts, and the ball must have been inside
        the field within `approach_lookback_frames`, so a ball already dead behind
        the goal does not register.
    """
    if not ball_field_positions or fps <= 0:
        return []

    length, width = _pitch_bounds(vertices)
    centre_y = width / 2.0
    half_mouth = GOAL_WIDTH_CM / 2.0

    frames = sorted(ball_field_positions)
    out: list[GoalCandidate] = []

    run: list[int] = []
    run_side: str | None = None

    def flush() -> None:
        nonlocal run, run_side
        if run_side and len(run) >= min_frames_past_line:
            depths, offsets = [], []
            for f in run:
                x, y = ball_field_positions[f]
                depths.append(-x if run_side == "left" else x - length)
                offsets.append(abs(y - centre_y))
            deepest = max(depths)
            if deepest <= max_depth_cm:
                first = run[0]
                approached = any(
                    0 <= ball_field_positions[g][0] <= length
                    for g in frames
                    if first - approach_lookback_frames <= g < first
                )
                if approached:
                    out.append(GoalCandidate(
                        frame=first,
                        time_seconds=first / fps,
                        side=run_side,
                        depth_past_line_cm=deepest,
                        frames_past_line=len(run),
                        offset_from_centre_cm=min(offsets),
                        approached_from_field=True,
                        evidence={
                            "goal_width_cm": GOAL_WIDTH_CM,
                            "pitch_length_cm": round(length, 1),
                            "last_frame_past_line": run[-1],
                        },
                    ))
        run, run_side = [], None

    for f in frames:
        x, y = ball_field_positions[f]
        side = "left" if x < 0 else ("right" if x > length else None)
        between_posts = abs(y - centre_y) <= half_mouth
        if side is not None and between_posts:
            if run_side is not None and side != run_side:
                flush()
            run_side = side
            run.append(f)
        else:
            flush()
    flush()

    return out
