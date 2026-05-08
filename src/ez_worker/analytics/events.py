from __future__ import annotations

import math
from collections import defaultdict

from ez_worker.schemas import Event, TrackObservation, VideoMeta


def detect_events(
    tracks: list[TrackObservation],
    video: VideoMeta,
    ball_track_id: int,
    possession_distance_threshold_px: float,
    possession_min_consecutive_frames: int,
    ball_direction_change_min_deg: float = 25.0,
    shot_min_speed_px_per_frame: float = 25.0,
    shot_no_catch_frames: int = 20,
) -> list[Event]:
    by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for obs in tracks:
        by_frame[obs.frame_index].append(obs)

    sorted_frames = sorted(by_frame)
    ball_vels = _compute_ball_velocities(by_frame, sorted_frames, ball_track_id, video)

    last_confirmed_owner: int | None = None
    candidate_owner: int | None = None
    candidate_count = 0

    vel_history: list[tuple[float, float]] = []
    VEL_WINDOW = 5

    pending_shot: tuple[int, int] | None = None  # (frame_index, actor_track_id)

    events: list[Event] = []

    for frame_index in sorted_frames:
        frame_obs = by_frame[frame_index]

        ball = next(
            (o for o in frame_obs if o.label == "ball" and o.track_id == ball_track_id),
            None,
        ) or next((o for o in frame_obs if o.label == "ball"), None)

        players = [o for o in frame_obs if o.label == "player"]

        vel = ball_vels.get(frame_index)
        if vel is not None:
            vel_history.append(vel)
            if len(vel_history) > VEL_WINDOW:
                vel_history.pop(0)

        curr_speed = (vel[0] ** 2 + vel[1] ** 2) ** 0.5 if vel else 0.0

        # Confirm pending shot if no player caught the ball within the watch window
        if pending_shot is not None:
            shot_f, shot_actor = pending_shot
            if frame_index - shot_f >= shot_no_catch_frames:
                events.append(
                    Event(
                        frame_index=shot_f,
                        time_seconds=shot_f / video.fps,
                        event_type="shot_attempt",
                        actor_track_id=shot_actor,
                        details={"speed_px_per_frame": round(curr_speed, 2)},
                    )
                )
                pending_shot = None

        if ball is None or not players:
            continue

        closest = min(players, key=lambda p: _distance_px(p, ball, video))
        closest_dist = _distance_px(closest, ball, video)

        if closest_dist > possession_distance_threshold_px:
            # Ball away from all players — start pending shot if fast enough
            if (
                last_confirmed_owner is not None
                and curr_speed >= shot_min_speed_px_per_frame
                and pending_shot is None
            ):
                pending_shot = (frame_index, last_confirmed_owner)
            candidate_owner = None
            candidate_count = 0
            continue

        # Ball is near a player — cancel any pending shot (ball was received)
        pending_shot = None

        if candidate_owner == closest.track_id:
            candidate_count += 1
        else:
            candidate_owner = closest.track_id
            candidate_count = 1

        if candidate_count < possession_min_consecutive_frames:
            continue

        # Touch event
        events.append(
            Event(
                frame_index=frame_index,
                time_seconds=frame_index / video.fps,
                event_type="touch",
                actor_track_id=closest.track_id,
                details={"ball_distance_px": round(closest_dist, 2)},
            )
        )

        # Pass: confirmed ownership transfer WITH ball direction change
        if last_confirmed_owner is not None and closest.track_id != last_confirmed_owner:
            dir_change = _direction_change_deg(vel_history)
            if dir_change >= ball_direction_change_min_deg:
                events.append(
                    Event(
                        frame_index=frame_index,
                        time_seconds=frame_index / video.fps,
                        event_type="pass",
                        actor_track_id=last_confirmed_owner,
                        target_track_id=closest.track_id,
                        details={"direction_change_deg": round(dir_change, 1)},
                    )
                )

        last_confirmed_owner = closest.track_id

    # Flush any trailing pending shot at end of video
    if pending_shot is not None:
        shot_f, shot_actor = pending_shot
        events.append(
            Event(
                frame_index=shot_f,
                time_seconds=shot_f / video.fps,
                event_type="shot_attempt",
                actor_track_id=shot_actor,
            )
        )

    return _dedupe_dense_events(events)


def _compute_ball_velocities(
    by_frame: dict[int, list[TrackObservation]],
    sorted_frames: list[int],
    ball_track_id: int,
    video: VideoMeta,
) -> dict[int, tuple[float, float]]:
    vels: dict[int, tuple[float, float]] = {}
    prev_ball: TrackObservation | None = None
    prev_f: int | None = None
    for f in sorted_frames:
        ball = next(
            (o for o in by_frame[f] if o.label == "ball" and o.track_id == ball_track_id),
            None,
        ) or next((o for o in by_frame[f] if o.label == "ball"), None)
        if ball is not None and prev_ball is not None and prev_f is not None:
            gap = max(f - prev_f, 1)
            vx = (ball.bbox.cx - prev_ball.bbox.cx) * video.width / gap
            vy = (ball.bbox.cy - prev_ball.bbox.cy) * video.height / gap
            vels[f] = (vx, vy)
        if ball is not None:
            prev_ball, prev_f = ball, f
    return vels


def _direction_change_deg(vel_history: list[tuple[float, float]]) -> float:
    if len(vel_history) < 2:
        return 0.0
    mid = max(len(vel_history) // 2, 1)
    before = vel_history[:mid]
    after = vel_history[mid:]
    vb = (sum(v[0] for v in before) / len(before), sum(v[1] for v in before) / len(before))
    va = (sum(v[0] for v in after) / len(after), sum(v[1] for v in after) / len(after))
    mag_b = (vb[0] ** 2 + vb[1] ** 2) ** 0.5
    mag_a = (va[0] ** 2 + va[1] ** 2) ** 0.5
    if mag_b < 0.5 or mag_a < 0.5:
        return 0.0
    cos_a = max(-1.0, min(1.0, (vb[0] * va[0] + vb[1] * va[1]) / (mag_b * mag_a)))
    return math.degrees(math.acos(cos_a))


def _distance_px(a: TrackObservation, b: TrackObservation, video: VideoMeta) -> float:
    dx = (a.bbox.cx - b.bbox.cx) * video.width
    dy = (a.bbox.cy - b.bbox.cy) * video.height
    return (dx * dx + dy * dy) ** 0.5


def _dedupe_dense_events(events: list[Event], min_frame_gap: int = 10) -> list[Event]:
    filtered: list[Event] = []
    last_seen: dict[tuple[str, int | None, int | None], int] = {}

    for event in events:
        key = (event.event_type, event.actor_track_id, event.target_track_id)
        last_frame = last_seen.get(key)
        if last_frame is None or event.frame_index - last_frame >= min_frame_gap:
            filtered.append(event)
            last_seen[key] = event.frame_index
    return filtered
