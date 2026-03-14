from __future__ import annotations

from collections import defaultdict

from ez_worker.schemas import Event, TrackObservation, VideoMeta


def detect_events(
    tracks: list[TrackObservation],
    video: VideoMeta,
    ball_track_id: int,
    pass_distance_threshold: float,
    shot_zone_x_threshold: float,
    possession_distance_threshold_px: float,
    possession_min_consecutive_frames: int,
) -> list[Event]:
    by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for obs in tracks:
        by_frame[obs.frame_index].append(obs)

    last_owner: int | None = None
    candidate_owner: int | None = None
    candidate_count = 0
    events: list[Event] = []

    for frame_index in sorted(by_frame):
        frame_obs = by_frame[frame_index]
        ball_candidates = [obs for obs in frame_obs if obs.label == "ball"]
        ball = next((obs for obs in ball_candidates if obs.track_id == ball_track_id), None)
        if ball is None and ball_candidates:
            ball_candidates.sort(key=lambda obs: obs.confidence, reverse=True)
            ball = ball_candidates[0]
        players = [obs for obs in frame_obs if obs.label == "player"]
        if ball is None or not players:
            continue

        owner = min(players, key=lambda p: _distance_px(p, ball, video))
        owner_distance = _distance_px(owner, ball, video)
        if owner_distance > possession_distance_threshold_px:
            candidate_owner = None
            candidate_count = 0
            continue

        if candidate_owner == owner.track_id:
            candidate_count += 1
        else:
            candidate_owner = owner.track_id
            candidate_count = 1

        if candidate_count < possession_min_consecutive_frames:
            continue

        if owner_distance <= pass_distance_threshold / 2:
            events.append(
                Event(
                    frame_index=frame_index,
                    time_seconds=frame_index / video.fps,
                    event_type="touch",
                    actor_track_id=owner.track_id,
                    details={"ball_distance_px": round(owner_distance, 2)},
                )
            )

        if last_owner is not None and owner.track_id != last_owner:
            events.append(
                Event(
                    frame_index=frame_index,
                    time_seconds=frame_index / video.fps,
                    event_type="pass",
                    actor_track_id=last_owner,
                    target_track_id=owner.track_id,
                )
            )

        if ball.bbox.cx >= shot_zone_x_threshold:
            events.append(
                Event(
                    frame_index=frame_index,
                    time_seconds=frame_index / video.fps,
                    event_type="shot_attempt",
                    actor_track_id=owner.track_id,
                )
            )

        last_owner = owner.track_id

    return _dedupe_dense_events(events)


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
