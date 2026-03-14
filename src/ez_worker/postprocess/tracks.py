from __future__ import annotations

from collections import defaultdict

from ez_worker.schemas import TrackObservation, VideoMeta


def cleanup_tracks(
    tracks: list[TrackObservation],
    video: VideoMeta,
    *,
    max_players_per_frame: int,
    max_unique_players: int,
    min_player_track_frames: int,
    merge_tracklets: bool,
    max_track_gap_frames: int,
    max_track_merge_distance_px: float,
    interpolate_gaps: bool,
    frame_step: int,
    max_interpolation_gap_frames: int,
    clean_ball_path: bool,
    max_ball_jump_px: float,
    ball_reset_gap_frames: int,
    ball_reset_confidence: float,
    drop_ambiguous_ball_frames: bool,
) -> list[TrackObservation]:
    cleaned = limit_players_per_frame(tracks, max_players_per_frame=max_players_per_frame)
    cleaned = limit_ball_per_frame(cleaned, drop_ambiguous_frames=drop_ambiguous_ball_frames)
    if clean_ball_path:
        cleaned = clean_ball_detections(
            cleaned,
            video,
            frame_step=frame_step,
            max_ball_jump_px=max_ball_jump_px,
            ball_reset_gap_frames=ball_reset_gap_frames,
            ball_reset_confidence=ball_reset_confidence,
        )
    if merge_tracklets:
        cleaned = merge_player_tracklets(
            cleaned,
            video,
            max_gap_frames=max_track_gap_frames,
            max_merge_distance_px=max_track_merge_distance_px,
        )
    cleaned = keep_best_player_tracks(
        cleaned,
        max_unique_players=max_unique_players,
        min_player_track_frames=min_player_track_frames,
    )
    if interpolate_gaps:
        cleaned = interpolate_short_gaps(
            cleaned,
            frame_step=frame_step,
            max_gap_frames=max_interpolation_gap_frames,
        )
    return cleaned


def limit_players_per_frame(
    tracks: list[TrackObservation],
    *,
    max_players_per_frame: int,
) -> list[TrackObservation]:
    by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for track in tracks:
        by_frame[track.frame_index].append(track)

    limited: list[TrackObservation] = []
    for frame_index in sorted(by_frame):
        frame_tracks = by_frame[frame_index]
        players = [track for track in frame_tracks if track.label == "player"]
        non_players = [track for track in frame_tracks if track.label != "player"]
        players.sort(key=lambda track: track.confidence, reverse=True)
        limited.extend(players[:max_players_per_frame])
        limited.extend(non_players)
    return sorted(limited, key=lambda track: (track.frame_index, track.label, track.track_id))


def limit_ball_per_frame(
    tracks: list[TrackObservation],
    *,
    drop_ambiguous_frames: bool,
) -> list[TrackObservation]:
    by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for track in tracks:
        by_frame[track.frame_index].append(track)

    limited: list[TrackObservation] = []
    for frame_index in sorted(by_frame):
        frame_tracks = by_frame[frame_index]
        balls = [track for track in frame_tracks if track.label == "ball"]
        non_balls = [track for track in frame_tracks if track.label != "ball"]
        if balls:
            if drop_ambiguous_frames and len(balls) > 1:
                limited.extend(non_balls)
                continue
            balls.sort(key=lambda track: track.confidence, reverse=True)
            limited.append(balls[0])
        limited.extend(non_balls)
    return sorted(limited, key=lambda track: (track.frame_index, track.label, track.track_id))


def clean_ball_detections(
    tracks: list[TrackObservation],
    video: VideoMeta,
    *,
    frame_step: int,
    max_ball_jump_px: float,
    ball_reset_gap_frames: int,
    ball_reset_confidence: float,
) -> list[TrackObservation]:
    balls = [track for track in tracks if track.label == "ball"]
    non_balls = [track for track in tracks if track.label != "ball"]
    if not balls:
        return tracks

    balls_by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for ball in balls:
        balls_by_frame[ball.frame_index].append(ball.model_copy(update={"track_id": 0}))

    cleaned_balls: list[TrackObservation] = []
    previous_ball: TrackObservation | None = None
    last_ball: TrackObservation | None = None
    for frame_index in sorted(balls_by_frame):
        frame_candidates = sorted(
            balls_by_frame[frame_index],
            key=lambda track: track.confidence,
            reverse=True,
        )
        if last_ball is None:
            selected = frame_candidates[0]
            cleaned_balls.append(selected)
            last_ball = selected
            continue

        gap_frames = frame_index - last_ball.frame_index
        allowed_jump_px = max_ball_jump_px * max(1.0, gap_frames / max(frame_step, 1))
        selected = _select_best_ball_candidate(
            frame_candidates,
            previous_ball=previous_ball,
            last_ball=last_ball,
            video=video,
            allowed_jump_px=allowed_jump_px,
        )
        if selected is None:
            if gap_frames >= ball_reset_gap_frames:
                reset_candidate = frame_candidates[0]
                if reset_candidate.confidence >= ball_reset_confidence:
                    selected = reset_candidate
                else:
                    continue
            else:
                continue

        cleaned_balls.append(selected)
        previous_ball = last_ball
        last_ball = selected

    return sorted(non_balls + cleaned_balls, key=lambda track: (track.frame_index, track.label, track.track_id))


def _select_best_ball_candidate(
    candidates: list[TrackObservation],
    *,
    previous_ball: TrackObservation | None,
    last_ball: TrackObservation,
    video: VideoMeta,
    allowed_jump_px: float,
) -> TrackObservation | None:
    ranked: list[tuple[float, float, float, TrackObservation]] = []
    predicted_cx, predicted_cy = _predict_ball_center(previous_ball, last_ball)
    for candidate in candidates:
        dx = (candidate.bbox.cx - predicted_cx) * video.width
        dy = (candidate.bbox.cy - predicted_cy) * video.height
        predicted_distance_px = (dx * dx + dy * dy) ** 0.5
        distance_from_last_px = _center_distance_px(last_ball, candidate, video)
        if min(predicted_distance_px, distance_from_last_px) > allowed_jump_px:
            continue
        ranked.append((predicted_distance_px, distance_from_last_px, -candidate.confidence, candidate))

    if not ranked:
        return None

    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked[0][3]


def _predict_ball_center(
    previous_ball: TrackObservation | None,
    last_ball: TrackObservation,
) -> tuple[float, float]:
    if previous_ball is None:
        return last_ball.bbox.cx, last_ball.bbox.cy

    frame_delta = last_ball.frame_index - previous_ball.frame_index
    if frame_delta <= 0:
        return last_ball.bbox.cx, last_ball.bbox.cy

    vx = (last_ball.bbox.cx - previous_ball.bbox.cx) / frame_delta
    vy = (last_ball.bbox.cy - previous_ball.bbox.cy) / frame_delta
    return last_ball.bbox.cx + vx, last_ball.bbox.cy + vy


def keep_best_player_tracks(
    tracks: list[TrackObservation],
    *,
    max_unique_players: int,
    min_player_track_frames: int,
) -> list[TrackObservation]:
    players_by_id: dict[int, list[TrackObservation]] = defaultdict(list)
    other_tracks: list[TrackObservation] = []
    for track in tracks:
        if track.label == "player":
            players_by_id[track.track_id].append(track)
        else:
            other_tracks.append(track)

    scored_ids = []
    for track_id, obs_list in players_by_id.items():
        frame_count = len(obs_list)
        avg_confidence = sum(obs.confidence for obs in obs_list) / max(frame_count, 1)
        scored_ids.append((track_id, frame_count, avg_confidence))

    scored_ids = [item for item in scored_ids if item[1] >= min_player_track_frames]
    scored_ids.sort(key=lambda item: (item[1], item[2]), reverse=True)
    keep_ids = {track_id for track_id, _, _ in scored_ids[:max_unique_players]}

    kept_players = [
        track
        for track in tracks
        if track.label == "player" and track.track_id in keep_ids
    ]
    return sorted(kept_players + other_tracks, key=lambda track: (track.frame_index, track.label, track.track_id))


def interpolate_short_gaps(
    tracks: list[TrackObservation],
    *,
    frame_step: int,
    max_gap_frames: int,
) -> list[TrackObservation]:
    players_by_id: dict[int, list[TrackObservation]] = defaultdict(list)
    balls: list[TrackObservation] = []
    others: list[TrackObservation] = []

    for track in tracks:
        if track.label == "player":
            players_by_id[track.track_id].append(track)
        elif track.label == "ball":
            balls.append(track)
        else:
            others.append(track)

    interpolated: list[TrackObservation] = list(others)
    for track_id, obs_list in players_by_id.items():
        interpolated.extend(_interpolate_sequence(sorted(obs_list, key=lambda item: item.frame_index), frame_step, max_gap_frames))

    if balls:
        best_ball_by_frame: dict[int, TrackObservation] = {}
        for ball in balls:
            current = best_ball_by_frame.get(ball.frame_index)
            if current is None or ball.confidence > current.confidence:
                best_ball_by_frame[ball.frame_index] = ball
        ball_sequence = [best_ball_by_frame[idx] for idx in sorted(best_ball_by_frame)]
        interpolated.extend(_interpolate_sequence(ball_sequence, frame_step, max_gap_frames, normalize_track_id=True))

    deduped: dict[tuple[int, int, str], TrackObservation] = {}
    for track in interpolated:
        key = (track.frame_index, track.track_id, track.label)
        current = deduped.get(key)
        if current is None or track.confidence > current.confidence:
            deduped[key] = track
    return sorted(deduped.values(), key=lambda track: (track.frame_index, track.label, track.track_id))


def _interpolate_sequence(
    obs_list: list[TrackObservation],
    frame_step: int,
    max_gap_frames: int,
    normalize_track_id: bool = False,
) -> list[TrackObservation]:
    if not obs_list:
        return []

    sequence = [obs.model_copy(update={"track_id": 0}) if normalize_track_id else obs for obs in obs_list]
    output: list[TrackObservation] = [sequence[0]]
    for prev, cur in zip(sequence, sequence[1:]):
        gap = cur.frame_index - prev.frame_index
        if gap > frame_step and gap <= max_gap_frames:
            for frame_index in range(prev.frame_index + frame_step, cur.frame_index, frame_step):
                alpha = (frame_index - prev.frame_index) / gap
                output.append(_interpolate_observation(prev, cur, frame_index, alpha))
        output.append(cur)
    return output


def _interpolate_observation(
    prev: TrackObservation,
    cur: TrackObservation,
    frame_index: int,
    alpha: float,
) -> TrackObservation:
    return prev.model_copy(
        update={
            "frame_index": frame_index,
            "confidence": min(prev.confidence, cur.confidence) * 0.9,
            "bbox": prev.bbox.model_copy(
                update={
                    "x1": _lerp(prev.bbox.x1, cur.bbox.x1, alpha),
                    "y1": _lerp(prev.bbox.y1, cur.bbox.y1, alpha),
                    "x2": _lerp(prev.bbox.x2, cur.bbox.x2, alpha),
                    "y2": _lerp(prev.bbox.y2, cur.bbox.y2, alpha),
                }
            ),
        }
    )


def merge_player_tracklets(
    tracks: list[TrackObservation],
    video: VideoMeta,
    *,
    max_gap_frames: int,
    max_merge_distance_px: float,
) -> list[TrackObservation]:
    players_by_id: dict[int, list[TrackObservation]] = defaultdict(list)
    other_tracks: list[TrackObservation] = []
    for track in tracks:
        if track.label == "player":
            players_by_id[track.track_id].append(track)
        else:
            other_tracks.append(track)

    tracklets: list[dict] = []
    for track_id, obs_list in players_by_id.items():
        ordered = sorted(obs_list, key=lambda item: item.frame_index)
        tracklets.append(
            {
                "track_id": track_id,
                "obs": ordered,
                "start": ordered[0].frame_index,
                "end": ordered[-1].frame_index,
                "start_obs": ordered[0],
                "end_obs": ordered[-1],
            }
        )

    tracklets.sort(key=lambda item: item["start"])
    replacements: dict[int, int] = {}

    for index, current in enumerate(tracklets):
        if current["track_id"] in replacements:
            continue

        for candidate in tracklets[index + 1 :]:
            if candidate["track_id"] in replacements:
                continue
            gap = candidate["start"] - current["end"]
            if gap < 0 or gap > max_gap_frames:
                if gap > max_gap_frames:
                    break
                continue

            distance_px = _center_distance_px(current["end_obs"], candidate["start_obs"], video)
            if distance_px <= max_merge_distance_px:
                replacements[candidate["track_id"]] = current["track_id"]
                current["obs"].extend(candidate["obs"])
                current["obs"].sort(key=lambda item: item.frame_index)
                current["end"] = current["obs"][-1].frame_index
                current["end_obs"] = current["obs"][-1]

    merged: list[TrackObservation] = []
    for track in tracks:
        if track.label != "player":
            merged.append(track)
            continue
        target_id = replacements.get(track.track_id, track.track_id)
        if target_id == track.track_id:
            merged.append(track)
        else:
            merged.append(track.model_copy(update={"track_id": target_id}))

    deduped: dict[tuple[int, int, str], TrackObservation] = {}
    for track in merged:
        key = (track.frame_index, track.track_id, track.label)
        current = deduped.get(key)
        if current is None or track.confidence > current.confidence:
            deduped[key] = track
    return sorted(deduped.values(), key=lambda track: (track.frame_index, track.label, track.track_id))


def _center_distance_px(a: TrackObservation, b: TrackObservation, video: VideoMeta) -> float:
    dx = (a.bbox.cx - b.bbox.cx) * video.width
    dy = (a.bbox.cy - b.bbox.cy) * video.height
    return (dx * dx + dy * dy) ** 0.5


def _lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha
