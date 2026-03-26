from __future__ import annotations

from collections import Counter
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
    ball_hold_max_gap_frames: int,
    ball_smoothing_alpha: float,
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
            ball_hold_max_gap_frames=ball_hold_max_gap_frames,
            ball_smoothing_alpha=ball_smoothing_alpha,
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
    cleaned = stabilize_player_source_labels(cleaned)
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
        players.sort(
            key=lambda track: (
                _player_priority(track),
                _touchline_priority(track),
                track.confidence,
            ),
            reverse=True,
        )
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
    ball_hold_max_gap_frames: int,
    ball_smoothing_alpha: float,
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
    for frame_index in range(min(balls_by_frame), max(balls_by_frame) + frame_step, frame_step):
        frame_candidates = sorted(
            balls_by_frame.get(frame_index, []),
            key=lambda track: track.confidence,
            reverse=True,
        )
        if not frame_candidates:
            if last_ball is None:
                continue
            gap_frames = frame_index - last_ball.frame_index
            if 0 < gap_frames <= ball_hold_max_gap_frames:
                held = last_ball.model_copy(
                    update={
                        "frame_index": frame_index,
                        "confidence": max(0.05, last_ball.confidence * 0.55),
                    }
                )
                cleaned_balls.append(held)
                previous_ball = last_ball
                last_ball = held
            continue

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

        if last_ball is not None and ball_smoothing_alpha > 0.0:
            selected = _smooth_ball_observation(last_ball, selected, ball_smoothing_alpha)

        cleaned_balls.append(selected)
        previous_ball = last_ball
        last_ball = selected

    cleaned_balls = _interpolate_ball_short_gaps(
        cleaned_balls,
        frame_step=frame_step,
        max_gap_frames=ball_hold_max_gap_frames,
    )

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


def _smooth_ball_observation(
    previous: TrackObservation,
    current: TrackObservation,
    alpha: float,
) -> TrackObservation:
    alpha = max(0.0, min(1.0, alpha))
    if alpha <= 0.0:
        return current

    def blend(prev_value: float, cur_value: float) -> float:
        return (1.0 - alpha) * cur_value + alpha * prev_value

    return current.model_copy(
        update={
            "bbox": current.bbox.model_copy(
                update={
                    "x1": blend(previous.bbox.x1, current.bbox.x1),
                    "y1": blend(previous.bbox.y1, current.bbox.y1),
                    "x2": blend(previous.bbox.x2, current.bbox.x2),
                    "y2": blend(previous.bbox.y2, current.bbox.y2),
                }
            )
        }
    )


def _interpolate_ball_short_gaps(
    balls: list[TrackObservation],
    *,
    frame_step: int,
    max_gap_frames: int,
) -> list[TrackObservation]:
    if len(balls) <= 1:
        return balls

    ordered = sorted(balls, key=lambda track: track.frame_index)
    output: list[TrackObservation] = [ordered[0]]
    for prev, cur in zip(ordered, ordered[1:]):
        gap = cur.frame_index - prev.frame_index
        if gap > frame_step and gap <= max_gap_frames:
            for frame_index in range(prev.frame_index + frame_step, cur.frame_index, frame_step):
                alpha = (frame_index - prev.frame_index) / max(gap, 1)
                output.append(_interpolate_observation(prev, cur, frame_index, alpha))
        output.append(cur)
    return output


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

    track_infos = []
    for track_id, obs_list in players_by_id.items():
        frame_count = len(obs_list)
        avg_confidence = sum(obs.confidence for obs in obs_list) / max(frame_count, 1)
        avg_priority = sum(_player_priority(obs) for obs in obs_list) / max(frame_count, 1)
        avg_touchline_priority = sum(_touchline_priority(obs) for obs in obs_list) / max(frame_count, 1)
        source_counts = Counter((obs.source_label or obs.label or "").lower() for obs in obs_list)
        dominant_source_label = source_counts.most_common(1)[0][0] if source_counts else "player"
        frame_indexes = {obs.frame_index for obs in obs_list}
        track_infos.append(
            {
                "track_id": track_id,
                "avg_priority": avg_priority,
                "avg_touchline_priority": avg_touchline_priority,
                "frame_count": frame_count,
                "avg_confidence": avg_confidence,
                "dominant_source_label": dominant_source_label,
                "frame_indexes": frame_indexes,
            }
        )

    track_infos = [item for item in track_infos if item["frame_count"] >= min_player_track_frames]
    keep_ids = _select_player_track_ids(track_infos, max_unique_players=max_unique_players)

    # Preserve up to two goalkeeper-like tracks even when global player pruning is aggressive.
    goalkeeper_candidates = [
        item for item in track_infos if item["dominant_source_label"] == "goalkeeper"
    ]
    goalkeeper_candidates.sort(
        key=lambda item: (item["frame_count"], item["avg_confidence"]),
        reverse=True,
    )
    for candidate in goalkeeper_candidates[:2]:
        keep_ids.add(int(candidate["track_id"]))

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


def _player_priority(track: TrackObservation) -> float:
    source_label = (track.source_label or track.label or "").lower()
    if source_label == "goalkeeper":
        return 3.4
    if source_label == "player":
        return 3.0
    if source_label == "referee":
        return 1.0
    return 0.5


def _touchline_priority(track: TrackObservation) -> float:
    width = max(0.0, track.bbox.x2 - track.bbox.x1)
    height = max(0.0, track.bbox.y2 - track.bbox.y1)
    if height <= 0.02 or width <= 0.002:
        return 0.0

    near_left = track.bbox.x1 <= 0.035
    near_right = track.bbox.x2 >= 0.965
    near_touchline = near_left or near_right
    if not near_touchline:
        return 0.0

    # Give a small retention bonus to plausible sideline players so they are not
    # consistently dropped by confidence-only pruning.
    return 0.35


def _select_player_track_ids(
    track_infos: list[dict],
    *,
    max_unique_players: int,
) -> set[int]:
    if len(track_infos) <= max_unique_players:
        return {int(item["track_id"]) for item in track_infos}

    remaining = list(track_infos)
    covered_frames: set[int] = set()
    selected_ids: set[int] = set()

    while remaining and len(selected_ids) < max_unique_players:
        best_item = None
        best_score = None
        for item in remaining:
            new_frames = len(item["frame_indexes"] - covered_frames)
            score = (
                new_frames * 5.0,
                item["avg_priority"] * 2.0 + item["avg_touchline_priority"],
                item["frame_count"],
                item["avg_confidence"],
            )
            if best_score is None or score > best_score:
                best_score = score
                best_item = item

        if best_item is None:
            break

        selected_ids.add(int(best_item["track_id"]))
        covered_frames.update(best_item["frame_indexes"])
        remaining = [item for item in remaining if int(item["track_id"]) != int(best_item["track_id"])]

    return selected_ids


def stabilize_player_source_labels(tracks: list[TrackObservation]) -> list[TrackObservation]:
    players_by_id: dict[int, list[TrackObservation]] = defaultdict(list)
    for track in tracks:
        if track.label == "player":
            players_by_id[int(track.track_id)].append(track)

    goalkeeper_vote_rows: list[tuple[int, int, float, float]] = []
    for track_id, obs_list in players_by_id.items():
        source_counts = Counter((obs.source_label or obs.label or "").lower() for obs in obs_list)
        goalkeeper_votes = int(source_counts.get("goalkeeper", 0))
        referee_votes = int(source_counts.get("referee", 0))
        if goalkeeper_votes < 2 or referee_votes > goalkeeper_votes:
            continue
        mean_cx = sum(obs.bbox.cx for obs in obs_list) / max(len(obs_list), 1)
        edge_score = abs(mean_cx - 0.5)
        gk_ratio = goalkeeper_votes / max(len(obs_list), 1)
        goalkeeper_vote_rows.append((track_id, goalkeeper_votes, gk_ratio, edge_score))

    goalkeeper_vote_rows.sort(key=lambda row: (row[1], row[2], row[3]), reverse=True)
    goalkeeper_track_ids = {row[0] for row in goalkeeper_vote_rows[:2]}
    if not goalkeeper_track_ids:
        return tracks

    stabilized: list[TrackObservation] = []
    for track in tracks:
        if track.label == "player" and int(track.track_id) in goalkeeper_track_ids:
            stabilized.append(track.model_copy(update={"source_label": "goalkeeper"}))
        else:
            stabilized.append(track)
    return stabilized
