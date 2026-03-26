from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import cv2

from ez_worker.schemas import Event, TrackObservation, VideoMeta


PLAYER_COLOR = (64, 220, 96)
GOALKEEPER_COLOR = (255, 128, 0)
REFEREE_COLOR = (255, 64, 192)
BALL_COLOR = (0, 165, 255)
EVENT_COLOR = (255, 255, 255)


def render_tracks_video(
    *,
    video: VideoMeta,
    tracks: list[TrackObservation],
    events: list[Event],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for rendering: {video.path}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video.fps,
        (video.width, video.height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to create rendered video at: {output_path}")

    tracks_by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for track in tracks:
        tracks_by_frame[track.frame_index].append(track)
    sticky_source_by_track = _build_sticky_source_labels(tracks)

    events_by_frame: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        events_by_frame[event.frame_index].append(event)

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            for track in tracks_by_frame.get(frame_index, []):
                _draw_track(frame, track, video, sticky_source_by_track)

            for event in events_by_frame.get(frame_index, []):
                _draw_event(frame, event)

            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()


def _draw_track(
    frame,
    track: TrackObservation,
    video: VideoMeta,
    sticky_source_by_track: dict[int, str],
) -> None:
    x1 = int(track.bbox.x1 * video.width)
    y1 = int(track.bbox.y1 * video.height)
    x2 = int(track.bbox.x2 * video.width)
    y2 = int(track.bbox.y2 * video.height)

    if track.label == "ball":
        cx = int(track.bbox.cx * video.width)
        cy = int(track.bbox.cy * video.height)
        radius = max(6, int(max(x2 - x1, y2 - y1) / 2))
        cv2.circle(frame, (cx, cy), radius, BALL_COLOR, 2)
        cv2.putText(
            frame,
            f"ball {track.confidence:.2f}",
            (max(0, cx - 30), max(18, cy - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            BALL_COLOR,
            1,
            cv2.LINE_AA,
        )
        return

    sticky_source = sticky_source_by_track.get(int(track.track_id), track.source_label or track.label)
    track_color = _track_color(track, sticky_source)
    track_role = _track_role_tag(sticky_source)
    line_width = 3 if track_role == "GK" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, line_width)
    cv2.putText(
        frame,
        f"{track.track_id} {track_role}",
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        track_color,
        1,
        cv2.LINE_AA,
    )


def _draw_event(frame, event: Event) -> None:
    label = event.event_type.replace("_", " ")
    cv2.putText(
        frame,
        label,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        EVENT_COLOR,
        2,
        cv2.LINE_AA,
    )


def _track_color(track: TrackObservation, source_label: str | None = None) -> tuple[int, int, int]:
    source_label = (source_label or track.source_label or track.label or "").lower()
    if source_label == "goalkeeper":
        return GOALKEEPER_COLOR
    if source_label == "referee":
        return REFEREE_COLOR
    return PLAYER_COLOR


def _track_role_tag(source_label: str | None) -> str:
    source_label = (source_label or "").lower()
    if source_label == "goalkeeper":
        return "GK"
    if source_label == "referee":
        return "REF"
    return "PLY"


def _build_sticky_source_labels(tracks: list[TrackObservation]) -> dict[int, str]:
    by_track: dict[int, list[TrackObservation]] = defaultdict(list)
    for track in tracks:
        if track.label == "player":
            by_track[int(track.track_id)].append(track)

    goalkeeper_candidates: list[tuple[int, int, float, float]] = []
    for track_id, obs_list in by_track.items():
        source_counts = Counter((obs.source_label or obs.label or "player").lower() for obs in obs_list)
        goalkeeper_votes = int(source_counts.get("goalkeeper", 0))
        referee_votes = int(source_counts.get("referee", 0))
        if goalkeeper_votes >= 2 and referee_votes <= goalkeeper_votes:
            mean_cx = sum(obs.bbox.cx for obs in obs_list) / max(len(obs_list), 1)
            edge_score = abs(mean_cx - 0.5)
            ratio = goalkeeper_votes / max(len(obs_list), 1)
            goalkeeper_candidates.append((track_id, goalkeeper_votes, ratio, edge_score))
    goalkeeper_candidates.sort(key=lambda row: (row[1], row[2], row[3]), reverse=True)
    forced_goalkeeper_ids = {row[0] for row in goalkeeper_candidates[:2]}

    sticky: dict[int, str] = {}
    for track_id, obs_list in by_track.items():
        if track_id in forced_goalkeeper_ids:
            sticky[track_id] = "goalkeeper"
            continue
        source_counts = Counter((obs.source_label or obs.label or "player").lower() for obs in obs_list)
        total = len(obs_list)
        goalkeeper_votes = int(source_counts.get("goalkeeper", 0))
        referee_votes = int(source_counts.get("referee", 0))
        if goalkeeper_votes >= max(2, int(total * 0.08)):
            sticky[track_id] = "goalkeeper"
        elif referee_votes >= max(2, int(total * 0.10)):
            sticky[track_id] = "referee"
        else:
            sticky[track_id] = "player"
    return sticky
