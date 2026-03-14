from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2

from ez_worker.schemas import TrackObservation, VideoMeta


def export_player_crops(
    video: VideoMeta,
    tracks: list[TrackObservation],
    output_dir: Path,
    *,
    max_crops_per_track: int,
) -> int:
    player_tracks = [track for track in tracks if track.label == "player"]
    if not player_tracks:
        return 0

    by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for track in player_tracks:
        by_frame[track.frame_index].append(track)

    selected_counts: dict[int, int] = defaultdict(int)
    crops_dir = output_dir / "player_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for crop export: {video.path}")

    exported = 0
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_tracks = by_frame.get(frame_index, [])
            for track in frame_tracks:
                if selected_counts[track.track_id] >= max_crops_per_track:
                    continue

                crop = _crop_frame(frame, track, video)
                if crop is None:
                    continue

                track_dir = crops_dir / f"track_{track.track_id:04d}"
                track_dir.mkdir(parents=True, exist_ok=True)
                crop_path = track_dir / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(crop_path), crop)
                selected_counts[track.track_id] += 1
                exported += 1

            frame_index += 1
    finally:
        cap.release()

    return exported


def _crop_frame(frame, track: TrackObservation, video: VideoMeta):
    x1 = max(0, int(track.bbox.x1 * video.width))
    y1 = max(0, int(track.bbox.y1 * video.height))
    x2 = min(video.width, int(track.bbox.x2 * video.width))
    y2 = min(video.height, int(track.bbox.y2 * video.height))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop
