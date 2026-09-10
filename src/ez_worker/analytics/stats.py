from __future__ import annotations

from collections import Counter, defaultdict

from ez_worker.schemas import Event, TrackObservation, TrackStats, VideoMeta


def build_track_stats(
    tracks: list[TrackObservation],
    events: list[Event],
    video: VideoMeta,
) -> list[TrackStats]:
    by_track: dict[int, list[TrackObservation]] = defaultdict(list)
    for obs in tracks:
        if obs.label in ("player", "goalkeeper") and obs.source_label != "referee":
            by_track[obs.track_id].append(obs)

    touch_counts = Counter(event.actor_track_id for event in events if event.event_type == "touch")
    pass_counts = Counter(event.actor_track_id for event in events if event.event_type == "pass")
    shot_counts = Counter(event.actor_track_id for event in events if event.event_type in ("shot", "shot_attempt", "goal"))

    stats: list[TrackStats] = []
    for track_id, obs_list in sorted(by_track.items()):
        obs_list.sort(key=lambda item: item.frame_index)
        distance_px = 0.0
        for prev, cur in zip(obs_list, obs_list[1:]):
            dx = (cur.bbox.cx - prev.bbox.cx) * video.width
            dy = (cur.bbox.cy - prev.bbox.cy) * video.height
            distance_px += (dx * dx + dy * dy) ** 0.5

        frame_count = len(obs_list)
        stats.append(
            TrackStats(
                track_id=track_id,
                label="player",
                frame_count=frame_count,
                approx_distance_px=round(distance_px, 2),
                avg_speed_px_per_frame=round(distance_px / max(frame_count - 1, 1), 2),
                touch_count=touch_counts.get(track_id, 0),
                pass_count=pass_counts.get(track_id, 0),
                shot_count=shot_counts.get(track_id, 0),
            )
        )

    return stats
