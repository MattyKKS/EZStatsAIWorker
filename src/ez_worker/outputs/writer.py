from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from ez_worker.schemas import AnalysisArtifacts, Event, TrackObservation, TrackStats, VideoMeta


def write_artifacts(artifacts: AnalysisArtifacts, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(output_dir / "video_meta.json", artifacts.video.model_dump(mode="json"))
    _write_json(
        output_dir / "tracks.json",
        [track.model_dump(mode="json") for track in artifacts.tracks],
    )
    _write_json(
        output_dir / "events.json",
        [event.model_dump(mode="json") for event in artifacts.events],
    )
    _write_json(
        output_dir / "player_stats.json",
        [stat.model_dump(mode="json") for stat in artifacts.stats],
    )
    _write_json(output_dir / "summary.json", _build_summary(artifacts))
    _write_json(output_dir / "match_report.json", _build_match_report(artifacts, output_dir))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_match_report(artifacts: AnalysisArtifacts, output_dir: Path) -> dict:
    """
    Consolidated dashboard-ready JSON used by viewer.html and the future frontend.
    """
    # Resolve team_id per track: most common team_id seen across all observations
    track_team: dict[int, int] = {}
    team_votes: dict[int, Counter] = defaultdict(Counter)
    for obs in artifacts.tracks:
        if obs.team_id is not None:
            team_votes[obs.track_id][obs.team_id] += 1
    for tid, votes in team_votes.items():
        track_team[tid] = votes.most_common(1)[0][0]

    # Possession from touch events (count touches per team)
    team_touch_counts: Counter = Counter()
    for evt in artifacts.events:
        if evt.event_type == "touch" and evt.actor_track_id is not None:
            t = track_team.get(evt.actor_track_id)
            if t is not None:
                team_touch_counts[t] += 1
    total_touches = sum(team_touch_counts.values()) or 1
    possession: dict[str, float] = {
        str(tid): round(count / total_touches * 100, 1)
        for tid, count in sorted(team_touch_counts.items())
    }

    # Player rows
    players = []
    for stat in artifacts.stats:
        players.append({
            "track_id":    stat.track_id,
            "team_id":     track_team.get(stat.track_id),
            "touches":     stat.touch_count,
            "passes":      stat.pass_count,
            "shots":       stat.shot_count,
            "distance_px": round(stat.approx_distance_px, 1),
        })

    # Events
    events = [
        {
            "type":     e.event_type,
            "frame":    e.frame_index,
            "time_s":   e.time_seconds,
            "actor":    e.actor_track_id,
            "target":   e.target_track_id,
            "details":  e.details,
        }
        for e in artifacts.events
    ]

    # Pass network: aggregate PASS events into directed edge counts
    edge_counts: Counter = Counter()
    for e in artifacts.events:
        if e.event_type == "pass" and e.actor_track_id is not None and e.target_track_id is not None:
            edge_counts[(e.actor_track_id, e.target_track_id)] += 1
    pass_network = {
        "nodes": [
            {"id": p["track_id"], "team_id": p["team_id"]}
            for p in players
        ],
        "edges": [
            {"from": src, "to": dst, "count": cnt}
            for (src, dst), cnt in sorted(edge_counts.items(), key=lambda x: -x[1])
        ],
    }

    # Summary counts
    type_counts: Counter = Counter(e.event_type for e in artifacts.events)

    return {
        "match_id":     output_dir.name,
        "video":        str(artifacts.video.path.name),
        "duration_s":   round(artifacts.video.frame_count / artifacts.video.fps, 1),
        "fps":          artifacts.video.fps,
        "possession":   possession,
        "players":      players,
        "events":       events,
        "pass_network": pass_network,
        "summary": {
            "total_touches":      type_counts.get("touch", 0),
            "total_passes":       type_counts.get("pass", 0),
            "total_interceptions": type_counts.get("interception", 0),
            "total_shots":        type_counts.get("shot_attempt", 0),
            "total_goals":        type_counts.get("goal", 0),
        },
    }


def _build_summary(artifacts: AnalysisArtifacts) -> dict:
    return {
        "video_path": str(artifacts.video.path),
        "fps": artifacts.video.fps,
        "frame_count": artifacts.video.frame_count,
        "track_count": len(artifacts.tracks),
        "event_count": len(artifacts.events),
        "player_count": len(artifacts.stats),
        "processed_video_path": (
            str(artifacts.processed_video_path) if artifacts.processed_video_path else None
        ),
    }
