from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def build_possession_report(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    events_path = run_dir / "events_analysis_ready.json"

    if not events_path.exists():
        raise FileNotFoundError(
            f"Analysis-ready events not found: {events_path}. "
            "Run 'ez-worker build-analysis-ready-events --run-dir ...' first."
        )

    payload = json.loads(events_path.read_text(encoding="utf-8"))
    events = payload.get("events", [])

    touches_by_team = Counter()
    passes_by_team = Counter()
    timeline = []
    last_team_id = None
    possession_segments: list[dict] = []
    segment_start_time = None
    segment_event_count = 0

    for event in events:
        event_type = event.get("event_type")
        actor_team_id = event.get("actor_team_id")
        time_seconds = float(event.get("time_seconds", 0.0))

        if event_type == "touch":
            touches_by_team[actor_team_id] += 1
        if event_type == "pass":
            passes_by_team[actor_team_id] += 1

        timeline.append(
            {
                "time_seconds": round(time_seconds, 2),
                "event_type": event_type,
                "actor_track_id": event.get("actor_track_id"),
                "actor_team_id": actor_team_id,
                "target_track_id": event.get("target_track_id"),
                "target_team_id": event.get("target_team_id"),
            }
        )

        if actor_team_id is None:
            continue

        if last_team_id is None:
            last_team_id = actor_team_id
            segment_start_time = time_seconds
            segment_event_count = 1
            continue

        if actor_team_id == last_team_id:
            segment_event_count += 1
            continue

        possession_segments.append(
            {
                "team_id": last_team_id,
                "start_time_seconds": round(segment_start_time or 0.0, 2),
                "end_time_seconds": round(time_seconds, 2),
                "event_count": segment_event_count,
            }
        )
        last_team_id = actor_team_id
        segment_start_time = time_seconds
        segment_event_count = 1

    if last_team_id is not None and segment_start_time is not None:
        end_time = float(timeline[-1]["time_seconds"]) if timeline else segment_start_time
        possession_segments.append(
            {
                "team_id": last_team_id,
                "start_time_seconds": round(segment_start_time, 2),
                "end_time_seconds": round(end_time, 2),
                "event_count": segment_event_count,
            }
        )

    report = {
        "run_dir": str(run_dir),
        "source_event_method": payload.get("method"),
        "event_count": len(events),
        "touches_by_team": [
            {"team_id": team_id, "count": count}
            for team_id, count in sorted(touches_by_team.items(), key=lambda item: (item[0] is None, item[0]))
        ],
        "passes_by_team": [
            {"team_id": team_id, "count": count}
            for team_id, count in sorted(passes_by_team.items(), key=lambda item: (item[0] is None, item[0]))
        ],
        "possession_segments": possession_segments,
        "timeline": timeline,
    }

    output_path = run_dir / "possession_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path
