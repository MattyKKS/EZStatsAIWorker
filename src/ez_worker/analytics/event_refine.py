from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def build_analysis_ready_events(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    events_path = run_dir / "events.json"
    analysis_ready_path = run_dir / "analysis_ready_summary.json"
    player_stats_path = run_dir / "player_stats_analysis_ready.json"

    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")
    if not analysis_ready_path.exists():
        raise FileNotFoundError(
            f"Analysis-ready summary not found: {analysis_ready_path}. "
            "Run 'ez-worker build-analysis-ready --run-dir ...' first."
        )
    if not player_stats_path.exists():
        raise FileNotFoundError(
            f"Analysis-ready player stats not found: {player_stats_path}. "
            "Run 'ez-worker build-analysis-ready --run-dir ...' first."
        )

    events = json.loads(events_path.read_text(encoding="utf-8"))
    analysis_ready = json.loads(analysis_ready_path.read_text(encoding="utf-8"))
    player_stats = json.loads(player_stats_path.read_text(encoding="utf-8"))

    removed_track_ids = {int(track_id) for track_id in analysis_ready.get("removed_track_ids", [])}
    stats_by_track = {int(row["track_id"]): row for row in player_stats}

    refined_events = []
    dropped_events = []
    for event in events:
        actor_track_id = event.get("actor_track_id")
        target_track_id = event.get("target_track_id")
        if actor_track_id in removed_track_ids or target_track_id in removed_track_ids:
            dropped_events.append(event)
            continue

        enriched = dict(event)
        actor_row = stats_by_track.get(int(actor_track_id)) if actor_track_id is not None else None
        target_row = stats_by_track.get(int(target_track_id)) if target_track_id is not None else None
        enriched["actor_team_id"] = actor_row.get("team_id") if actor_row else None
        enriched["target_team_id"] = target_row.get("team_id") if target_row else None
        enriched["actor_role_hint"] = actor_row.get("role_hint") if actor_row else None
        enriched["target_role_hint"] = target_row.get("role_hint") if target_row else None
        refined_events.append(enriched)

    payload = {
        "run_dir": str(run_dir),
        "method": "analysis-ready-event-filter-v1",
        "source_event_count": len(events),
        "kept_event_count": len(refined_events),
        "dropped_event_count": len(dropped_events),
        "dropped_event_types": dict(Counter(event["event_type"] for event in dropped_events)),
        "events": refined_events,
    }

    output_path = run_dir / "events_analysis_ready.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
