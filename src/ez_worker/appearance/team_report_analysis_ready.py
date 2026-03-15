from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def build_analysis_ready_team_report(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    assignments_path = run_dir / "team_assignments.json"
    stats_path = run_dir / "player_stats_analysis_ready.json"
    events_path = run_dir / "events_analysis_ready.json"

    if not assignments_path.exists():
        raise FileNotFoundError(
            f"Team assignments not found: {assignments_path}. "
            "Run 'ez-worker apply-team-clusters --run-dir ...' first."
        )
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Analysis-ready player stats not found: {stats_path}. "
            "Run 'ez-worker build-analysis-ready --run-dir ...' first."
        )
    if not events_path.exists():
        raise FileNotFoundError(
            f"Analysis-ready events not found: {events_path}. "
            "Run 'ez-worker build-analysis-ready-events --run-dir ...' first."
        )

    assignments = json.loads(assignments_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    event_payload = json.loads(events_path.read_text(encoding="utf-8"))
    events = event_payload.get("events", [])

    report = {
        "run_dir": str(run_dir),
        "cluster_method": assignments.get("cluster_method"),
        "cluster_preprocessing": assignments.get("cluster_preprocessing"),
        "source_event_method": event_payload.get("method"),
        "team_rows": _team_rows(stats),
        "event_summary": _event_summary(events),
    }

    output_path = run_dir / "team_report_analysis_ready.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def _team_rows(stats: list[dict]) -> list[dict]:
    by_team: dict[int | None, list[dict]] = defaultdict(list)
    for row in stats:
        by_team[row.get("team_id")].append(row)

    rows: list[dict] = []
    for team_id, members in sorted(by_team.items(), key=lambda item: (item[0] is None, item[0])):
        rows.append(
            {
                "team_id": team_id,
                "player_count": len(members),
                "touch_count": sum(int(member.get("touch_count", 0)) for member in members),
                "pass_count": sum(int(member.get("pass_count", 0)) for member in members),
                "shot_count": sum(int(member.get("shot_count", 0)) for member in members),
                "approx_distance_px": round(
                    sum(float(member.get("approx_distance_px", 0.0)) for member in members),
                    2,
                ),
            }
        )
    return rows


def _event_summary(events: list[dict]) -> dict:
    touches = Counter()
    passes = Counter()
    shots = Counter()

    for event in events:
        event_type = event.get("event_type")
        actor_team_id = event.get("actor_team_id")
        target_team_id = event.get("target_team_id")

        if event_type == "touch":
            touches[actor_team_id] += 1
        elif event_type == "pass":
            passes[(actor_team_id, target_team_id)] += 1
        elif event_type == "shot_attempt":
            shots[actor_team_id] += 1

    return {
        "touches_by_team": [
            {"team_id": team_id, "count": count}
            for team_id, count in sorted(touches.items(), key=lambda item: (item[0] is None, item[0]))
        ],
        "passes_by_team_pair": [
            {"from_team_id": pair[0], "to_team_id": pair[1], "count": count}
            for pair, count in sorted(
                passes.items(),
                key=lambda item: (item[0][0] is None, item[0][0], item[0][1] is None, item[0][1]),
            )
        ],
        "shots_by_team": [
            {"team_id": team_id, "count": count}
            for team_id, count in sorted(shots.items(), key=lambda item: (item[0] is None, item[0]))
        ],
    }
