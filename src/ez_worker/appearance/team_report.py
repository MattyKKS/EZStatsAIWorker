from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def build_team_report(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    assignments_path = run_dir / "team_assignments.json"
    stats_path = run_dir / "player_stats_with_teams.json"
    events_path = run_dir / "events.json"

    if not assignments_path.exists():
        raise FileNotFoundError(
            f"Team assignments not found: {assignments_path}. "
            "Run 'ez-worker apply-team-clusters --run-dir ...' first."
        )
    if not stats_path.exists():
        raise FileNotFoundError(f"Team-aware player stats not found: {stats_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Event file not found: {events_path}")

    assignments = json.loads(assignments_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    events = json.loads(events_path.read_text(encoding="utf-8"))

    track_to_team = {int(item["track_id"]): item.get("team_id") for item in stats}
    team_rows = _build_team_rows(stats)
    event_summary = _build_event_summary(events, track_to_team)

    report = {
        "run_dir": str(run_dir),
        "cluster_method": assignments.get("cluster_method"),
        "cluster_preprocessing": assignments.get("cluster_preprocessing"),
        "team_rows": team_rows,
        "event_summary": event_summary,
    }
    output_path = run_dir / "team_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def _build_team_rows(stats: list[dict]) -> list[dict]:
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


def _build_event_summary(events: list[dict], track_to_team: dict[int, int | None]) -> dict:
    touches_by_team: Counter[int | None] = Counter()
    passes_by_pair: Counter[tuple[int | None, int | None]] = Counter()
    shots_by_team: Counter[int | None] = Counter()

    for event in events:
        actor_team = _team_for_track(event.get("actor_track_id"), track_to_team)
        target_team = _team_for_track(event.get("target_track_id"), track_to_team)
        event_type = event.get("event_type")

        if event_type == "touch":
            touches_by_team[actor_team] += 1
        elif event_type == "pass":
            passes_by_pair[(actor_team, target_team)] += 1
        elif event_type == "shot_attempt":
            shots_by_team[actor_team] += 1

    return {
        "touches_by_team": _normalize_counter(touches_by_team),
        "passes_by_team_pair": [
            {
                "from_team_id": from_team,
                "to_team_id": to_team,
                "count": count,
            }
            for (from_team, to_team), count in sorted(
                passes_by_pair.items(),
                key=lambda item: (-item[1], str(item[0])),
            )
        ],
        "shots_by_team": _normalize_counter(shots_by_team),
    }


def _team_for_track(track_id: int | None, track_to_team: dict[int, int | None]) -> int | None:
    if track_id is None:
        return None
    return track_to_team.get(int(track_id))


def _normalize_counter(counter: Counter[int | None]) -> list[dict]:
    return [
        {
            "team_id": team_id,
            "count": count,
        }
        for team_id, count in sorted(counter.items(), key=lambda item: (item[0] is None, item[0]))
    ]
