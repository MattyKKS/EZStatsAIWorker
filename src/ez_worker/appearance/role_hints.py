from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def build_role_hints(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    player_stats_path = run_dir / "player_stats_with_teams.json"
    tracks_with_teams_path = run_dir / "tracks_with_teams.json"
    spatial_report_path = run_dir / "spatial_report.json"

    if not player_stats_path.exists():
        raise FileNotFoundError(
            f"Team-aware player stats not found: {player_stats_path}. "
            "Run 'ez-worker apply-team-clusters --run-dir ...' first."
        )
    if not tracks_with_teams_path.exists():
        raise FileNotFoundError(
            f"Team-aware tracks not found: {tracks_with_teams_path}. "
            "Run 'ez-worker apply-team-clusters --run-dir ...' first."
        )
    if not spatial_report_path.exists():
        raise FileNotFoundError(
            f"Spatial report not found: {spatial_report_path}. "
            "Run 'ez-worker build-spatial-report --run-dir ...' first."
        )

    player_stats = json.loads(player_stats_path.read_text(encoding="utf-8"))
    tracks_with_teams = json.loads(tracks_with_teams_path.read_text(encoding="utf-8"))
    spatial_report = json.loads(spatial_report_path.read_text(encoding="utf-8"))
    source_votes = _build_source_votes(tracks_with_teams)

    player_positions = {
        int(row["track_id"]): row for row in spatial_report.get("player_average_positions", [])
    }
    team_centroids = {
        row["team_id"]: row["average_pitch_xy_m"]
        for row in spatial_report.get("team_average_positions", [])
        if row.get("average_pitch_xy_m") is not None
    }

    role_rows = []
    by_team: dict[int | None, list[dict]] = defaultdict(list)
    for stat in player_stats:
        by_team[stat.get("team_id")].append(stat)

    goalkeeper_ids = _goalkeeper_candidates(
        by_team,
        player_positions,
        team_centroids,
        source_votes,
    )

    for stat in player_stats:
        track_id = int(stat["track_id"])
        team_id = stat.get("team_id")
        position_row = player_positions.get(track_id)
        avg_xy = position_row.get("average_pitch_xy_m") if position_row else None
        dominant_source_label = _dominant_source_label(source_votes.get(track_id))

        role_label = "player"
        confidence = "low"
        reasons: list[str] = []

        if track_id in goalkeeper_ids:
            role_label = "goalkeeper_candidate"
            confidence = "medium"
            if dominant_source_label == "goalkeeper":
                confidence = "high"
                reasons.append("detector class votes lean strongly toward goalkeeper")
            reasons.append("extreme team-depth position compared with own team centroid")
        elif dominant_source_label == "referee":
            role_label = "referee_candidate"
            confidence = "high" if team_id is None else "medium"
            reasons.append("detector class votes lean strongly toward referee")
            if team_id is not None:
                reasons.append("appearance clustering likely attached referee to a nearby team group")
        elif team_id is None:
            role_label = _null_team_role(avg_xy, dominant_source_label)
            confidence = "medium" if role_label == "referee_candidate" else "low"
            if role_label == "referee_candidate":
                reasons.append("outlier appearance group with central roaming position")
            else:
                reasons.append("outlier appearance group without clear team assignment")

        role_rows.append(
            {
                "track_id": track_id,
                "team_id": team_id,
                "role_hint": role_label,
                "confidence": confidence,
                "average_pitch_xy_m": avg_xy,
                "dominant_source_label": dominant_source_label,
                "frame_count": int(stat.get("frame_count", 0)),
                "avg_speed_px_per_frame": float(stat.get("avg_speed_px_per_frame", 0.0)),
                "reasons": reasons,
            }
        )

    enriched_stats = []
    role_by_track = {row["track_id"]: row for row in role_rows}
    for stat in player_stats:
        enriched = dict(stat)
        enriched["role_hint"] = role_by_track[int(stat["track_id"])]["role_hint"]
        enriched["role_hint_confidence"] = role_by_track[int(stat["track_id"])]["confidence"]
        enriched_stats.append(enriched)

    payload = {
        "run_dir": str(run_dir),
        "method": "heuristic-role-hints-v2",
        "notes": [
            "This file is a conservative hint layer, not a final classifier.",
            "It follows the next-step plan after team grouping by isolating referee and goalkeeper confusion safely.",
            "Original detector class votes are preserved and used as a hint signal for goalkeeper and referee separation.",
        ],
        "roles": role_rows,
    }

    (run_dir / "player_stats_with_roles.json").write_text(
        json.dumps(enriched_stats, indent=2),
        encoding="utf-8",
    )
    output_path = run_dir / "role_hints.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _goalkeeper_candidates(
    by_team: dict[int | None, list[dict]],
    player_positions: dict[int, dict],
    team_centroids: dict[int | None, list[float]],
    source_votes: dict[int, Counter],
) -> set[int]:
    goalkeeper_ids: set[int] = set()
    for team_id, members in by_team.items():
        if team_id is None or len(members) < 5:
            continue
        centroid = team_centroids.get(team_id)
        if centroid is None:
            continue

        scored: list[tuple[float, int]] = []
        for member in members:
            track_id = int(member["track_id"])
            position_row = player_positions.get(track_id)
            if not position_row:
                continue
            avg_xy = position_row.get("average_pitch_xy_m")
            if avg_xy is None:
                continue
            source_label = _dominant_source_label(source_votes.get(track_id))
            depth_distance = abs(float(avg_xy[0]) - float(centroid[0]))
            width_distance = abs(float(avg_xy[1]) - float(centroid[1]))
            speed = float(member.get("avg_speed_px_per_frame", 0.0))
            source_bonus = 0.0
            if source_label == "goalkeeper":
                source_bonus = 6.0
            elif source_label == "referee":
                source_bonus = -6.0
            score = depth_distance - (0.15 * width_distance) - (0.5 * speed) + source_bonus
            scored.append((score, track_id))

        if not scored:
            continue

        scored.sort(reverse=True)
        best_score, best_track_id = scored[0]
        if best_score >= 10.0:
            goalkeeper_ids.add(best_track_id)
    return goalkeeper_ids


def _null_team_role(avg_xy: list[float] | None, dominant_source_label: str | None) -> str:
    if dominant_source_label == "goalkeeper":
        return "outlier_candidate"
    if dominant_source_label == "referee":
        return "referee_candidate"
    if avg_xy is None:
        return "outlier_candidate"
    x, y = float(avg_xy[0]), float(avg_xy[1])
    if 35.0 <= x <= 70.0 and 10.0 <= y <= 58.0:
        return "referee_candidate"
    return "outlier_candidate"


def _build_source_votes(tracks: list[dict]) -> dict[int, Counter]:
    votes: dict[int, Counter] = defaultdict(Counter)
    for row in tracks:
        label = row.get("label")
        if label != "player":
            continue
        track_id = int(row["track_id"])
        source_label = str(row.get("source_label") or label).lower()
        votes[track_id][source_label] += 1
    return votes


def _dominant_source_label(counter: Counter | None) -> str | None:
    if not counter:
        return None
    return counter.most_common(1)[0][0]
