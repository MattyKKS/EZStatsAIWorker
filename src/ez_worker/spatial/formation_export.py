from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def build_formation_export(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    heatmap_export_path = run_dir / "heatmap_export.json"
    if not heatmap_export_path.exists():
        raise FileNotFoundError(
            f"Heatmap export not found: {heatmap_export_path}. "
            "Run 'ez-worker build-heatmap-export --run-dir ...' first."
        )

    payload = json.loads(heatmap_export_path.read_text(encoding="utf-8"))
    formation_points = payload.get("formation_points", [])

    by_team: dict[int | None, list[dict]] = defaultdict(list)
    for point in formation_points:
        by_team[point.get("team_id")].append(point)

    teams = []
    for team_id, members in sorted(by_team.items(), key=lambda item: (item[0] is None, item[0])):
        ordered_members = sorted(
            members,
            key=lambda point: (
                float(point["average_pitch_xy_m"][0]),
                float(point["average_pitch_xy_m"][1]),
                int(point["track_id"]),
            ),
        )
        visible_line_count = _suggest_line_count(len(ordered_members))
        lines = _build_lines(ordered_members, visible_line_count)
        teams.append(
            {
                "team_id": team_id,
                "track_count": len(ordered_members),
                "team_centroid_pitch_xy_m": _average_xy(ordered_members),
                "visible_line_shape": [len(line["member_track_ids"]) for line in lines],
                "ordered_players": [
                    {
                        "track_id": int(member["track_id"]),
                        "average_pitch_xy_m": member["average_pitch_xy_m"],
                        "sample_count": int(member["sample_count"]),
                    }
                    for member in ordered_members
                ],
                "lines": lines,
            }
        )

    output = {
        "run_dir": str(run_dir),
        "source_file": str(heatmap_export_path),
        "teams": teams,
    }

    output_path = run_dir / "formation_export.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output_path


def _suggest_line_count(track_count: int) -> int:
    if track_count >= 9:
        return 4
    if track_count >= 6:
        return 3
    return 2


def _build_lines(members: list[dict], line_count: int) -> list[dict]:
    if not members:
        return []

    line_buckets: list[list[dict]] = [[] for _ in range(line_count)]
    total = len(members)
    for index, member in enumerate(members):
        bucket_index = min(line_count - 1, int((index / total) * line_count))
        line_buckets[bucket_index].append(member)

    lines: list[dict] = []
    for line_index, bucket in enumerate(line_buckets, start=1):
        if not bucket:
            continue
        sorted_bucket = sorted(bucket, key=lambda point: float(point["average_pitch_xy_m"][1]))
        lines.append(
            {
                "line_index": line_index,
                "member_track_ids": [int(member["track_id"]) for member in sorted_bucket],
                "line_average_pitch_xy_m": _average_xy(sorted_bucket),
                "members": [
                    {
                        "track_id": int(member["track_id"]),
                        "average_pitch_xy_m": member["average_pitch_xy_m"],
                        "sample_count": int(member["sample_count"]),
                    }
                    for member in sorted_bucket
                ],
            }
        )
    return lines


def _average_xy(points: list[dict]) -> list[float] | None:
    if not points:
        return None
    avg_x = sum(float(point["average_pitch_xy_m"][0]) for point in points) / len(points)
    avg_y = sum(float(point["average_pitch_xy_m"][1]) for point in points) / len(points)
    return [round(avg_x, 3), round(avg_y, 3)]
