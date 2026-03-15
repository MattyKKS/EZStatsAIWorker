from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
TEAM_GRID_X = 12
TEAM_GRID_Y = 8
PLAYER_GRID_X = 10
PLAYER_GRID_Y = 6


def build_heatmap_export(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    tracks_pitch_path = run_dir / "tracks_pitch.json"
    if not tracks_pitch_path.exists():
        raise FileNotFoundError(
            f"Projected pitch tracks not found: {tracks_pitch_path}. "
            "Run 'ez-worker apply-pitch-mapping --run-dir ...' first."
        )

    tracks = json.loads(tracks_pitch_path.read_text(encoding="utf-8"))
    player_tracks = [track for track in tracks if track.get("label") == "player"]

    payload = {
        "run_dir": str(run_dir),
        "pitch_size_m": {"length": PITCH_LENGTH_M, "width": PITCH_WIDTH_M},
        "team_heatmaps": _team_heatmaps(player_tracks),
        "player_heatmaps": _player_heatmaps(player_tracks),
        "formation_points": _formation_points(player_tracks),
    }

    output_path = run_dir / "heatmap_export.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _team_heatmaps(tracks: list[dict]) -> list[dict]:
    by_team: dict[int | None, list[dict]] = defaultdict(list)
    for track in tracks:
        by_team[track.get("team_id")].append(track)

    rows: list[dict] = []
    for team_id, members in sorted(by_team.items(), key=lambda item: (item[0] is None, item[0])):
        rows.append(
            {
                "team_id": team_id,
                "grid_rows": TEAM_GRID_Y,
                "grid_cols": TEAM_GRID_X,
                "heatmap_grid": _accumulate_grid(members, TEAM_GRID_X, TEAM_GRID_Y),
            }
        )
    return rows


def _player_heatmaps(tracks: list[dict]) -> list[dict]:
    by_player: dict[int, list[dict]] = defaultdict(list)
    for track in tracks:
        by_player[int(track["track_id"])].append(track)

    rows: list[dict] = []
    for track_id in sorted(by_player):
        members = by_player[track_id]
        rows.append(
            {
                "track_id": track_id,
                "team_id": members[0].get("team_id"),
                "grid_rows": PLAYER_GRID_Y,
                "grid_cols": PLAYER_GRID_X,
                "heatmap_grid": _accumulate_grid(members, PLAYER_GRID_X, PLAYER_GRID_Y),
            }
        )
    return rows


def _formation_points(tracks: list[dict]) -> list[dict]:
    by_player: dict[int, list[dict]] = defaultdict(list)
    for track in tracks:
        by_player[int(track["track_id"])].append(track)

    rows: list[dict] = []
    for track_id in sorted(by_player):
        members = by_player[track_id]
        avg_x = sum(float(member["pitch_xy_m"][0]) for member in members) / len(members)
        avg_y = sum(float(member["pitch_xy_m"][1]) for member in members) / len(members)
        rows.append(
            {
                "track_id": track_id,
                "team_id": members[0].get("team_id"),
                "average_pitch_xy_m": [round(avg_x, 3), round(avg_y, 3)],
                "sample_count": len(members),
            }
        )
    return rows


def _accumulate_grid(tracks: list[dict], grid_x: int, grid_y: int) -> list[list[int]]:
    grid = [[0 for _ in range(grid_x)] for _ in range(grid_y)]
    for track in tracks:
        x, y = track["pitch_xy_m"]
        x_idx = min(grid_x - 1, max(0, int((float(x) / PITCH_LENGTH_M) * grid_x)))
        y_idx = min(grid_y - 1, max(0, int((float(y) / PITCH_WIDTH_M) * grid_y)))
        grid[y_idx][x_idx] += 1
    return grid
