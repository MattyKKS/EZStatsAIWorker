from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GRID_X_BINS = 6
GRID_Y_BINS = 4


def build_spatial_report(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    tracks_pitch_path = run_dir / "tracks_pitch.json"
    if not tracks_pitch_path.exists():
        raise FileNotFoundError(
            f"Projected pitch tracks not found: {tracks_pitch_path}. "
            "Run 'ez-worker apply-pitch-mapping --run-dir ...' first."
        )

    tracks = json.loads(tracks_pitch_path.read_text(encoding="utf-8"))
    player_tracks = [track for track in tracks if track.get("label") == "player"]
    ball_tracks = [track for track in tracks if track.get("label") == "ball"]

    report = {
        "run_dir": str(run_dir),
        "pitch_size_m": {"length": PITCH_LENGTH_M, "width": PITCH_WIDTH_M},
        "team_average_positions": _team_average_positions(player_tracks),
        "player_average_positions": _player_average_positions(player_tracks),
        "team_pitch_bins": _team_pitch_bins(player_tracks),
        "ball_average_position": _average_position(ball_tracks),
    }

    output_path = run_dir / "spatial_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def _team_average_positions(tracks: list[dict]) -> list[dict]:
    by_team: dict[int | None, list[dict]] = defaultdict(list)
    for track in tracks:
        by_team[track.get("team_id")].append(track)

    rows: list[dict] = []
    for team_id, members in sorted(by_team.items(), key=lambda item: (item[0] is None, item[0])):
        avg = _average_position(members)
        rows.append(
            {
                "team_id": team_id,
                "sample_count": len(members),
                "average_pitch_xy_m": avg,
            }
        )
    return rows


def _player_average_positions(tracks: list[dict]) -> list[dict]:
    by_player: dict[int, list[dict]] = defaultdict(list)
    for track in tracks:
        by_player[int(track["track_id"])].append(track)

    rows: list[dict] = []
    for track_id in sorted(by_player):
        members = by_player[track_id]
        avg = _average_position(members)
        team_id = members[0].get("team_id")
        rows.append(
            {
                "track_id": track_id,
                "team_id": team_id,
                "sample_count": len(members),
                "average_pitch_xy_m": avg,
            }
        )
    return rows


def _team_pitch_bins(tracks: list[dict]) -> list[dict]:
    by_team: dict[int | None, list[dict]] = defaultdict(list)
    for track in tracks:
        by_team[track.get("team_id")].append(track)

    rows: list[dict] = []
    for team_id, members in sorted(by_team.items(), key=lambda item: (item[0] is None, item[0])):
        grid = [[0 for _ in range(GRID_X_BINS)] for _ in range(GRID_Y_BINS)]
        for member in members:
            x, y = member["pitch_xy_m"]
            x_idx = min(GRID_X_BINS - 1, max(0, int((float(x) / PITCH_LENGTH_M) * GRID_X_BINS)))
            y_idx = min(GRID_Y_BINS - 1, max(0, int((float(y) / PITCH_WIDTH_M) * GRID_Y_BINS)))
            grid[y_idx][x_idx] += 1
        rows.append(
            {
                "team_id": team_id,
                "grid_rows": GRID_Y_BINS,
                "grid_cols": GRID_X_BINS,
                "occupancy_grid": grid,
            }
        )
    return rows


def _average_position(tracks: list[dict]) -> list[float] | None:
    if not tracks:
        return None
    avg_x = sum(float(track["pitch_xy_m"][0]) for track in tracks) / len(tracks)
    avg_y = sum(float(track["pitch_xy_m"][1]) for track in tracks) / len(tracks)
    return [round(avg_x, 3), round(avg_y, 3)]
