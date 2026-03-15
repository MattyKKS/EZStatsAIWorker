from __future__ import annotations

import json
from pathlib import Path


EXCLUDED_ROLE_HINTS = {"referee_candidate", "outlier_candidate"}


def build_role_filtered_outputs(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    tracks_path = run_dir / "tracks_with_teams.json"
    stats_path = run_dir / "player_stats_with_roles.json"
    role_hints_path = run_dir / "role_hints.json"

    if not tracks_path.exists():
        raise FileNotFoundError(
            f"Team-aware tracks not found: {tracks_path}. "
            "Run 'ez-worker apply-team-clusters --run-dir ...' first."
        )
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Role-aware player stats not found: {stats_path}. "
            "Run 'ez-worker build-role-hints --run-dir ...' first."
        )
    if not role_hints_path.exists():
        raise FileNotFoundError(
            f"Role hints not found: {role_hints_path}. "
            "Run 'ez-worker build-role-hints --run-dir ...' first."
        )

    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    player_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    role_hints = json.loads(role_hints_path.read_text(encoding="utf-8"))

    role_by_track = {
        int(item["track_id"]): item["role_hint"]
        for item in role_hints.get("roles", [])
    }

    filtered_tracks = []
    removed_track_ids: list[int] = []
    for track in tracks:
        if track.get("label") != "player":
            filtered_tracks.append(track)
            continue
        track_id = int(track["track_id"])
        role_hint = role_by_track.get(track_id, "player")
        if role_hint in EXCLUDED_ROLE_HINTS:
            removed_track_ids.append(track_id)
            continue
        enriched = dict(track)
        enriched["role_hint"] = role_hint
        filtered_tracks.append(enriched)

    filtered_stats = []
    for stat in player_stats:
        track_id = int(stat["track_id"])
        if role_by_track.get(track_id, "player") in EXCLUDED_ROLE_HINTS:
            continue
        filtered_stats.append(dict(stat))

    summary = {
        "run_dir": str(run_dir),
        "method": "role-filter-v1",
        "excluded_role_hints": sorted(EXCLUDED_ROLE_HINTS),
        "removed_track_ids": sorted(set(removed_track_ids)),
        "removed_track_count": len(sorted(set(removed_track_ids))),
        "kept_player_count": len(filtered_stats),
        "tracks_output_path": str(run_dir / "tracks_analysis_ready.json"),
        "player_stats_output_path": str(run_dir / "player_stats_analysis_ready.json"),
    }

    (run_dir / "tracks_analysis_ready.json").write_text(
        json.dumps(filtered_tracks, indent=2),
        encoding="utf-8",
    )
    (run_dir / "player_stats_analysis_ready.json").write_text(
        json.dumps(filtered_stats, indent=2),
        encoding="utf-8",
    )
    output_path = run_dir / "analysis_ready_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path
