from __future__ import annotations

import json
from pathlib import Path


def build_integration_bundle(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()

    required_files = {
        "summary": run_dir / "summary.json",
        "analysis_ready_summary": run_dir / "analysis_ready_summary.json",
        "team_assignments": run_dir / "team_assignments.json",
        "team_report": run_dir / "team_report_analysis_ready.json",
        "spatial_report": run_dir / "spatial_report.json",
        "heatmap_export": run_dir / "heatmap_export.json",
        "formation_export": run_dir / "formation_export.json",
        "possession_report": run_dir / "possession_report.json",
        "events": run_dir / "events_analysis_ready.json",
        "player_stats": run_dir / "player_stats_analysis_ready.json",
    }

    missing = [name for name, path in required_files.items() if not path.exists()]
    if missing:
        missing_names = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required files for integration bundle: {missing_names}. "
            "Run the earlier post-run steps first."
        )

    summary = json.loads(required_files["summary"].read_text(encoding="utf-8"))
    analysis_ready_summary = json.loads(required_files["analysis_ready_summary"].read_text(encoding="utf-8"))
    team_report = json.loads(required_files["team_report"].read_text(encoding="utf-8"))
    possession_report = json.loads(required_files["possession_report"].read_text(encoding="utf-8"))
    formation_export = json.loads(required_files["formation_export"].read_text(encoding="utf-8"))
    cleaned_player_stats = json.loads(required_files["player_stats"].read_text(encoding="utf-8"))

    video_cleaned = dict(summary)
    video_cleaned["event_count"] = int(possession_report.get("event_count", 0))
    video_cleaned["player_count"] = int(analysis_ready_summary.get("kept_player_count", 0))
    video_cleaned["removed_track_count"] = int(analysis_ready_summary.get("removed_track_count", 0))

    payload = {
        "run_dir": str(run_dir),
        "bundle_version": "integration-bundle-v2",
        "video_raw": summary,
        "video_cleaned": video_cleaned,
        "team_summary": team_report.get("team_rows", []),
        "event_summary": team_report.get("event_summary", {}),
        "possession_summary": {
            "event_count": possession_report.get("event_count", 0),
            "touches_by_team": possession_report.get("touches_by_team", []),
            "passes_by_team": possession_report.get("passes_by_team", []),
            "possession_segments": possession_report.get("possession_segments", []),
        },
        "formation_summary": [
            {
                "team_id": team.get("team_id"),
                "track_count": team.get("track_count"),
                "team_centroid_pitch_xy_m": team.get("team_centroid_pitch_xy_m"),
                "visible_line_shape": team.get("visible_line_shape"),
            }
            for team in formation_export.get("teams", [])
        ],
        "analysis_ready_players": [
            {
                "track_id": int(player["track_id"]),
                "team_id": player.get("team_id"),
                "role_hint": player.get("role_hint"),
                "role_hint_confidence": player.get("role_hint_confidence"),
            }
            for player in cleaned_player_stats
        ],
        "artifact_paths": {name: str(path) for name, path in required_files.items()},
    }

    output_path = run_dir / "integration_bundle.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
