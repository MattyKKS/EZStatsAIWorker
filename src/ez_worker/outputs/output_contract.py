from __future__ import annotations

import json
from pathlib import Path


def build_output_contract(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    bundle_path = run_dir / "integration_bundle.json"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Integration bundle not found: {bundle_path}. "
            "Run 'ez-worker build-integration-bundle --run-dir ...' first."
        )

    contract = {
        "run_dir": str(run_dir),
        "contract_version": "output-contract-v1",
        "main_entry_file": "integration_bundle.json",
        "primary_sections": [
            {
                "name": "video_raw",
                "purpose": "Original run-level summary before analysis-ready filtering.",
                "key_fields": ["video_path", "fps", "frame_count", "track_count", "event_count", "player_count"],
            },
            {
                "name": "video_cleaned",
                "purpose": "Preferred cleaned summary for product-side use.",
                "key_fields": [
                    "video_path",
                    "fps",
                    "frame_count",
                    "track_count",
                    "event_count",
                    "player_count",
                    "removed_track_count",
                    "processed_video_path",
                ],
            },
            {
                "name": "team_summary",
                "purpose": "Per-team totals after cleaned event filtering.",
                "key_fields": ["team_id", "player_count", "touch_count", "pass_count", "shot_count"],
            },
            {
                "name": "event_summary",
                "purpose": "Cleaned event totals and team-to-team pass counts.",
                "key_fields": ["touches_by_team", "passes_by_team_pair", "shots_by_team"],
            },
            {
                "name": "possession_summary",
                "purpose": "Simplified possession segments and counts for timeline use.",
                "key_fields": ["event_count", "touches_by_team", "passes_by_team", "possession_segments"],
            },
            {
                "name": "formation_summary",
                "purpose": "Simple team tactical layout summary from pitch-projected tracks.",
                "key_fields": ["team_id", "track_count", "team_centroid_pitch_xy_m", "visible_line_shape"],
            },
            {
                "name": "analysis_ready_players",
                "purpose": "Cleaned player list with provisional team and role-hint metadata.",
                "key_fields": ["track_id", "team_id", "role_hint", "role_hint_confidence"],
            },
            {
                "name": "artifact_paths",
                "purpose": "Pointers to the detailed source files for deeper product integration.",
                "key_fields": [
                    "summary",
                    "analysis_ready_summary",
                    "team_assignments",
                    "team_report",
                    "spatial_report",
                    "heatmap_export",
                    "formation_export",
                    "possession_report",
                    "events",
                    "player_stats",
                ],
            },
        ],
        "recommended_product_file": "integration_bundle.json",
        "notes": [
            "Use video_cleaned instead of video_raw for product-facing summaries.",
            "Treat role hints as provisional, not final labels.",
            "Use artifact_paths when the frontend or backend needs the full detailed files.",
        ],
    }

    output_path = run_dir / "output_contract.json"
    output_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return output_path
