from __future__ import annotations

import json
from pathlib import Path


def build_ui_payload(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    bundle_path = run_dir / "integration_bundle.json"
    possession_path = run_dir / "possession_report.json"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Integration bundle not found: {bundle_path}. "
            "Run 'ez-worker build-integration-bundle --run-dir ...' first."
        )
    if not possession_path.exists():
        raise FileNotFoundError(
            f"Possession report not found: {possession_path}. "
            "Run 'ez-worker build-possession-report --run-dir ...' first."
        )

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    possession = json.loads(possession_path.read_text(encoding="utf-8"))

    payload = {
        "run_id": run_dir.name,
        "video": {
            "path": bundle["video_cleaned"].get("processed_video_path"),
            "fps": bundle["video_cleaned"].get("fps"),
            "frame_count": bundle["video_cleaned"].get("frame_count"),
            "event_count": bundle["video_cleaned"].get("event_count"),
            "player_count": bundle["video_cleaned"].get("player_count"),
        },
        "scoreboard": {
            "team_rows": bundle.get("team_summary", []),
            "touches_by_team": bundle.get("event_summary", {}).get("touches_by_team", []),
            "passes_by_team_pair": bundle.get("event_summary", {}).get("passes_by_team_pair", []),
        },
        "formation_cards": bundle.get("formation_summary", []),
        "timeline": possession.get("timeline", []),
        "possession_segments": possession.get("possession_segments", []),
        "players": bundle.get("analysis_ready_players", []),
    }

    output_path = run_dir / "ui_payload.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
