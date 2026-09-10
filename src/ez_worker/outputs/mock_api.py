from __future__ import annotations

import json
from pathlib import Path


def build_mock_api(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    api_response_path = run_dir / "api_response.json"

    if not api_response_path.exists():
        raise FileNotFoundError(
            f"API response not found: {api_response_path}. "
            "Run 'ez-worker build-api-response --run-dir ...' first."
        )

    api_response = json.loads(api_response_path.read_text(encoding="utf-8"))
    data = api_response.get("data", {})

    mock_api_dir = run_dir / "mock_api"
    mock_api_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "match.json": {
            "status": api_response.get("status"),
            "api_version": api_response.get("api_version"),
            "match_id": data.get("match_id"),
            "video": data.get("video", {}),
        },
        "scoreboard.json": data.get("scoreboard", {}),
        "formation.json": {"formation_cards": data.get("formation_cards", [])},
        "timeline.json": {"timeline": data.get("timeline", [])},
        "possession.json": {"possession_segments": data.get("possession_segments", [])},
        "players.json": {"players": data.get("players", [])},
    }

    for filename, payload in files.items():
        (mock_api_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    index = {
        "run_id": run_dir.name,
        "base_dir": str(mock_api_dir),
        "files": {name.replace(".json", ""): str(mock_api_dir / name) for name in files},
    }
    index_path = mock_api_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index_path
