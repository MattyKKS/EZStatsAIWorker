from __future__ import annotations

import json
from pathlib import Path


def build_route_manifest(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    mock_api_index_path = run_dir / "mock_api" / "index.json"
    if not mock_api_index_path.exists():
        raise FileNotFoundError(
            f"Mock API index not found: {mock_api_index_path}. "
            "Run 'ez-worker build-mock-api --run-dir ...' first."
        )

    index = json.loads(mock_api_index_path.read_text(encoding="utf-8"))
    files = index.get("files", {})
    run_id = index.get("run_id", run_dir.name)

    routes = [
        {
            "method": "GET",
            "path": f"/api/matches/{run_id}",
            "mock_file": files.get("match"),
            "purpose": "basic match and video summary",
        },
        {
            "method": "GET",
            "path": f"/api/matches/{run_id}/scoreboard",
            "mock_file": files.get("scoreboard"),
            "purpose": "team summary and scoreboard totals",
        },
        {
            "method": "GET",
            "path": f"/api/matches/{run_id}/formation",
            "mock_file": files.get("formation"),
            "purpose": "formation cards and team layout summary",
        },
        {
            "method": "GET",
            "path": f"/api/matches/{run_id}/timeline",
            "mock_file": files.get("timeline"),
            "purpose": "event timeline items",
        },
        {
            "method": "GET",
            "path": f"/api/matches/{run_id}/possession",
            "mock_file": files.get("possession"),
            "purpose": "possession segments",
        },
        {
            "method": "GET",
            "path": f"/api/matches/{run_id}/players",
            "mock_file": files.get("players"),
            "purpose": "analysis-ready players and role hints",
        },
    ]

    payload = {
        "run_id": run_id,
        "base_mock_dir": index.get("base_dir"),
        "routes": routes,
    }

    output_path = run_dir / "mock_api" / "route_manifest.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
