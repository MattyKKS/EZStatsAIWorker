from __future__ import annotations

import json
from pathlib import Path


def build_api_response(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    ui_payload_path = run_dir / "ui_payload.json"
    contract_path = run_dir / "output_contract.json"

    if not ui_payload_path.exists():
        raise FileNotFoundError(
            f"UI payload not found: {ui_payload_path}. "
            "Run 'ez-worker build-ui-payload --run-dir ...' first."
        )
    if not contract_path.exists():
        raise FileNotFoundError(
            f"Output contract not found: {contract_path}. "
            "Run 'ez-worker build-output-contract --run-dir ...' first."
        )

    ui_payload = json.loads(ui_payload_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    response = {
        "status": "ok",
        "api_version": "match-analysis-response-v1",
        "data": {
            "match_id": run_dir.name,
            "video": ui_payload.get("video", {}),
            "scoreboard": ui_payload.get("scoreboard", {}),
            "formation_cards": ui_payload.get("formation_cards", []),
            "timeline": ui_payload.get("timeline", []),
            "possession_segments": ui_payload.get("possession_segments", []),
            "players": ui_payload.get("players", []),
        },
        "meta": {
            "run_dir": str(run_dir),
            "recommended_contract_file": contract.get("recommended_product_file"),
            "contract_version": contract.get("contract_version"),
        },
    }

    output_path = run_dir / "api_response.json"
    output_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
    return output_path
