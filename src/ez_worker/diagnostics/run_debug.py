from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def build_run_debug_report(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    tracks_path = run_dir / "tracks.json"
    summary_path = run_dir / "summary.json"

    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracks file not found: {tracks_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    by_frame: dict[int, list[dict]] = defaultdict(list)
    player_by_track: dict[int, list[dict]] = defaultdict(list)
    source_counter: Counter = Counter()
    label_counter: Counter = Counter()
    touchline_track_ids: set[int] = set()
    touchline_obs_count = 0

    for row in tracks:
        frame_index = int(row["frame_index"])
        by_frame[frame_index].append(row)
        label = str(row.get("label", "unknown"))
        source_label = str(row.get("source_label") or label)
        label_counter[label] += 1
        source_counter[source_label] += 1

        if label == "player":
            track_id = int(row["track_id"])
            player_by_track[track_id].append(row)
            bbox = row.get("bbox", {})
            x1 = float(bbox.get("x1", 0.0))
            x2 = float(bbox.get("x2", 0.0))
            if x1 <= 0.035 or x2 >= 0.965:
                touchline_track_ids.add(track_id)
                touchline_obs_count += 1

    player_counts = [sum(1 for row in rows if row.get("label") == "player") for _, rows in sorted(by_frame.items())]
    ball_counts = [sum(1 for row in rows if row.get("label") == "ball") for _, rows in sorted(by_frame.items())]

    track_lengths = sorted(
        (
            {
                "track_id": track_id,
                "frame_count": len(obs),
                "dominant_source_label": Counter(
                    str(row.get("source_label") or row.get("label") or "unknown")
                    for row in obs
                ).most_common(1)[0][0],
            }
            for track_id, obs in player_by_track.items()
        ),
        key=lambda row: row["frame_count"],
        reverse=True,
    )

    warnings: list[str] = []
    if source_counter.get("goalkeeper", 0) == 0:
        warnings.append("No goalkeeper detections were produced in this run.")
    if source_counter.get("referee", 0) == 0:
        warnings.append("No referee detections were produced in this run.")
    if player_counts and min(player_counts) <= 6:
        warnings.append("Some frames have very low player coverage after cleanup.")
    if len(touchline_track_ids) <= 2:
        warnings.append("Very few touchline player tracks survived cleanup.")

    payload = {
        "run_dir": str(run_dir),
        "summary": summary,
        "label_counts": dict(label_counter),
        "source_label_counts": dict(source_counter),
        "frame_coverage": {
            "frames_with_tracks": len(by_frame),
            "min_players_in_frame": min(player_counts) if player_counts else 0,
            "max_players_in_frame": max(player_counts) if player_counts else 0,
            "avg_players_in_frame": round(sum(player_counts) / max(len(player_counts), 1), 3),
            "frames_with_ball": sum(1 for count in ball_counts if count > 0),
        },
        "touchline_players": {
            "track_count": len(touchline_track_ids),
            "observation_count": touchline_obs_count,
            "track_ids": sorted(touchline_track_ids),
        },
        "longest_player_tracks": track_lengths[:12],
        "warnings": warnings,
    }

    output_path = run_dir / "run_debug_report.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
