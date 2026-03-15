from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def apply_team_clusters(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    cluster_path = run_dir / "team_clusters.json"
    tracks_path = run_dir / "tracks.json"
    stats_path = run_dir / "player_stats.json"

    if not cluster_path.exists():
        raise FileNotFoundError(
            f"Team cluster file not found: {cluster_path}. "
            "Run 'ez-worker cluster-teams --run-dir ...' first."
        )
    if not tracks_path.exists():
        raise FileNotFoundError(f"Track file not found: {tracks_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Player stats file not found: {stats_path}")

    clusters = json.loads(cluster_path.read_text(encoding="utf-8"))
    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    cluster_counts = Counter(item["cluster_id"] for item in clusters["tracks"])
    team_like_clusters = [cluster_id for cluster_id, _ in cluster_counts.most_common(2)]
    outlier_clusters = sorted(cluster_id for cluster_id in cluster_counts if cluster_id not in team_like_clusters)

    cluster_to_team: dict[int, int | None] = {}
    for team_id, cluster_id in enumerate(team_like_clusters):
        cluster_to_team[cluster_id] = team_id
    for cluster_id in outlier_clusters:
        cluster_to_team[cluster_id] = None

    track_to_team: dict[int, int | None] = {
        int(item["track_id"]): cluster_to_team.get(int(item["cluster_id"]))
        for item in clusters["tracks"]
    }

    enriched_tracks = []
    for track in tracks:
        enriched = dict(track)
        if track.get("label") == "player":
            enriched["team_id"] = track_to_team.get(int(track["track_id"]))
        enriched_tracks.append(enriched)

    enriched_stats = []
    for stat in stats:
        enriched = dict(stat)
        enriched["team_id"] = track_to_team.get(int(stat["track_id"]))
        enriched_stats.append(enriched)

    team_summary = _build_team_summary(enriched_stats)
    payload = {
        "run_dir": str(run_dir),
        "cluster_method": clusters.get("method"),
        "cluster_preprocessing": clusters.get("preprocessing"),
        "cluster_to_team_mapping": cluster_to_team,
        "team_like_clusters": team_like_clusters,
        "outlier_clusters": outlier_clusters,
        "tracks_with_teams_path": str(run_dir / "tracks_with_teams.json"),
        "player_stats_with_teams_path": str(run_dir / "player_stats_with_teams.json"),
        "team_summary": team_summary,
    }

    (run_dir / "tracks_with_teams.json").write_text(
        json.dumps(enriched_tracks, indent=2),
        encoding="utf-8",
    )
    (run_dir / "player_stats_with_teams.json").write_text(
        json.dumps(enriched_stats, indent=2),
        encoding="utf-8",
    )
    output_path = run_dir / "team_assignments.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _build_team_summary(stats: list[dict]) -> list[dict]:
    by_team: dict[int, list[dict]] = defaultdict(list)
    for item in stats:
        team_id = item.get("team_id")
        if team_id is None:
            continue
        by_team[int(team_id)].append(item)

    summary: list[dict] = []
    for team_id in sorted(by_team):
        rows = by_team[team_id]
        summary.append(
            {
                "team_id": team_id,
                "player_count": len(rows),
                "touch_count": sum(int(row.get("touch_count", 0)) for row in rows),
                "pass_count": sum(int(row.get("pass_count", 0)) for row in rows),
                "shot_count": sum(int(row.get("shot_count", 0)) for row in rows),
                "approx_distance_px": round(
                    sum(float(row.get("approx_distance_px", 0.0)) for row in rows),
                    2,
                ),
            }
        )
    return summary
