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

    # Only count non-excluded tracks when deciding team clusters
    # Tutorial uses exactly 2 clusters — both are teams, no outlier cluster
    active_items = [item for item in clusters["tracks"] if not item.get("excluded")]
    cluster_counts = Counter(item["cluster_id"] for item in active_items if item["cluster_id"] is not None)
    team_like_clusters = [cluster_id for cluster_id, _ in cluster_counts.most_common(2)]

    cluster_to_team: dict[int, int] = {}
    for team_id, cluster_id in enumerate(team_like_clusters):
        cluster_to_team[cluster_id] = team_id

    # Assign player tracks to teams
    track_to_team: dict[int, int | None] = {}
    for item in clusters["tracks"]:
        tid = int(item["track_id"])
        if item.get("excluded") or item["cluster_id"] is None:
            track_to_team[tid] = None  # will be resolved for GKs below
        else:
            track_to_team[tid] = cluster_to_team.get(int(item["cluster_id"]))

    # Tutorial goalkeeper heuristic: assign each GK to the team whose centroid
    # (average x position of outfield players) is closest to the GK's position.
    excluded_ids = set(clusters.get("excluded_track_ids", []))

    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    # Always compute mean bottom-center position per track (used for GKs + fallback)
    positions_by_track: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for obs in tracks:
        tid = int(obs["track_id"])
        bbox = obs.get("bbox", {})
        cx = (float(bbox.get("x1", 0)) + float(bbox.get("x2", 0))) / 2.0
        by = float(bbox.get("y2", 0))
        positions_by_track[tid].append((cx, by))

    mean_pos_by_track: dict[int, tuple[float, float]] = {
        tid: (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        for tid, pts in positions_by_track.items()
    }

    # Team centroid = mean 2D position of confidently-assigned outfield players
    team_pts: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for tid, team_id in track_to_team.items():
        if team_id is not None and tid not in excluded_ids and tid in mean_pos_by_track:
            team_pts[team_id].append(mean_pos_by_track[tid])

    centroid_2d: dict[int, tuple[float, float]] = {
        team_id: (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        for team_id, pts in team_pts.items() if pts
    }

    # Assign GKs and any other unassigned tracks by proximity to team centroids
    for tid, team_id in list(track_to_team.items()):
        if team_id is None and tid in mean_pos_by_track and len(centroid_2d) >= 2:
            pos = mean_pos_by_track[tid]
            dist0 = _dist(pos, centroid_2d.get(0, (0.25, 0.5)))
            dist1 = _dist(pos, centroid_2d.get(1, (0.75, 0.5)))
            track_to_team[tid] = 0 if dist0 <= dist1 else 1

    # Track stitching: new tracks appearing near a recently-disappeared track
    # inherit its team. Fixes ByteTrack ID swaps during player overlap.
    STITCH_GAP = 30   # frames — how recently a track must have ended
    STITCH_DIST = 0.08  # normalised distance — how close positions must be

    track_lifespan: dict[int, dict] = {}
    for obs in tracks:
        if obs.get("label") != "player":
            continue
        tid = int(obs["track_id"])
        fi = int(obs["frame_index"])
        bbox = obs.get("bbox", {})
        cx = (float(bbox.get("x1", 0)) + float(bbox.get("x2", 0))) / 2.0
        cy = (float(bbox.get("y1", 0)) + float(bbox.get("y2", 0))) / 2.0
        if tid not in track_lifespan:
            track_lifespan[tid] = {"start": fi, "end": fi, "start_pos": (cx, cy), "end_pos": (cx, cy)}
        else:
            if fi < track_lifespan[tid]["start"]:
                track_lifespan[tid]["start"] = fi
                track_lifespan[tid]["start_pos"] = (cx, cy)
            if fi > track_lifespan[tid]["end"]:
                track_lifespan[tid]["end"] = fi
                track_lifespan[tid]["end_pos"] = (cx, cy)

    for tid, info in sorted(track_lifespan.items(), key=lambda x: x[1]["start"]):
        if track_to_team.get(tid) is not None:
            continue  # already assigned
        start_frame = info["start"]
        sx, sy = info["start_pos"]
        best_team = None
        best_dist = STITCH_DIST
        for other_tid, other_info in track_lifespan.items():
            if other_tid == tid:
                continue
            if track_to_team.get(other_tid) is None:
                continue
            gap = start_frame - other_info["end"]
            if gap < 0 or gap > STITCH_GAP:
                continue
            ex, ey = other_info["end_pos"]
            d = ((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_team = track_to_team[other_tid]
        if best_team is not None:
            track_to_team[tid] = best_team

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
        "outlier_clusters": [],
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
