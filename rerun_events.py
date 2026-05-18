"""
Quick re-run of event detection on an existing run directory.
Prefers tracks_with_teams.json (has team_id set) over tracks.json.
Loads referee_track_ids.json if present (written by render_stats_video).

Usage:
    python rerun_events.py outputs/20260513_031215
"""
import json
import sys
from pathlib import Path

from ez_worker.analytics.events import detect_events
from ez_worker.config import DEFAULT_CONFIG
from ez_worker.schemas import TrackObservation, VideoMeta

run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/20260513_031215")

print(f"Run dir: {run_dir}")

video = VideoMeta(**json.loads((run_dir / "video_meta.json").read_text()))

# Prefer tracks_with_teams (team_id set) over raw tracks
tracks_path = run_dir / "tracks_with_teams.json"
if not tracks_path.exists():
    tracks_path = run_dir / "tracks.json"
print(f"Loading tracks from: {tracks_path.name}")
tracks_raw = json.loads(tracks_path.read_text())
tracks = [TrackObservation(**t) for t in tracks_raw]

# Load referee track IDs saved by render_stats_video (if available)
referee_path = run_dir / "referee_track_ids.json"
excluded_track_ids: set[int] | None = None
if referee_path.exists():
    excluded_track_ids = set(json.loads(referee_path.read_text(encoding="utf-8")))
    print(f"Excluding referee track IDs: {sorted(excluded_track_ids)}")
else:
    print("No referee_track_ids.json found - referees not explicitly excluded")

cfg = DEFAULT_CONFIG

events = detect_events(
    tracks=tracks,
    video=video,
    ball_track_id=cfg.ball_track_id,
    possession_distance_threshold_cm=cfg.possession_distance_threshold_cm,
    possession_distance_threshold_px=120.0,   # match run_full_pipeline.ps1 override
    possession_min_seconds=cfg.possession_min_seconds,
    pass_min_speed_cms=cfg.pass_min_speed_cms,
    pass_min_speed_px_per_s=150.0,            # lower than default to catch slow goalkeeper rolls
    shot_min_speed_cms=cfg.shot_min_speed_cms,
    shot_min_speed_px_per_s=cfg.shot_min_speed_px_per_s,
    shot_no_catch_seconds=cfg.shot_no_catch_seconds,
    ball_direction_change_min_deg=cfg.ball_direction_change_min_deg,
    clearance_min_flight_seconds=cfg.clearance_min_flight_seconds,
    clearance_min_arc_frac=cfg.clearance_min_arc_frac,
    pass_min_flight_frames=cfg.pass_min_flight_frames,
    owner_min_possession_frames=cfg.owner_min_possession_frames,
    pass_min_ball_travel_px=cfg.pass_min_ball_travel_px,
    reception_decel_fraction=cfg.reception_decel_fraction,
    touch_min_ball_speed_px_per_s=cfg.touch_min_ball_speed_px_per_s,
    excluded_track_ids=excluded_track_ids,
)

out = [e.model_dump(mode="json") for e in events]
(run_dir / "events.json").write_text(json.dumps(out, indent=2))
print(f"Wrote {len(events)} events to {run_dir / 'events.json'}")
for e in events:
    print(f"  {e.time_seconds:6.2f}s  {e.event_type:<14}  {e.actor_track_id} → {e.target_track_id}  {e.details}")
