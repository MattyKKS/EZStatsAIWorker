"""
Re-run event detection on an existing run directory.

Prefers tracks_with_teams.json (has team_id set) over tracks.json.
Loads referee_track_ids.json if present (written by render_stats_video).

If pitch_keypoints.json is present, builds a homography matrix from the saved
keypoint pixel↔cm correspondences and projects all player feet + ball positions
to real-world pitch cm coordinates before running event detection. This gives
accurate Euclidean distances regardless of camera perspective distortion.

Falls back to pixel-space mode if:
  - pitch_keypoints.json is missing
  - fewer than 6 valid keypoints are available
  - the homography reprojection error exceeds 100 cm
  - --pixel flag is passed

Usage:
    python rerun_events.py outputs/20260527_215826
    python rerun_events.py outputs/20260527_215826 --pixel        # force pixel mode
    python rerun_events.py outputs/20260527_215826 --debug        # trace all frames
    python rerun_events.py outputs/20260527_215826 --debug 15 30  # trace 15s–30s only
"""
from __future__ import annotations

import bisect
import json
import math
import shutil
import sys
from pathlib import Path

# Allow running as a standalone script from the project root without needing
# the package to be pip-installed — just add src/ to the search path.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np

from ez_worker.analytics.events import detect_events
from ez_worker.config import DEFAULT_CONFIG
from ez_worker.schemas import TrackObservation, VideoMeta

# Pitch model vertex table — lets a per-frame keypoint index be resolved to its
# (x_cm, y_cm) landmark even when that index is absent from the best frame.
try:
    from sports.configs.soccer import SoccerPitchConfiguration
    _VERT = SoccerPitchConfiguration().vertices
except Exception:
    _VERT = []

# ── CLI args ──────────────────────────────────────────────────────────────────
_args = sys.argv[1:]
force_pixel  = "--pixel"  in _args
debug_mode   = "--debug"  in _args
_args = [a for a in _args if not a.startswith("--")]
run_dir      = Path(_args[0]) if _args else Path("outputs/20260513_031215")
debug_t_start = float(_args[1]) if debug_mode and len(_args) > 1 else 0.0
debug_t_end   = float(_args[2]) if debug_mode and len(_args) > 2 else 9999.0

print(f"Run dir: {run_dir}")

# ── Load video meta ───────────────────────────────────────────────────────────
video = VideoMeta(**json.loads((run_dir / "video_meta.json").read_text()))
print(f"Video: {video.width}×{video.height}  {video.fps:.2f}fps  {video.frame_count} frames")

# ── Load tracks (prefer tracks_with_teams for team_id) ───────────────────────
tracks_path = run_dir / "tracks_with_teams.json"
if not tracks_path.exists():
    tracks_path = run_dir / "tracks.json"
print(f"Loading tracks from: {tracks_path.name}")
tracks_raw = json.loads(tracks_path.read_text())
tracks = [TrackObservation(**t) for t in tracks_raw]

# ── Referee exclusions ────────────────────────────────────────────────────────
referee_path = run_dir / "referee_track_ids.json"
excluded_track_ids: set[int] | None = None
if referee_path.exists():
    excluded_track_ids = set(json.loads(referee_path.read_text(encoding="utf-8")))
    print(f"Excluding referee track IDs: {sorted(excluded_track_ids)}")
else:
    print("No referee_track_ids.json — referees not explicitly excluded")

cfg = DEFAULT_CONFIG

# ── Pitch homography ──────────────────────────────────────────────────────────
ball_pitch_pos: dict[int, tuple[float, float]] | None = None
player_pitch_pos: dict[int, dict[int, tuple[float, float]]] | None = None

kp_path = run_dir / "pitch_keypoints.json"

if not force_pixel and kp_path.exists():
    print("\nBuilding homography from pitch_keypoints.json ...")
    kp_data = json.loads(kp_path.read_text())

    kp_pixel: dict[str, list[float]] = kp_data.get("keypoints", {})
    kp_cm:    dict[str, list[float]] = kp_data.get("keypoint_pitch_xy_cm", {})

    # Collect matching pairs (same keypoint index present in both dicts)
    pixel_pts: list[list[float]] = []
    cm_pts:    list[list[float]] = []
    for idx_str, px_pos in kp_pixel.items():
        if idx_str in kp_cm:
            pixel_pts.append(px_pos)
            cm_pts.append(kp_cm[idx_str])

    print(f"  Keypoint pairs available: {len(pixel_pts)}")

    if len(pixel_pts) >= 6:
        src = np.array(pixel_pts, dtype=np.float32)
        dst = np.array(cm_pts,    dtype=np.float32)

        # RANSAC findHomography — tolerant of up to 50 cm reprojection error
        H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 50.0)

        if H is not None:
            # Validate: measure reprojection error on RANSAC inliers only.
            # Outlier keypoints (wrong detections) are excluded by RANSAC — don't
            # let them inflate the error metric for an otherwise good H matrix.
            inlier_bool = inlier_mask.ravel().astype(bool)
            n_inliers   = int(inlier_bool.sum())
            reproj = cv2.perspectiveTransform(
                src.reshape(-1, 1, 2), H
            ).reshape(-1, 2)
            errors_cm  = np.sqrt(np.sum((reproj - dst) ** 2, axis=1))
            inlier_err = errors_cm[inlier_bool]
            mean_err   = float(inlier_err.mean()) if len(inlier_err) > 0 else 9999.0
            max_err    = float(inlier_err.max())  if len(inlier_err) > 0 else 9999.0

            print(f"  Reprojection error: mean={mean_err:.1f}cm  max={max_err:.1f}cm")
            print(f"  Inliers: {n_inliers}/{len(pixel_pts)}")

            # Sanity check: project a known pitch landmark to verify orientation
            # Keypoint at centre of pitch should be near (6000, 3500) cm
            # We use the midfield keypoint if available
            _centre_check_ok = True
            for k, expected in [("31", (6915.0, 3500.0)), ("13", (6000.0, 0.0))]:
                if k in kp_pixel:
                    test_px = np.array([[kp_pixel[k]]], dtype=np.float32)
                    test_cm = cv2.perspectiveTransform(test_px, H)[0][0]
                    expected_cm = np.array(kp_cm.get(k, expected), dtype=np.float32)
                    err = float(np.linalg.norm(test_cm - expected_cm))
                    print(f"  Sanity kp{k}: projected={test_cm[0]:.0f},{test_cm[1]:.0f}cm  expected={expected_cm[0]:.0f},{expected_cm[1]:.0f}cm  err={err:.1f}cm")
                    break

            if mean_err <= 100.0 and n_inliers >= 5:
                print("  Homography valid OK -- projecting tracks to pitch cm ...")

                # Build convex hull of INLIER keypoints in pixel space.
                # Used for diagnostics only — we still project ALL ball positions
                # so that pitch velocities are computed everywhere (required for
                # IN_FLIGHT transitions). Extrapolated positions outside the hull
                # are less accurate but still provide usable velocity signals.
                inlier_src    = src[inlier_bool]
                coverage_hull = cv2.convexHull(inlier_src.reshape(-1, 1, 2))

                # Group tracks by frame
                by_frame: dict[int, list[TrackObservation]] = {}
                for t in tracks:
                    by_frame.setdefault(t.frame_index, []).append(t)

                ball_pitch_pos    = {}
                player_pitch_pos  = {}

                n_ball_outside_hull = 0

                # ── Per-frame homography (RC-1 fix) ───────────────────────────
                # A single H calibrated on one frame drifts as the camera pans:
                # measured camera travel from the best frame was 434 px median on
                # the benchmark clip but 3106 px on a panning highlight, which is
                # more than a frame width — so "pitch cm" became fiction. When
                # pitch_keypoints_per_frame.json is present, use the homography
                # from the NEAREST sampled frame instead of one for the whole clip.
                H_samples: dict[int, np.ndarray] = {}
                pf_path = run_dir / "pitch_keypoints_per_frame.json"
                if pf_path.exists():
                    try:
                        pf = json.loads(pf_path.read_text())
                        for fstr, entry in pf.get("frames", {}).items():
                            kps = entry.get("keypoints", {})
                            sp, sc = [], []
                            for idx_str, px_pos in kps.items():
                                if idx_str in kp_cm:
                                    sp.append(px_pos)
                                    sc.append(kp_cm[idx_str])
                                elif int(idx_str) < len(_VERT):
                                    sp.append(px_pos)
                                    sc.append(list(_VERT[int(idx_str)]))
                            if len(sp) < 6:
                                continue
                            Hs, mask = cv2.findHomography(
                                np.array(sp, dtype=np.float32),
                                np.array(sc, dtype=np.float32), cv2.RANSAC, 50.0)
                            if Hs is None or mask is None or int(mask.sum()) < 5:
                                continue
                            rp = cv2.perspectiveTransform(
                                np.array(sp, dtype=np.float32).reshape(-1, 1, 2), Hs
                            ).reshape(-1, 2)
                            err = np.sqrt(((rp - np.array(sc)) ** 2).sum(axis=1))
                            if float(err[mask.ravel().astype(bool)].mean()) <= 100.0:
                                H_samples[int(fstr)] = Hs
                    except Exception as exc:
                        print(f"  Per-frame homography unavailable ({exc}) - using single H.")

                if H_samples:
                    _skeys = sorted(H_samples)
                    _max_gap = int(pf.get("stride", 10)) * 3
                    print(f"  Per-frame homography: {len(H_samples)} valid matrices "
                          f"(of {len(pf.get('frames', {}))} sampled), max gap {_max_gap} frames")

                    def H_for(fi: int) -> np.ndarray:
                        j = bisect.bisect_left(_skeys, fi)
                        best, bd = None, None
                        for k in (j - 1, j):
                            if 0 <= k < len(_skeys):
                                d = abs(_skeys[k] - fi)
                                if bd is None or d < bd:
                                    best, bd = _skeys[k], d
                        return H_samples[best] if best is not None and bd <= _max_gap else H
                else:
                    print("  Per-frame homography: none - using the single best-frame H.")

                    def H_for(fi: int) -> np.ndarray:
                        return H

                for fi, frame_tracks in by_frame.items():
                    frame_players: dict[int, tuple[float, float]] = {}

                    # Pick the ball the same way detect_events does (_get_ball)
                    ball_obs = (
                        next((t for t in frame_tracks
                              if t.label == "ball" and t.track_id == cfg.ball_track_id), None)
                        or next((t for t in frame_tracks if t.label == "ball"), None)
                    )

                    all_pts: list[tuple[str, int, float, float]] = []  # (role, id, px, py)

                    if ball_obs is not None:
                        bx = (ball_obs.bbox.x1 + ball_obs.bbox.x2) / 2 * video.width
                        by = ball_obs.bbox.y2 * video.height
                        all_pts.append(("ball", ball_obs.track_id, bx, by))
                        # Track coverage for diagnostics
                        if cv2.pointPolygonTest(coverage_hull, (float(bx), float(by)), False) < 0:
                            n_ball_outside_hull += 1

                    for t in frame_tracks:
                        if t.label in ("player", "goalkeeper"):
                            px = (t.bbox.x1 + t.bbox.x2) / 2 * video.width
                            py = t.bbox.y2 * video.height
                            all_pts.append(("player", t.track_id, px, py))

                    if not all_pts:
                        continue

                    # Batch-transform all points in one perspectiveTransform call
                    raw_px = np.array([[p[2], p[3]] for p in all_pts], dtype=np.float32)
                    transformed = cv2.perspectiveTransform(
                        raw_px.reshape(-1, 1, 2), H_for(fi)
                    ).reshape(-1, 2)

                    for (role, tid, _, _), (x_cm, y_cm) in zip(all_pts, transformed):
                        x_cm, y_cm = float(x_cm), float(y_cm)
                        if role == "ball":
                            ball_pitch_pos[fi] = (x_cm, y_cm)
                        else:
                            frame_players[tid] = (x_cm, y_cm)

                    if frame_players:
                        player_pitch_pos[fi] = frame_players

                n_ball   = len(ball_pitch_pos)
                n_player = sum(len(v) for v in player_pitch_pos.values())
                print(f"  Projected: {n_ball} ball frames, {n_player} player-frame positions")
                if n_ball_outside_hull:
                    pct = 100 * n_ball_outside_hull / max(n_ball, 1)
                    print(f"  Note: {n_ball_outside_hull} ball frames ({pct:.0f}%) outside inlier hull"
                          f" (extrapolated -- less accurate but still projected)")

            else:
                print(f"  WARNING: Homography rejected - mean_err={mean_err:.1f}cm > 100cm "
                      f"or n_inliers={n_inliers} < 5. Falling back to pixel mode.")
        else:
            print("  WARNING: cv2.findHomography returned None. Falling back to pixel mode.")
    else:
        print(f"  WARNING: Only {len(pixel_pts)} keypoint pairs (need ≥ 6). Falling back to pixel mode.")

elif force_pixel:
    print("Pixel mode forced (--pixel flag).")
else:
    print("No pitch_keypoints.json found — running in pixel mode.")

use_pitch_mode = ball_pitch_pos is not None and player_pitch_pos is not None
print(f"\nEvent detection mode: {'PITCH (cm)' if use_pitch_mode else 'PIXEL'}")

# ── Optional frame-level debug trace ─────────────────────────────────────────
if debug_mode:
    import math as _math
    from collections import defaultdict as _defaultdict

    _by_frame: dict[int, list] = _defaultdict(list)
    for t in tracks:
        _by_frame[t.frame_index].append(t)

    print(f"\n{'='*80}")
    print(f"DEBUG TRACE  {debug_t_start:.1f}s to {debug_t_end:.1f}s")
    print(f"{'='*80}")
    print(f"{'frame':>6}  {'time':>6}  {'phase':<12}  {'owner':>6}  {'ball_ab':>7}  "
          f"{'closest':>8}  {'dist':>8}  {'speed':>8}  {'cand':>6}  {'cand_n':>6}")
    print("-"*80)

    _phase_name = "IDLE"
    _owner = None
    _owner_since = None
    _owner_from_flt = False
    _prev_owner = None
    _cand_tid = None
    _cand_cnt = 0
    _flight_frame = None
    _flight_speed = 0.0
    _flight_y: list = []
    _launch_cx = _launch_cy = None
    _vel_hist: list = []
    _vel_win = max(3, int(video.fps * 0.25))

    _poss_min_f = max(1, int(cfg.possession_min_seconds * video.fps))
    _shot_no_catch_f = max(1, int(cfg.shot_no_catch_seconds * video.fps))
    _poss_dist = cfg.possession_distance_threshold_cm if use_pitch_mode else 120.0
    _pass_speed = cfg.pass_min_speed_cms if use_pitch_mode else (150.0 / video.fps)
    _touch_speed = 200.0 if use_pitch_mode else (cfg.touch_min_ball_speed_px_per_s / video.fps)

    from ez_worker.analytics.events import (
        _get_ball, _closest_player, _is_ball_airborne, _compute_ball_velocity
    )

    _ball_vel_d = _compute_ball_velocity(
        _by_frame, sorted(_by_frame.keys()), cfg.ball_track_id, video,
        ball_pitch_pos, use_pitch_mode
    )

    for _fi in sorted(_by_frame.keys()):
        _t_s = _fi / video.fps
        if _t_s < debug_t_start or _t_s > debug_t_end:
            continue

        _fobs = _by_frame[_fi]
        _ball = _get_ball(_fobs, cfg.ball_track_id)
        _players = [o for o in _fobs
                    if o.label in ("player", "goalkeeper")
                    and o.source_label != "referee"
                    and (excluded_track_ids is None or o.track_id not in excluded_track_ids)]

        _vel = _ball_vel_d.get(_fi)
        if _vel:
            _vel_hist.append(_vel)
            if len(_vel_hist) > _vel_win:
                _vel_hist.pop(0)
        _spd = _math.hypot(*_vel) if _vel else 0.0

        _airborne = bool(_ball and _players and _is_ball_airborne(_ball, _players, video))

        _cl_str = "N/A"
        _d_str  = "N/A"
        if _ball and _players:
            try:
                _cl, _d = _closest_player(
                    _ball, _players, _fi, video,
                    use_pitch_mode, player_pitch_pos, ball_pitch_pos
                )
                _cl_str = str(_cl.track_id)
                _d_str  = f"{_d:.0f}"
            except Exception:
                pass

        print(f"{_fi:>6}  {_t_s:>6.2f}  {_phase_name:<12}  "
              f"{str(_owner) if _owner else '-':>6}  "
              f"{'YES' if _airborne else 'no':>7}  "
              f"{_cl_str:>8}  {_d_str:>8}  {_spd:>8.1f}  "
              f"{str(_cand_tid) if _cand_tid else '-':>6}  {_cand_cnt:>6}")

    print(f"{'='*80}\n")

# ── Run event detection ───────────────────────────────────────────────────────
events = detect_events(
    tracks=tracks,
    video=video,
    ball_track_id=cfg.ball_track_id,
    ball_pitch_pos=ball_pitch_pos,
    player_pitch_pos=player_pitch_pos,
    possession_distance_threshold_cm=cfg.possession_distance_threshold_cm,
    possession_distance_threshold_px=120.0,   # match run_full_pipeline.ps1 override
    possession_min_seconds=cfg.possession_min_seconds,
    pass_min_speed_cms=cfg.pass_min_speed_cms,
    pass_min_speed_px_per_s=150.0,            # lower to catch slow goalkeeper rolls
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

# ── Write output (back up previous events.json first) ────────────────────────
events_path = run_dir / "events.json"
if events_path.exists():
    backup_name = "events_pixel_fallback.json" if use_pitch_mode else "events_prev.json"
    backup_path = run_dir / backup_name
    shutil.copy2(events_path, backup_path)
    print(f"Backed up previous events → {backup_path.name}")

out = [e.model_dump(mode="json") for e in events]
events_path.write_text(json.dumps(out, indent=2))

print(f"\nWrote {len(events)} events to {events_path}")
for e in events:
    print(f"  {e.time_seconds:6.2f}s  {e.event_type:<14}  actor={e.actor_track_id}  "
          f"target={e.target_track_id}  {e.details}")
