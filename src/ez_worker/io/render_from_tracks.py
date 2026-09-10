"""Draw-only renderer: the video is drawn FROM the pipeline's tracks.

Why this exists
---------------
`stats_video.render_stats_video` re-runs YOLO detection, re-tracks with a second
tracker (`sv.ByteTrack`), and re-classifies teams with SigLIP **per frame** — then
Hungarian-matches its own detections back to `tracks.json` so the drawn ID
"matches events.json". That is two independent ID spaces glued together by an
80 px nearest-point match, and the match is ambiguous exactly when players
overlap, which is when identity matters most.

It also re-derives something the pipeline already computed. Measured on the
benchmark run: `tracks_with_teams.json` holds **one team per track, and not a
single track changes team**. `cluster-teams` gets this right. Only the video
flickers, because it decides again on every frame.

So this module draws the tracks the pipeline already produced:

  * players / goalkeepers / referees, IDs and team colours  <- tracks_with_teams.json
  * ball                                                    <- tracks.json
  * radar                                                   <- pitch_keypoints_per_frame.json

No detection, no tracking, no SigLIP. The drawn ID **is** the ID in
`match_report.json` by construction rather than by re-association, and the drawn
team is the team the stats used.

What this does NOT fix: two players of the same team crossing and swapping
identity. ByteTrack associates on IoU and Kalman motion with no appearance model,
so that swap happens upstream in `analyze` and is inherited here. It is corrected
post-hoc by `report_cleanup.detect_id_swaps`, not by drawing.

The original renderer is untouched; choose this one with
`render-stats-video --from-tracks`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from ez_worker.schemas import Event, TrackObservation, VideoMeta

from .stats_video import (
    COLORS,
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    REFEREE_CLASS_ID,
    _SavedBallAnnotator,
    _compute_possession_by_frame,
    _compute_team_stats,
    _draw_event_label,
    _draw_stats_panel,
    render_radar,
)

_PALETTE = sv.ColorPalette.from_hex(COLORS)
_ELLIPSE = sv.EllipseAnnotator(color=_PALETTE, thickness=2)
_LABEL = sv.LabelAnnotator(
    color=_PALETTE,
    text_color=sv.Color.from_hex("#FFFFFF"),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


def _load(run_dir: Path, name: str, default):
    p = run_dir / name
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _keypoints_for_frame(per_frame: dict, frame_index: int, wh) -> Optional[sv.KeyPoints]:
    """Keypoints from the nearest sampled frame, as an sv.KeyPoints in pixel space.

    The pitch model is not run here — `detect-pitch-keypoints --per-frame-stride`
    already sampled the clip, so the radar reuses that instead of paying for a
    second inference pass over every frame.
    """
    frames = per_frame.get("frames") or {}
    if not frames:
        return None
    keys = per_frame.get("_sorted")
    if keys is None:
        keys = sorted(int(k) for k in frames)
        per_frame["_sorted"] = keys
    if not keys:
        return None
    # nearest sampled frame
    lo, hi = 0, len(keys) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < frame_index:
            lo = mid + 1
        else:
            hi = mid
    cands = {keys[max(0, lo - 1)], keys[lo]}
    best = min(cands, key=lambda k: abs(k - frame_index))
    kps = frames[str(best)].get("keypoints") or {}
    if len(kps) < 4:
        return None

    n_vertices = 32
    xy = np.zeros((1, n_vertices, 2), dtype=np.float32)
    conf = np.zeros((1, n_vertices), dtype=np.float32)
    for idx_str, pos in kps.items():
        i = int(idx_str)
        if 0 <= i < n_vertices:
            xy[0, i] = pos
            conf[0, i] = 1.0
    try:
        return sv.KeyPoints(xy=xy, confidence=conf)
    except Exception:
        return None


def render_from_tracks(
    run_dir: Path,
    *,
    show_ball_trail: bool = True,
    show_radar: bool = True,
) -> Path:
    run_dir = Path(run_dir).resolve()

    meta = json.loads((run_dir / "video_meta.json").read_text(encoding="utf-8"))
    video = VideoMeta(**meta)
    video_path = Path(meta["path"])
    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")

    tracks_path = run_dir / "tracks_with_teams.json"
    has_teams = tracks_path.exists()
    if not has_teams:
        tracks_path = run_dir / "tracks.json"
    tracks = [TrackObservation(**t) for t in json.loads(tracks_path.read_text(encoding="utf-8"))]
    print(f"  Drawing from {tracks_path.name} ({len(tracks)} observations)")

    events = [Event(**e) for e in _load(run_dir, "events.json", [])]
    raw_stats = _load(run_dir, "player_stats.json", [])
    per_frame_kp = _load(run_dir, "pitch_keypoints_per_frame.json", {})
    referee_ids = set(_load(run_dir, "referee_track_ids.json", []))

    # One team per track, taken straight from the pipeline. No per-frame vote,
    # so a player cannot change team between two consecutive frames.
    team_of: dict[int, int | None] = {}
    for o in tracks:
        if o.label != "ball" and o.team_id is not None:
            team_of.setdefault(o.track_id, o.team_id)

    team_stats = _compute_team_stats(raw_stats, team_of)
    possession_by_frame = _compute_possession_by_frame(events, video.frame_count, team_of)
    event_by_frame = {e.frame_index: e for e in events}

    by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for o in tracks:
        by_frame[o.frame_index].append(o)

    radar_config = None
    if show_radar and per_frame_kp:
        try:
            from sports.configs.soccer import SoccerPitchConfiguration

            radar_config = SoccerPitchConfiguration()
        except Exception as exc:
            print(f"  Radar disabled ({exc})")
    elif show_radar:
        print("  Radar disabled: no pitch_keypoints_per_frame.json "
              "(run detect-pitch-keypoints --per-frame-stride 10)")

    out_path = run_dir / "stats_video.mp4"
    W, H = video.width, video.height
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), video.fps, (W, H))
    ball_trail = _SavedBallAnnotator() if show_ball_trail else None
    last_radar: Optional[np.ndarray] = None

    cap = cv2.VideoCapture(str(video_path))
    frame_index = 0
    try:
        with tqdm(total=video.frame_count, desc="render(from-tracks)") as bar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_tracks = by_frame.get(frame_index, [])
                players = [t for t in frame_tracks if t.label in ("player", "goalkeeper")
                           and t.track_id not in referee_ids]
                refs = [t for t in frame_tracks if t.label == "referee"
                        or t.track_id in referee_ids]
                ball = next((t for t in frame_tracks if t.label == "ball"), None)

                drawn = players + refs
                if drawn:
                    xyxy = np.array(
                        [[t.bbox.x1 * W, t.bbox.y1 * H, t.bbox.x2 * W, t.bbox.y2 * H]
                         for t in drawn], dtype=np.float32)
                    tracker_id = np.array([t.track_id for t in drawn], dtype=int)
                    dets = sv.Detections(xyxy=xyxy, tracker_id=tracker_id)

                    colour = []
                    for t in drawn:
                        if t.label == "referee" or t.track_id in referee_ids:
                            colour.append(REFEREE_CLASS_ID)
                        else:
                            team = team_of.get(t.track_id)
                            colour.append(team if team in (0, 1) else GOALKEEPER_CLASS_ID)
                    colour = np.array(colour, dtype=int)

                    labels = []
                    for t in drawn:
                        if t.label == "referee" or t.track_id in referee_ids:
                            labels.append("REF")
                        else:
                            team = team_of.get(t.track_id)
                            labels.append(f"T{team + 1}#{t.track_id}" if team in (0, 1)
                                          else f"#{t.track_id}")

                    frame = _ELLIPSE.annotate(frame, dets, custom_color_lookup=colour)
                    frame = _LABEL.annotate(frame, dets, labels, custom_color_lookup=colour)
                else:
                    dets = sv.Detections.empty()
                    colour = np.array([], dtype=int)

                ball_xy = None
                if ball is not None:
                    ball_xy = np.array([[(ball.bbox.x1 + ball.bbox.x2) / 2 * W,
                                         (ball.bbox.y1 + ball.bbox.y2) / 2 * H]])
                if ball_trail is not None:
                    # the trail wants a flat (x, y); the radar wants (1, 2)
                    frame = ball_trail.update(
                        frame, ball_xy[0] if ball_xy is not None else None)

                if radar_config is not None:
                    kps = _keypoints_for_frame(per_frame_kp, frame_index, (W, H))
                    radar = None
                    if kps is not None and len(dets) > 0:
                        try:
                            radar = render_radar(dets, kps, colour, radar_config, ball_xy)
                        except Exception:
                            radar = None
                    if radar is None:
                        radar = last_radar
                    else:
                        last_radar = radar
                    if radar is not None:
                        rh, rw = radar.shape[:2]
                        scale = (W // 2) / rw
                        small = cv2.resize(radar, (W // 2, int(rh * scale)))
                        frame = sv.draw_image(
                            frame, small, opacity=0.5,
                            rect=sv.Rect(x=W // 4, y=H - small.shape[0] - 10,
                                         width=small.shape[1], height=small.shape[0]),
                        )

                poss = possession_by_frame[frame_index] if frame_index < len(possession_by_frame) else (50.0, 50.0)
                _draw_stats_panel(frame, team_stats, poss, has_teams)
                evt = event_by_frame.get(frame_index)
                if evt is not None:
                    _draw_event_label(frame, evt)

                writer.write(frame)
                frame_index += 1
                bar.update(1)
    finally:
        cap.release()
        writer.release()

    print(f"  Wrote {out_path}")
    return out_path
