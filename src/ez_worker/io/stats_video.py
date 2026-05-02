from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from ez_worker.schemas import BBox, Event, TrackObservation, VideoMeta


# Tutorial-exact constants
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3
STRIDE = 60

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
REFEREE_COLOR_IDX = 3

_COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(color=_COLOR_PALETTE, thickness=2)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=_COLOR_PALETTE,
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

POSSESSION_WINDOW_FRAMES = 75

TEAM_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (220, 80, 60),
    1: (60, 180, 220),
    2: (160, 160, 160),
}


# ---------------------------------------------------------------------------
# Ball annotator using saved track data (fallback when no dedicated model)
# ---------------------------------------------------------------------------

class _SavedBallAnnotator:
    def __init__(self, radius: int = 6, buffer_size: int = 10) -> None:
        from collections import deque
        self._buf: deque = deque(maxlen=buffer_size)
        self.radius = radius
        try:
            self._palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        except Exception:
            self._palette = None

    def update(self, frame: np.ndarray, xy: np.ndarray | None) -> np.ndarray:
        self._buf.append(xy)
        for i, pos in enumerate(self._buf):
            if pos is None:
                continue
            r = max(1, int(1 + i * (self.radius - 1) / max(len(self._buf) - 1, 1)))
            color = self._palette.by_idx(i).as_bgr() if self._palette else (0, 255, 255)
            cv2.circle(frame, (int(pos[0]), int(pos[1])), r, color, 2)
        return frame


# ---------------------------------------------------------------------------
# Ball detector (tutorial: BallTracker + InferenceSlicer)
# ---------------------------------------------------------------------------

class _BallDetector:
    def __init__(self, model_path: Path) -> None:
        from ultralytics import YOLO
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = YOLO(str(model_path)).to(device)
        from sports.common.ball import BallTracker, BallAnnotator
        self.tracker = BallTracker(buffer_size=20)
        self.annotator = BallAnnotator(radius=6, buffer_size=10)

        def _callback(image_slice: np.ndarray) -> sv.Detections:
            result = self._model(image_slice, imgsz=640, verbose=False)[0]
            return sv.Detections.from_ultralytics(result)

        self._slicer = sv.InferenceSlicer(
            callback=_callback,
            overlap_filter=sv.OverlapFilter.NONE,
            slice_wh=(640, 640),
            overlap_ratio_wh=None,
            overlap_wh=(0, 0),
        )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        detections = self._slicer(frame).with_nms(threshold=0.1)
        return self.tracker.update(detections)


# ---------------------------------------------------------------------------
# Tutorial-exact helpers (copied from examples/soccer/main.py)
# ---------------------------------------------------------------------------

def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections,
) -> np.ndarray:
    if len(goalkeepers) == 0:
        return np.array([], dtype=int)
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_xy = players_xy[players_team_id == 0]
    team_1_xy = players_xy[players_team_id == 1]
    if len(team_0_xy) == 0 or len(team_1_xy) == 0:
        return np.zeros(len(goalkeepers), dtype=int)
    team_0_centroid = team_0_xy.mean(axis=0)
    team_1_centroid = team_1_xy.mean(axis=0)
    goalkeepers_team_id = []
    for gk_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
        dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray,
    config,
    ball_xy_px: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Tutorial-exact radar: builds fresh ViewTransformer from live keypoints each frame."""
    from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
    from sports.common.view import ViewTransformer

    if keypoints.xy is None or len(keypoints.xy) == 0:
        return None

    # Filter by position > 1px (tutorial) AND by confidence (our addition:
    # the pitch model returns low-confidence guesses for unseen keypoints — these
    # skew findHomography and pull far-side players toward the centre of the radar)
    pos_mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    if keypoints.confidence is not None and len(keypoints.confidence) > 0:
        conf_mask = keypoints.confidence[0] > 0.5
        mask = pos_mask & conf_mask
    else:
        mask = pos_mask

    if mask.sum() < 6:
        return None
    try:
        transformer = ViewTransformer(
            source=keypoints.xy[0][mask].astype(np.float32),
            target=np.array(config.vertices)[mask].astype(np.float32),
        )
    except Exception:
        return None

    radar = draw_pitch(config=config)

    if len(detections) > 0:
        xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_xy = transformer.transform_points(points=xy)
        for idx, color_hex in enumerate(COLORS):
            mask_c = color_lookup == idx
            if mask_c.any():
                radar = draw_points_on_pitch(
                    config=config,
                    xy=transformed_xy[mask_c],
                    face_color=sv.Color.from_hex(color_hex),
                    radius=20,
                    pitch=radar,
                )

    if ball_xy_px is not None:
        ball_cm = transformer.transform_points(ball_xy_px[np.newaxis].astype(np.float32))
        radar = draw_points_on_pitch(
            config=config,
            xy=ball_cm,
            face_color=sv.Color.from_hex('#FFFFFF'),
            radius=15,
            pitch=radar,
        )

    return radar


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_stats_video(
    run_dir: Path,
    pitch_model_path: Optional[Path] = None,
    ball_model_path: Optional[Path] = None,
    player_model_path: Optional[Path] = None,
    show_keypoints: bool = False,
) -> Path:
    import torch
    from ultralytics import YOLO
    from sports.common.team import TeamClassifier
    from sports.configs.soccer import SoccerPitchConfiguration

    run_dir = run_dir.resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    video_path = Path(summary["video_path"])

    tracks_with_teams_path = run_dir / "tracks_with_teams.json"
    tracks_path = run_dir / "tracks.json"
    has_teams = tracks_with_teams_path.exists()
    raw_tracks = json.loads(
        (tracks_with_teams_path if has_teams else tracks_path).read_text(encoding="utf-8")
    )
    tracks = [_parse_track(t) for t in raw_tracks]

    events_path = run_dir / "events_analysis_ready.json"
    if not events_path.exists():
        events_path = run_dir / "events.json"
    raw_events = json.loads(events_path.read_text(encoding="utf-8"))
    events = [_parse_event(e) for e in raw_events]

    stats_path = run_dir / "player_stats_analysis_ready.json"
    if not stats_path.exists():
        stats_path = run_dir / "player_stats.json"
    raw_stats = json.loads(stats_path.read_text(encoding="utf-8"))

    video_meta_raw = json.loads((run_dir / "video_meta.json").read_text(encoding="utf-8"))
    video = VideoMeta(
        path=video_path,
        fps=video_meta_raw["fps"],
        frame_count=video_meta_raw["frame_count"],
        width=video_meta_raw["width"],
        height=video_meta_raw["height"],
    )

    # Pre-computed data used only for stats panel and events
    team_id_by_track = {t.track_id: t.team_id for t in tracks if t.team_id is not None}
    team_stats = _compute_team_stats(raw_stats, team_id_by_track)
    possession_by_frame = _compute_possession_by_frame(events, video.frame_count, team_id_by_track)

    tracks_by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for t in tracks:
        tracks_by_frame[t.frame_index].append(t)
    events_by_frame: dict[int, list[Event]] = defaultdict(list)
    for e in events:
        events_by_frame[e.frame_index].append(e)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Player model (auto-discover baseline)
    _candidate_player = [player_model_path] if player_model_path else []
    _candidate_player += [Path("artifacts/training/roboflow_detector_v1_light/weights/best.pt")]
    player_model = None
    for p in _candidate_player:
        if p is not None and Path(p).exists():
            player_model = YOLO(str(p)).to(device)
            print(f"  Player model: {p}")
            break
    if player_model is None:
        raise RuntimeError("Player model not found — pass --player-model-path or add to artifacts/")

    # Pitch model
    pitch_model = None
    radar_config = None
    _candidate_pitch = [pitch_model_path] if pitch_model_path else []
    _candidate_pitch += [
        Path("artifacts/pitch/football-pitch-detectionV2.pt"),
        Path("artifacts/pitch/football-pitch-detection.pt"),
    ]
    for p in _candidate_pitch:
        if p is not None and Path(p).exists():
            pitch_model = YOLO(str(p)).to(device)
            radar_config = SoccerPitchConfiguration()
            print(f"  Pitch model: {p}")
            break
    if pitch_model is None:
        print("  Warning: no pitch model found — radar disabled")

    # Keypoint debug annotators (created once, used per-frame when show_keypoints=True)
    kp_vertex_annotator = None
    kp_edge_annotator = None
    if show_keypoints and radar_config is not None:
        kp_vertex_annotator = sv.VertexLabelAnnotator(
            color=[sv.Color.from_hex(c) for c in radar_config.colors],
            text_color=sv.Color.from_hex('#FFFFFF'),
            border_radius=5,
            text_thickness=1,
            text_scale=0.5,
            text_padding=5,
        )
        kp_edge_annotator = sv.EdgeAnnotator(
            color=sv.Color.from_hex('#FF1493'),
            thickness=2,
            edges=radar_config.edges,
        )

    # Tutorial-exact pre-loop: collect player crops at stride=60, fit TeamClassifier
    print(f"  Collecting player crops at stride={STRIDE} for team classification...")
    team_classifier = TeamClassifier(device=device)
    crops: List[np.ndarray] = []
    for frame in tqdm(sv.get_video_frames_generator(str(video_path), stride=STRIDE), desc='crops'):
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, dets[dets.class_id == PLAYER_CLASS_ID])
    team_classifier.fit(crops)
    print(f"  TeamClassifier fitted on {len(crops)} crops.")

    # Tutorial-exact ByteTrack
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    # Ball detector (optional)
    ball_detector = None
    _candidate_ball = [ball_model_path] if ball_model_path else []
    _candidate_ball += [Path("artifacts/ball/football-ball-detection.pt")]
    for p in _candidate_ball:
        if p is not None and Path(p).exists():
            try:
                ball_detector = _BallDetector(Path(p))
                print(f"  Ball detector: {p}")
            except Exception as e:
                print(f"  Ball detector init failed: {e}")
            break

    saved_ball_annotator = _SavedBallAnnotator(radius=6, buffer_size=10)

    output_path = run_dir / "stats_video.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video.fps,
        (video.width, video.height),
    )

    # Last successfully rendered radar — shown on frames where keypoints fail
    last_radar: Optional[np.ndarray] = None

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_tracks = tracks_by_frame.get(frame_index, [])

            # --- Tutorial-exact: both models run on the same clean frame ---

            # 1. Pitch keypoints (clean frame, no imgsz override — tutorial-exact)
            keypoints = None
            if pitch_model is not None:
                result = pitch_model(frame, verbose=False)[0]
                keypoints = sv.KeyPoints.from_ultralytics(result)

            # 2. Player/GK/referee detection (clean frame, imgsz=1280 — tutorial-exact)
            result = player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            # 3. Split by class (tutorial-exact)
            players = detections[detections.class_id == PLAYER_CLASS_ID]
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            referees = detections[detections.class_id == REFEREE_CLASS_ID]

            # 4. Team classification via SigLIP (tutorial-exact, per-frame)
            player_crops = get_crops(frame, players)
            if len(player_crops) > 0:
                players_team_id = team_classifier.predict(player_crops)
            else:
                players_team_id = np.array([], dtype=int)

            # 5. GK team resolution (tutorial-exact)
            try:
                goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
            except Exception:
                goalkeepers_team_id = np.zeros(len(goalkeepers), dtype=int)

            # 6. Merge + color_lookup (tutorial-exact)
            all_dets = sv.Detections.merge([players, goalkeepers, referees])
            color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees),
                dtype=int,
            )
            labels = (
                [str(tid) for tid in all_dets.tracker_id]
                if all_dets.tracker_id is not None
                else [""] * len(all_dets)
            )

            # 7. Annotate ellipses (tutorial-exact)
            if len(all_dets) > 0:
                frame = ELLIPSE_ANNOTATOR.annotate(frame, all_dets, custom_color_lookup=color_lookup)
                frame = ELLIPSE_LABEL_ANNOTATOR.annotate(frame, all_dets, labels, custom_color_lookup=color_lookup)

            # 7b. Pitch keypoint debug overlay (--show-keypoints)
            if show_keypoints and keypoints is not None and kp_vertex_annotator is not None:
                try:
                    frame = kp_edge_annotator.annotate(frame, keypoints)
                    frame = kp_vertex_annotator.annotate(frame, keypoints, radar_config.labels)
                except Exception:
                    pass

            # 8. Ball: dedicated detector if available, else saved tracks fallback
            ball_xy_px = None
            for t in frame_tracks:
                if t.label == "ball":
                    bx = (t.bbox.x1 + t.bbox.x2) / 2 * video.width
                    by = (t.bbox.y1 + t.bbox.y2) / 2 * video.height
                    ball_xy_px = np.array([bx, by])
                    break
            ball_detections = sv.Detections.empty()
            if ball_detector is not None:
                try:
                    ball_detections = ball_detector.detect(frame)
                    if len(ball_detections) > 0:
                        ball_xy_px = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0]
                except Exception:
                    ball_detections = sv.Detections.empty()

            frame = saved_ball_annotator.update(frame, ball_xy_px)
            if ball_detector is not None and len(ball_detections) > 0:
                try:
                    frame = ball_detector.annotator.annotate(frame, ball_detections)
                except Exception:
                    pass

            # 9. Event labels (from pre-computed JSON)
            for event in events_by_frame.get(frame_index, []):
                _draw_event_label(frame, event)

            # 10. Stats panel (from pre-computed JSON)
            poss = possession_by_frame.get(frame_index, {})
            _draw_stats_panel(frame, team_stats, poss, has_teams)

            # 11. Radar — tutorial-exact: live keypoints + live detections from same clean frame
            # Falls back to last good radar when keypoints are insufficient this frame
            if radar_config is not None:
                try:
                    radar = None
                    if keypoints is not None:
                        radar = render_radar(all_dets, keypoints, color_lookup, radar_config, ball_xy_px)
                    if radar is not None:
                        last_radar = radar
                    elif last_radar is not None:
                        radar = last_radar
                    if radar is not None:
                        fh, fw = frame.shape[:2]
                        radar_resized = sv.resize_image(radar, (fw // 2, fh // 2))
                        rh, rw = radar_resized.shape[:2]
                        rect = sv.Rect(x=fw // 2 - rw // 2, y=fh - rh, width=rw, height=rh)
                        frame = sv.draw_image(frame, radar_resized, opacity=0.5, rect=rect)
                except Exception:
                    pass

            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    return output_path


# ---------------------------------------------------------------------------
# Stats overlay (our addition — not in tutorial)
# ---------------------------------------------------------------------------

def _draw_stats_panel(
    frame: np.ndarray,
    team_stats: dict[int, dict],
    possession: dict[int, float],
    has_teams: bool,
) -> None:
    h, w = frame.shape[:2]
    panel_w = 260
    panel_h = 170 if has_teams else 80
    margin = 12
    x0 = w - panel_w - margin
    y0 = margin

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if not has_teams:
        cv2.putText(frame, "Run apply-team-clusters first", (x0 + 6, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "to see team stats", (x0 + 6, y0 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        return

    col_a = x0 + 8
    col_b = x0 + panel_w // 2 + 4
    row_h = 26
    a = team_stats.get(0, {})
    b = team_stats.get(1, {})
    color_a = (60, 20, 255)
    color_b = (255, 191, 0)

    def text(x, row, s, color, scale=0.42):
        cv2.putText(frame, s, (x, y0 + row * row_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    text(col_a, 0, f"TEAM A ({a.get('players', 0)}p)", color_a, 0.44)
    text(col_b, 0, f"TEAM B ({b.get('players', 0)}p)", color_b, 0.44)
    cv2.line(frame, (x0, y0 + row_h), (x0 + panel_w, y0 + row_h), (80, 80, 80), 1)
    cv2.line(frame, (x0 + panel_w // 2, y0), (x0 + panel_w // 2, y0 + panel_h), (80, 80, 80), 1)

    poss_a = int(possession.get(0, 0) * 100)
    poss_b = int(possession.get(1, 0) * 100)
    if poss_a + poss_b == 0:
        poss_a = poss_b = 50

    text(col_a, 1, f"Poss  {poss_a}%", (220, 220, 220))
    text(col_b, 1, f"Poss  {poss_b}%", (220, 220, 220))
    text(col_a, 2, f"Pass  {a.get('passes', 0)}", (220, 220, 220))
    text(col_b, 2, f"Pass  {b.get('passes', 0)}", (220, 220, 220))
    text(col_a, 3, f"Touch {a.get('touches', 0)}", (220, 220, 220))
    text(col_b, 3, f"Touch {b.get('touches', 0)}", (220, 220, 220))
    text(col_a, 4, f"Shot  {a.get('shots', 0)}", (220, 220, 220))
    text(col_b, 4, f"Shot  {b.get('shots', 0)}", (220, 220, 220))
    text(col_a, 5, f"Dist  {a.get('distance_km', 0):.1f}km", (180, 180, 180))
    text(col_b, 5, f"Dist  {b.get('distance_km', 0):.1f}km", (180, 180, 180))


def _draw_event_label(frame: np.ndarray, event: Event) -> None:
    label = event.event_type.replace("_", " ")
    cv2.putText(frame, label, (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def _compute_team_stats(
    raw_stats: list[dict],
    team_id_by_track: dict[int, int | None],
) -> dict[int, dict]:
    teams: dict[int, dict] = {
        0: {"players": 0, "passes": 0, "touches": 0, "shots": 0, "distance_px": 0.0},
        1: {"players": 0, "passes": 0, "touches": 0, "shots": 0, "distance_px": 0.0},
    }
    for stat in raw_stats:
        tid = stat.get("track_id")
        team_id = team_id_by_track.get(tid)
        if team_id not in teams:
            continue
        t = teams[team_id]
        t["players"] += 1
        t["passes"] += stat.get("pass_count", 0)
        t["touches"] += stat.get("touch_count", 0)
        t["shots"] += stat.get("shot_count", 0)
        t["distance_px"] += stat.get("approx_distance_px", 0.0)
    for t in teams.values():
        t["distance_km"] = round(t["distance_px"] * 0.1 / 1000, 2)
    return teams


def _compute_possession_by_frame(
    events: list[Event],
    total_frames: int,
    team_id_by_track: dict[int, int | None],
) -> dict[int, dict[int, float]]:
    touch_events = [
        e for e in events
        if e.event_type in {"ball_touch", "pass", "shot"} and e.actor_track_id is not None
    ]
    result: dict[int, dict[int, float]] = {}
    for frame_idx in range(total_frames):
        window_start = max(0, frame_idx - POSSESSION_WINDOW_FRAMES)
        window_touches = [e for e in touch_events if window_start <= e.frame_index <= frame_idx]
        counts: dict[int, int] = defaultdict(int)
        for e in window_touches:
            tid = e.actor_track_id
            team_id = team_id_by_track.get(tid)
            if team_id in (0, 1):
                counts[team_id] += 1
        total = sum(counts.values())
        result[frame_idx] = {0: 0.5, 1: 0.5} if total == 0 else {k: counts[k] / total for k in (0, 1)}
    return result


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_track(raw: dict) -> TrackObservation:
    bbox = raw["bbox"]
    return TrackObservation(
        frame_index=raw["frame_index"],
        track_id=raw["track_id"],
        label=raw["label"],
        source_label=raw.get("source_label"),
        confidence=raw.get("confidence", 1.0),
        bbox=BBox(x1=bbox["x1"], y1=bbox["y1"], x2=bbox["x2"], y2=bbox["y2"]),
        team_id=raw.get("team_id"),
    )


def _parse_event(raw: dict) -> Event:
    return Event(
        frame_index=raw["frame_index"],
        time_seconds=raw["time_seconds"],
        event_type=raw["event_type"],
        actor_track_id=raw.get("actor_track_id"),
        target_track_id=raw.get("target_track_id"),
        details=raw.get("details", {}),
    )
