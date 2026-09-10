from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment
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

        import inspect as _inspect
        _slicer_params = _inspect.signature(sv.InferenceSlicer.__init__).parameters
        _slicer_kwargs: dict = dict(
            callback=_callback,
            slice_wh=(640, 640),
            overlap_wh=(0, 0),
        )
        # supervision ≥0.25 renamed overlap_filter → overlap_filter_strategy and removed overlap_ratio_wh
        if "overlap_filter_strategy" in _slicer_params:
            _slicer_kwargs["overlap_filter_strategy"] = sv.OverlapFilter.NONE
        else:
            _slicer_kwargs["overlap_filter"] = sv.OverlapFilter.NONE
            _slicer_kwargs["overlap_ratio_wh"] = None
        self._slicer = sv.InferenceSlicer(**_slicer_kwargs)

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
    disable_ball_detector: bool = False,
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
    # Per-track team vote accumulator — records which team each track was shown as
    # in the video. Written to video_track_teams.json after render for apply_clusters to use.
    from collections import Counter as _Counter
    _track_team_votes: dict[int, _Counter] = defaultdict(lambda: _Counter())

    # Ball detector (optional)
    ball_detector = None
    if disable_ball_detector:
        print("  Ball detector DISABLED — drawing ball from saved tracks.json positions")
    else:
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
    # Tracker IDs ever seen as referee — override their colour even on frames
    # where the model misclassifies them as a player
    known_referee_tids: set[int] = set()
    # Per-GK team lock — goalkeepers never switch sides so first stable
    # assignment is locked to prevent frame-to-frame flipping
    gk_team_cache: dict[int, int] = {}

    # Stable per-track team from pipeline clustering (prevents per-frame flickering).
    # Loaded from player_labels.json written by apply-team-clusters before this step.
    _stable_teams: dict[int, int] = {}
    _pl_path = run_dir / "player_labels.json"
    if _pl_path.exists():
        _pl_data = json.loads(_pl_path.read_text(encoding="utf-8"))
        for _tid_str, _label in _pl_data.items():
            if _label.startswith("T1"):
                _stable_teams[int(_tid_str)] = 0
            elif _label.startswith("T2"):
                _stable_teams[int(_tid_str)] = 1

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

            # Record referee tracker IDs so frames where model misclassifies
            # them as players can still receive the correct colour
            if referees.tracker_id is not None:
                for _tid in referees.tracker_id:
                    known_referee_tids.add(int(_tid))

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

            # Stabilise GK team — goalkeepers never switch sides
            if goalkeepers.tracker_id is not None and len(goalkeepers_team_id) > 0:
                stable_gk = goalkeepers_team_id.copy()
                for i, _tid in enumerate(goalkeepers.tracker_id):
                    _tid = int(_tid)
                    if _tid in gk_team_cache:
                        stable_gk[i] = gk_team_cache[_tid]
                    else:
                        gk_team_cache[_tid] = int(stable_gk[i])
                goalkeepers_team_id = stable_gk

            # 6. Merge + color_lookup (tutorial-exact)
            all_dets = sv.Detections.merge([players, goalkeepers, referees])
            color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees),
                dtype=int,
            )

            # Override: any tracker_id previously seen as referee → force yellow
            # (fixes frames where model misclassifies the referee as a player)
            if all_dets.tracker_id is not None:
                for _i, _tid in enumerate(all_dets.tracker_id):
                    if int(_tid) in known_referee_tids:
                        color_lookup[_i] = REFEREE_CLASS_ID

            # Match each live detection to a pre-tracked player from tracks.json
            # so the displayed ID matches events.json / match_report.json.
            # Uses Hungarian assignment (1-to-1) so two live detections can never
            # both claim the same pipeline track — eliminates most overlap ID swaps.
            _pretrack_pts = [
                (
                    (t.bbox.x1 + t.bbox.x2) / 2 * video.width,
                    t.bbox.y2 * video.height,
                    t.track_id,
                )
                for t in frame_tracks
                if t.label in ("player", "goalkeeper")
            ]
            _live_bcs = (
                all_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                if len(all_dets) > 0 else np.empty((0, 2))
            )

            # Build per-frame live_idx → pipeline_tid map via Hungarian assignment.
            _frame_tid_map: dict[int, int] = {}
            if _pretrack_pts and len(_live_bcs) > 0:
                _MATCH_DIST = 80.0
                n_live = len(_live_bcs)
                n_pre = len(_pretrack_pts)
                cost = np.full((n_live, n_pre), fill_value=1e6)
                for _i, (_lx, _ly) in enumerate(_live_bcs):
                    for _j, (_px, _py, _) in enumerate(_pretrack_pts):
                        _d = ((_lx - _px) ** 2 + (_ly - _py) ** 2) ** 0.5
                        if _d <= _MATCH_DIST:
                            cost[_i, _j] = _d
                _rows, _cols = linear_sum_assignment(cost)
                for _i, _j in zip(_rows, _cols):
                    if cost[_i, _j] < 1e6:
                        _frame_tid_map[_i] = _pretrack_pts[_j][2]

            def _resolve_tid(live_idx: int, live_tid: int) -> int:
                return _frame_tid_map.get(live_idx, live_tid)

            # Override color_lookup with stable pipeline team to prevent flickering.
            # Live per-frame classifier still runs for new/unmatched players.
            if all_dets.tracker_id is not None and _stable_teams:
                for _si, _stid in enumerate(all_dets.tracker_id):
                    if color_lookup[_si] == REFEREE_CLASS_ID:
                        continue
                    _spid = _resolve_tid(_si, int(_stid))
                    if _spid in _stable_teams:
                        color_lookup[_si] = _stable_teams[_spid]

            if all_dets.tracker_id is not None:
                labels = [
                    "REF" if color_lookup[i] == REFEREE_CLASS_ID
                    else f"T{color_lookup[i]+1}#{_resolve_tid(i, int(tid))}"
                    for i, tid in enumerate(all_dets.tracker_id)
                ]
            else:
                labels = [""] * len(all_dets)

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

            # Record per-track team shown this frame using pipeline track IDs
            if all_dets.tracker_id is not None:
                for _vi, _vtid in enumerate(all_dets.tracker_id):
                    _vteam = int(color_lookup[_vi])
                    if _vteam != REFEREE_CLASS_ID:
                        _pipeline_tid = _resolve_tid(_vi, int(_vtid))
                        _track_team_votes[_pipeline_tid][_vteam] += 1

            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    if known_referee_tids:
        (run_dir / "referee_track_ids.json").write_text(
            json.dumps(sorted(known_referee_tids)), encoding="utf-8"
        )

    # Write majority team per track as shown in the video.
    # apply_clusters reads this on a second pass to ensure label consistency.
    if _track_team_votes:
        video_track_teams = {
            str(tid): int(votes.most_common(1)[0][0])
            for tid, votes in _track_team_votes.items()
            if votes
        }
        (run_dir / "video_track_teams.json").write_text(
            json.dumps(video_track_teams, indent=2), encoding="utf-8"
        )

    return output_path


# ---------------------------------------------------------------------------
# Stats overlay (our addition — not in tutorial)
# ---------------------------------------------------------------------------

def _draw_stats_panel(
    frame: np.ndarray,
    team_stats: dict[int, dict],
    possession: dict[int, float],
    has_teams: bool,
    show_distance: bool = True,
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
    if show_distance:
        text(col_a, 5, f"Dist  {a.get('distance_km', 0):.1f}km", (180, 180, 180))
        text(col_b, 5, f"Dist  {b.get('distance_km', 0):.1f}km", (180, 180, 180))
    else:
        text(col_a, 5, "Visible players", (180, 180, 180))
        text(col_b, 5, "Visible players", (180, 180, 180))


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
# Spatial analysis helpers (tutorial 1:20–1:26)
# ---------------------------------------------------------------------------

def replace_outlier_based_on_distance(
    positions: list,
    max_distance: float = 500.0,
) -> list:
    """Return positions with jumps > max_distance cm removed (non-destructive)."""
    result = []
    for pos in positions:
        if len(result) == 0 or np.linalg.norm(np.array(pos) - np.array(result[-1])) <= max_distance:
            result.append(pos)
    return result


def interpolate_ball_path(positions: list, max_gap: int = 30) -> list:
    """Fill short None gaps with linear interpolation.

    When the ball is airborne it is often not detected (None). Interpolating
    between the last ground contact and next ground contact gives the correct
    straight-line ground-plane path without needing 3D tracking.
    Gaps longer than max_gap frames are left as None (ball truly lost).
    """
    result = list(positions)
    n = len(result)
    i = 0
    while i < n:
        if result[i] is None:
            gap_start = i
            while i < n and result[i] is None:
                i += 1
            gap_end = i
            gap_len = gap_end - gap_start
            if gap_len <= max_gap and gap_start > 0 and gap_end < n:
                p0 = result[gap_start - 1]
                p1 = result[gap_end]
                if p0 is not None and p1 is not None:
                    for k in range(gap_len):
                        t = (k + 1) / (gap_len + 1)
                        result[gap_start + k] = p0 + t * (np.asarray(p1) - np.asarray(p0))
        else:
            i += 1
    return result


def render_spatial_video(
    run_dir: Path,
    pitch_model_path: Optional[Path] = None,
    ball_model_path: Optional[Path] = None,
    disable_ball_detector: bool = False,
) -> Path:
    """Top-down spatial analysis video: Voronoi + player dots + ball trajectory.

    Completely separate from stats_video.mp4 — does not touch it.
    Uses pre-computed tracks_with_teams.json for team assignments so no
    player model re-inference is needed.  Output: <run_dir>/spatial_video.mp4.
    """
    import torch
    from collections import deque
    from ultralytics import YOLO
    from sports.configs.soccer import SoccerPitchConfiguration
    from sports.annotators.soccer import (
        draw_pitch,
        draw_points_on_pitch,
        draw_pitch_voronoi_diagram,
        draw_paths_on_pitch,
    )
    from sports.common.view import ViewTransformer

    run_dir = run_dir.resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    video_path = Path(summary["video_path"])

    tracks_with_teams_path = run_dir / "tracks_with_teams.json"
    tracks_path = run_dir / "tracks.json"
    raw_tracks = json.loads(
        (tracks_with_teams_path if tracks_with_teams_path.exists() else tracks_path)
        .read_text(encoding="utf-8")
    )
    tracks = [_parse_track(t) for t in raw_tracks]
    tracks_by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for t in tracks:
        tracks_by_frame[t.frame_index].append(t)

    video_meta_raw = json.loads((run_dir / "video_meta.json").read_text(encoding="utf-8"))
    video = VideoMeta(
        path=video_path,
        fps=video_meta_raw["fps"],
        frame_count=video_meta_raw["frame_count"],
        width=video_meta_raw["width"],
        height=video_meta_raw["height"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = SoccerPitchConfiguration()

    # Pitch model (required for homography)
    pitch_model = None
    for p in ([pitch_model_path] if pitch_model_path else []) + [
        Path("artifacts/pitch/football-pitch-detectionV2.pt"),
        Path("artifacts/pitch/football-pitch-detection.pt"),
    ]:
        if p is not None and Path(p).exists():
            pitch_model = YOLO(str(p)).to(device)
            print(f"  Spatial pitch model: {p}")
            break
    if pitch_model is None:
        raise RuntimeError("Pitch model not found — pass --pitch-model-path or place in artifacts/pitch/")

    # Ball model (optional — falls back to saved tracks.json positions)
    ball_detector = None
    if disable_ball_detector:
        print("  Spatial ball detector DISABLED — using saved tracks.json positions")
    else:
        for p in ([ball_model_path] if ball_model_path else []) + [
            Path("artifacts/ball/football-ball-detection.pt"),
        ]:
            if p is not None and Path(p).exists():
                try:
                    ball_detector = _BallDetector(Path(p))
                    print(f"  Spatial ball detector: {p}")
                except Exception as exc:
                    print(f"  Ball detector init failed: {exc}")
                break

    # Output dimensions = pitch diagram size (fixed by draw_pitch defaults)
    sample_pitch = draw_pitch(config=config)
    spatial_h, spatial_w = sample_pitch.shape[:2]

    output_path = run_dir / "spatial_video.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video.fps,
        (spatial_w, spatial_h),
    )

    # Averaged homography: keep last 5 valid H matrices (tutorial window=5)
    H_deque: deque = deque(maxlen=5)
    # Accumulated ball positions in pitch cm, capped to last ~5 s
    ball_path_cm: list = []
    BALL_TRAIL_MAX = 150

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_index = 0
    try:
        with tqdm(total=video.frame_count, desc="spatial") as pbar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_tracks = tracks_by_frame.get(frame_index, [])

                # 1. Pitch keypoints → ViewTransformer → collect H matrix
                result = pitch_model(frame, verbose=False)[0]
                keypoints = sv.KeyPoints.from_ultralytics(result)

                if keypoints.xy is not None and len(keypoints.xy) > 0:
                    pos_mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
                    if keypoints.confidence is not None and len(keypoints.confidence) > 0:
                        conf_mask = keypoints.confidence[0] > 0.5
                        mask = pos_mask & conf_mask
                    else:
                        mask = pos_mask
                    if mask.sum() >= 6:
                        try:
                            transformer = ViewTransformer(
                                source=keypoints.xy[0][mask].astype(np.float32),
                                target=np.array(config.vertices)[mask].astype(np.float32),
                            )
                            if transformer.m is not None:
                                H_deque.append(transformer.m.copy())
                        except Exception:
                            pass

                avg_H = np.mean(np.stack(list(H_deque)), axis=0) if len(H_deque) > 0 else None

                # 2. Player positions from pre-computed tracks (team_id already assigned)
                all_px, all_team = [], []
                for t in frame_tracks:
                    if t.label not in ("player", "goalkeeper"):
                        continue
                    bx = (t.bbox.x1 + t.bbox.x2) / 2 * video.width
                    by = t.bbox.y2 * video.height
                    all_px.append([bx, by])
                    all_team.append(t.team_id if t.team_id is not None else -1)

                team_0_cm = np.empty((0, 2), dtype=np.float32)
                team_1_cm = np.empty((0, 2), dtype=np.float32)
                if avg_H is not None and len(all_px) > 0:
                    pts = np.array(all_px, dtype=np.float32)
                    transformed = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), avg_H).reshape(-1, 2)
                    teams = np.array(all_team)
                    team_0_cm = transformed[teams == 0]
                    team_1_cm = transformed[teams == 1]

                # 3. Ball position (live detector preferred, tracks.json fallback)
                ball_xy_px = None
                for t in frame_tracks:
                    if t.label == "ball":
                        bx = (t.bbox.x1 + t.bbox.x2) / 2 * video.width
                        by = (t.bbox.y1 + t.bbox.y2) / 2 * video.height
                        ball_xy_px = np.array([bx, by])
                        break
                if ball_detector is not None:
                    try:
                        ball_dets = ball_detector.detect(frame)
                        if len(ball_dets) > 0:
                            ball_xy_px = ball_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0]
                    except Exception:
                        pass

                ball_cm_pos = None
                if avg_H is not None and ball_xy_px is not None:
                    ball_cm_pos = cv2.perspectiveTransform(
                        np.array([[ball_xy_px]], dtype=np.float32), avg_H
                    ).reshape(2)

                # Accumulate ball trail (capped)
                ball_path_cm.append(ball_cm_pos)
                if len(ball_path_cm) > BALL_TRAIL_MAX:
                    ball_path_cm = ball_path_cm[-BALL_TRAIL_MAX:]

                # 4. Build spatial frame: Voronoi background
                if len(team_0_cm) > 0 and len(team_1_cm) > 0:
                    spatial_frame = draw_pitch_voronoi_diagram(
                        config=config,
                        team_1_xy=team_0_cm,
                        team_2_xy=team_1_cm,
                        team_1_color=sv.Color.from_hex(COLORS[0]),
                        team_2_color=sv.Color.from_hex(COLORS[1]),
                        opacity=0.35,
                    )
                else:
                    spatial_frame = draw_pitch(config=config)

                # Player dots on top of Voronoi
                if len(team_0_cm) > 0:
                    spatial_frame = draw_points_on_pitch(
                        config=config, xy=team_0_cm,
                        face_color=sv.Color.from_hex(COLORS[0]), radius=16, pitch=spatial_frame)
                if len(team_1_cm) > 0:
                    spatial_frame = draw_points_on_pitch(
                        config=config, xy=team_1_cm,
                        face_color=sv.Color.from_hex(COLORS[1]), radius=16, pitch=spatial_frame)

                # Interpolate across short None gaps (ball airborne → not detected)
                interp_path = interpolate_ball_path(ball_path_cm, max_gap=30)

                # Current position dot: prefer live detection, fallback to interpolated
                display_ball_cm = ball_cm_pos
                if display_ball_cm is None and interp_path and interp_path[-1] is not None:
                    display_ball_cm = interp_path[-1]

                # Ball trail: use interpolated path, strip remaining Nones, filter outliers
                trail_pts = [p for p in interp_path if p is not None]
                if len(trail_pts) >= 2:
                    clean_trail = replace_outlier_based_on_distance(trail_pts, max_distance=500.0)
                    if len(clean_trail) >= 2:
                        spatial_frame = draw_paths_on_pitch(
                            config=config,
                            paths=[np.array(clean_trail)],
                            color=sv.Color.from_hex('#FFFFFF'),
                            thickness=2,
                            pitch=spatial_frame,
                        )

                # Ball current position dot
                if display_ball_cm is not None:
                    spatial_frame = draw_points_on_pitch(
                        config=config, xy=display_ball_cm[np.newaxis],
                        face_color=sv.Color.from_hex('#FFFFFF'), radius=12, pitch=spatial_frame)

                writer.write(spatial_frame)
                frame_index += 1
                pbar.update(1)
    finally:
        cap.release()
        writer.release()

    return output_path


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
