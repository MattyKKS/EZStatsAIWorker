from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from ez_worker.io.render import _build_sticky_source_labels
from ez_worker.schemas import BBox, Event, TrackObservation, VideoMeta


# Tutorial-exact colours: team A (pink), team B (blue), goalkeeper-fallback (orange), referee (gold)
TUTORIAL_COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

TEAM_COLOR_HEX = {0: '#FF1493', 1: '#00BFFF'}  # team A, team B
REFEREE_COLOR_IDX = 3    # maps to '#FFD700'

# Supervision annotators — created once, reused every frame (tutorial style)
_COLOR_PALETTE = sv.ColorPalette.from_hex(TUTORIAL_COLORS)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(color=_COLOR_PALETTE, thickness=2)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=_COLOR_PALETTE,
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

POSSESSION_WINDOW_FRAMES = 75  # ~3 sec look-back at 25fps

TEAM_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (220, 80, 60),
    1: (60, 180, 220),
    2: (160, 160, 160),
}


# ---------------------------------------------------------------------------
# Ball annotator using saved track data (used until dedicated model is ready)
# ---------------------------------------------------------------------------

class _SavedBallAnnotator:
    """
    Replicates BallAnnotator trail effect using ball positions from saved tracks.
    Keeps last 10 positions and draws circles growing in size toward current frame.
    """
    def __init__(self, radius: int = 6, buffer_size: int = 10) -> None:
        from collections import deque
        self._buf: deque = deque(maxlen=buffer_size)
        self.radius = radius
        try:
            self._palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        except Exception:
            self._palette = None

    def update(self, frame: np.ndarray, xy: np.ndarray | None) -> np.ndarray:
        """xy: (1, 2) pixel position or None if ball not seen this frame."""
        self._buf.append(xy)
        for i, pos in enumerate(self._buf):
            if pos is None:
                continue
            r = max(1, int(1 + i * (self.radius - 1) / max(len(self._buf) - 1, 1)))
            color = self._palette.by_idx(i).as_bgr() if self._palette else (0, 255, 255)
            cx, cy = int(pos[0]), int(pos[1])
            cv2.circle(frame, (cx, cy), r, color, 2)
        return frame

    def get_last_xy(self) -> np.ndarray | None:
        for pos in reversed(self._buf):
            if pos is not None:
                return pos
        return None


# ---------------------------------------------------------------------------
# Per-frame pitch homography (tutorial: rebuild every frame)
# ---------------------------------------------------------------------------

class _PitchHomographyTracker:
    """
    Runs the YOLO pitch-keypoint model every frame and rebuilds the transformer
    exactly as the tutorial does: fresh ViewTransformer each frame, no RANSAC,
    no smoothing buffer.
    """

    def __init__(self, model_path: Path, vertices: list) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics not installed") from exc
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(model_path))
        self.model.to(device)
        self.vertices = np.array(vertices, dtype=np.float32)
        self._last_transformer = None

    def update(self, frame: np.ndarray):
        """Tutorial-exact: detect keypoints, mask invisible ones, build fresh transformer."""
        from sports.common.view import ViewTransformer
        result = self.model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        if keypoints.xy is None or len(keypoints.xy) == 0:
            return self._last_transformer

        kp_xy = keypoints.xy[0]  # shape (32, 2)
        mask = (kp_xy[:, 0] > 1) & (kp_xy[:, 1] > 1)

        if mask.sum() < 4:
            return self._last_transformer

        try:
            transformer = ViewTransformer(
                source=kp_xy[mask].astype(np.float32),
                target=self.vertices[mask],
            )
            self._last_transformer = transformer
        except ValueError:
            pass  # homography failed, keep last good one

        return self._last_transformer


# ---------------------------------------------------------------------------
# Ball detection (tutorial: BallTracker + InferenceSlicer)
# ---------------------------------------------------------------------------

class _BallDetector:
    """
    Ball detector matching tutorial exactly:
    - sv.InferenceSlicer with slice_wh=(640, 640)
    - .with_nms(threshold=0.1)
    - BallTracker(buffer_size=20)
    - BallAnnotator(radius=6, buffer_size=10)
    """

    def __init__(self, model_path: Path) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics not installed") from exc
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = YOLO(str(model_path))
        self._model.to(device)
        from sports.common.ball import BallTracker, BallAnnotator
        self.tracker = BallTracker(buffer_size=20)
        self.annotator = BallAnnotator(radius=6, buffer_size=10)
        self._device = device

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
# Per-frame team classifier (SigLIP — loads from team_classifier.joblib)
# ---------------------------------------------------------------------------

def _load_team_classifier(run_dir: Path):
    classifier_path = run_dir / "team_classifier.joblib"
    if not classifier_path.exists():
        return None
    try:
        import joblib
        import torch
        from transformers import AutoProcessor, SiglipVisionModel
        data = joblib.load(classifier_path)
        model_name = data.get("model_name", "google/siglip-base-patch16-224")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Per-frame SigLIP using device: {device}")
        processor = AutoProcessor.from_pretrained(model_name)
        siglip = SiglipVisionModel.from_pretrained(model_name).to(device)
        siglip.eval()
        return {"processor": processor, "siglip": siglip, "reducer": data["reducer"],
                "km": data["km"], "focus": data.get("focus_upper_body", False),
                "torch": torch, "device": device}
    except Exception as e:
        print(f"  Warning: could not load team classifier: {e}")
        return None


def _predict_frame_teams(clf, frame: np.ndarray, frame_tracks: list, video) -> dict[int, int]:
    if not frame_tracks:
        return {}
    from PIL import Image as PILImage
    crops, tids = [], []
    for track in frame_tracks:
        if track.label != "player":
            continue
        x1 = max(0, int(track.bbox.x1 * video.width))
        y1 = max(0, int(track.bbox.y1 * video.height))
        x2 = min(video.width, int(track.bbox.x2 * video.width))
        y2 = min(video.height, int(track.bbox.y2 * video.height))
        if x2 <= x1 or y2 <= y1:
            continue
        crop_bgr = frame[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue
        img = PILImage.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        if clf["focus"]:
            w, h = img.size
            img = img.crop((int(w * 0.15), 0, int(w * 0.85), int(h * 0.55)))
        crops.append(img)
        tids.append(track.track_id)
    if not crops:
        return {}
    torch = clf["torch"]
    device = clf.get("device", "cpu")
    with torch.no_grad():
        inputs = clf["processor"](images=crops, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = clf["siglip"](**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    projections = clf["reducer"].transform(embeddings) if clf["reducer"] is not None else embeddings
    labels = clf["km"].predict(projections)
    return {tid: int(label) for tid, label in zip(tids, labels)}


# ---------------------------------------------------------------------------
# Frame entity builder + goalkeeper team resolution (tutorial approach)
# ---------------------------------------------------------------------------

def _resolve_gk_teams(
    player_entries: list,        # list of (track_id, x1, y1, x2, y2)
    players_team_id: np.ndarray,
    gk_entries: list,
    video,
) -> np.ndarray:
    """
    Assign each goalkeeper to nearest team centroid, matching tutorial's
    resolve_goalkeepers_team_id exactly.
    """
    if not gk_entries:
        return np.array([], dtype=int)
    if not player_entries or len(players_team_id) == 0:
        return np.zeros(len(gk_entries), dtype=int)

    def bottom_center(e):
        bx = (e[1] + e[3]) / 2
        by = e[4]
        return np.array([bx, by])

    players_xy = np.array([bottom_center(e) for e in player_entries])
    team_0_xy = players_xy[players_team_id == 0]
    team_1_xy = players_xy[players_team_id == 1]

    if len(team_0_xy) == 0 or len(team_1_xy) == 0:
        return np.zeros(len(gk_entries), dtype=int)

    c0 = team_0_xy.mean(axis=0)
    c1 = team_1_xy.mean(axis=0)

    result = []
    for e in gk_entries:
        pos = bottom_center(e)
        result.append(0 if np.linalg.norm(pos - c0) < np.linalg.norm(pos - c1) else 1)
    return np.array(result, dtype=int)


def _build_frame_detections(
    frame_tracks: list,
    sticky_source: dict,
    frame_team_lookup: dict,
    video,
) -> Tuple[sv.Detections, np.ndarray, list, list, list, list]:
    """
    Build supervision Detections + tutorial-style color_lookup for this frame.
    Returns: (all_dets, color_lookup, labels, player_entries, gk_entries, ref_entries)
    """
    player_entries, gk_entries, ref_entries = [], [], []

    for t in frame_tracks:
        if t.label == "ball":
            continue
        sticky = sticky_source.get(int(t.track_id), t.source_label or t.label or "player")
        x1 = t.bbox.x1 * video.width
        y1 = t.bbox.y1 * video.height
        x2 = t.bbox.x2 * video.width
        y2 = t.bbox.y2 * video.height
        entry = (t.track_id, x1, y1, x2, y2)
        if sticky == "referee":
            ref_entries.append(entry)
        elif sticky == "goalkeeper":
            gk_entries.append(entry)
        else:
            player_entries.append(entry)

    players_team_id = np.array(
        [frame_team_lookup.get(e[0], 0) for e in player_entries], dtype=int)
    gk_team_id = _resolve_gk_teams(player_entries, players_team_id, gk_entries, video)

    color_lookup = np.array(
        players_team_id.tolist() +
        gk_team_id.tolist() +
        [REFEREE_COLOR_IDX] * len(ref_entries),
        dtype=int,
    )

    all_entries = player_entries + gk_entries + ref_entries
    labels = [str(e[0]) for e in all_entries]

    if not all_entries:
        return sv.Detections.empty(), np.array([], dtype=int), [], [], [], []

    xyxy = np.array([[e[1], e[2], e[3], e[4]] for e in all_entries])
    tids = np.array([e[0] for e in all_entries])
    all_dets = sv.Detections(xyxy=xyxy, tracker_id=tids)
    return all_dets, color_lookup, labels, player_entries, gk_entries, ref_entries


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_stats_video(
    run_dir: Path,
    pitch_model_path: Optional[Path] = None,
    ball_model_path: Optional[Path] = None,
) -> Path:
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

    team_id_by_track = {t.track_id: t.team_id for t in tracks if t.team_id is not None}
    team_stats = _compute_team_stats(raw_stats, team_id_by_track)
    possession_by_frame = _compute_possession_by_frame(events, video.frame_count, team_id_by_track)

    tracks_by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for t in tracks:
        tracks_by_frame[t.frame_index].append(t)
    events_by_frame: dict[int, list[Event]] = defaultdict(list)
    for e in events:
        events_by_frame[e.frame_index].append(e)
    sticky_source = _build_sticky_source_labels(tracks)

    # Per-frame team classifier
    clf = _load_team_classifier(run_dir)
    if clf:
        print("  Per-frame team prediction enabled (SigLIP)")
    else:
        print("  Using static team assignments (run cluster-teams to enable per-frame)")

    # Per-frame pitch homography — tutorial: every frame, no stride
    pitch_tracker = None
    radar_config = None
    _candidate_pitch = [pitch_model_path] if pitch_model_path else []
    _candidate_pitch += [
        Path("artifacts/pitch/football-pitch-detection.pt"),
        Path("artifacts/pitch/best.pt"),
    ]
    for _mp in _candidate_pitch:
        if _mp is not None and Path(_mp).exists():
            try:
                from sports.configs.soccer import SoccerPitchConfiguration
                radar_config = SoccerPitchConfiguration()
                pitch_tracker = _PitchHomographyTracker(Path(_mp), radar_config.vertices)
                print(f"  Per-frame pitch homography enabled: {_mp}")
                break
            except Exception as e:
                print(f"  Pitch tracker init failed ({_mp}): {e}")

    # Static fallback from saved pitch_keypoints.json
    static_transformer = None
    if pitch_tracker is None:
        keypoints_path = run_dir / "pitch_keypoints.json"
        if keypoints_path.exists():
            try:
                from ez_worker.spatial.view_transformer import ViewTransformer
                from sports.configs.soccer import SoccerPitchConfiguration
                kp_data = json.loads(keypoints_path.read_text(encoding="utf-8"))
                kp_px = kp_data.get("keypoints", {})
                kp_cm = kp_data.get("keypoint_pitch_xy_cm", {})
                common = [k for k in kp_px if k in kp_cm]
                if len(common) >= 4:
                    src = np.array([kp_px[k] for k in common], dtype=np.float32)
                    dst = np.array([kp_cm[k] for k in common], dtype=np.float32)
                    static_transformer = ViewTransformer(src, dst)
                    radar_config = SoccerPitchConfiguration()
                    print(f"  Static minimap fallback ({len(common)} keypoints) — pass pitch model for per-frame")
            except Exception as e:
                print(f"  Minimap skipped: {e}")

    # Ball detector (tutorial: BallTracker + InferenceSlicer)
    ball_detector = None
    _candidate_ball = [ball_model_path] if ball_model_path else []
    _candidate_ball += [
        Path("artifacts/ball/football-ball-detection.pt"),
    ]
    for _bp in _candidate_ball:
        if _bp is not None and Path(_bp).exists():
            try:
                ball_detector = _BallDetector(Path(_bp))
                print(f"  Ball detector enabled: {_bp}")
                break
            except Exception as e:
                print(f"  Ball detector init failed ({_bp}): {e}")

    # Ball annotator using saved tracks (baseline until dedicated model ready)
    saved_ball_annotator = _SavedBallAnnotator(radius=6, buffer_size=10)

    # Sports annotators — imported once
    try:
        from sports.annotators.soccer import draw_pitch as _draw_pitch, draw_points_on_pitch as _draw_pts
        _sports_ok = True
    except ImportError:
        _sports_ok = False

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

    transformer = static_transformer  # will be overwritten each frame by pitch_tracker
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_tracks = tracks_by_frame.get(frame_index, [])

            # 1. Pitch keypoint detection on CLEAN frame — must be before any drawing
            #    (tutorial detects on raw frame first, then draws annotations)
            if pitch_tracker is not None:
                transformer = pitch_tracker.update(frame)

            # 2. Ball from saved tracks (baseline) OR dedicated detector if available
            ball_xy_px = None
            ball_detections = sv.Detections.empty()
            # Saved ball track position
            for t in frame_tracks:
                if t.label == "ball":
                    bx = (t.bbox.x1 + t.bbox.x2) / 2 * video.width
                    by = (t.bbox.y1 + t.bbox.y2) / 2 * video.height
                    ball_xy_px = np.array([bx, by])
                    break
            # Dedicated ball model (when available — overrides saved position)
            if ball_detector is not None:
                try:
                    ball_detections = ball_detector.detect(frame)
                    if len(ball_detections) > 0:
                        ball_xy_px = ball_detections.get_anchors_coordinates(
                            sv.Position.BOTTOM_CENTER)[0]
                except Exception:
                    ball_detections = sv.Detections.empty()

            # 3. Per-frame team prediction
            if clf and frame_tracks:
                live_teams = _predict_frame_teams(clf, frame, frame_tracks, video)
                frame_team_lookup = {**team_id_by_track, **live_teams}
            else:
                frame_team_lookup = team_id_by_track

            # 4. Build supervision detections + tutorial color_lookup
            all_dets, color_lookup, labels, player_entries, gk_entries, ref_entries = \
                _build_frame_detections(frame_tracks, sticky_source, frame_team_lookup, video)

            # 5. Annotate players/GKs/referees with supervision ellipses (tutorial style)
            if len(all_dets) > 0:
                frame = ELLIPSE_ANNOTATOR.annotate(frame, all_dets, custom_color_lookup=color_lookup)
                frame = ELLIPSE_LABEL_ANNOTATOR.annotate(frame, all_dets, labels, custom_color_lookup=color_lookup)

            # 6. Ball annotation with trail (saved tracks baseline)
            frame = saved_ball_annotator.update(frame, ball_xy_px)
            # If dedicated detector is active, also draw its result
            if ball_detector is not None and len(ball_detections) > 0:
                try:
                    frame = ball_detector.annotator.annotate(frame, ball_detections)
                except Exception:
                    pass

            # 7. Draw event labels
            for event in events_by_frame.get(frame_index, []):
                _draw_event_label(frame, event)

            # 8. Stats panel (our addition on top of tutorial)
            poss = possession_by_frame.get(frame_index, {})
            _draw_stats_panel(frame, team_stats, poss, has_teams)

            # 9. Radar (tutorial-exact: w//2 × h//2, bottom-center, 50% opacity)
            if transformer is not None and radar_config is not None and _sports_ok:
                try:
                    fh, fw = frame.shape[:2]

                    # Transform all person positions to pitch cm
                    if len(all_dets) > 0:
                        xy_px = all_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                        xy_cm = transformer.transform_points(xy_px)
                    else:
                        xy_cm = np.zeros((0, 2))

                    radar = _draw_pitch(config=radar_config)
                    for cls_idx, color_hex in enumerate(TUTORIAL_COLORS):
                        if len(xy_cm) > 0:
                            mask = color_lookup == cls_idx
                            if mask.any():
                                radar = _draw_pts(
                                    config=radar_config,
                                    xy=xy_cm[mask],
                                    face_color=sv.Color.from_hex(color_hex),
                                    radius=20,
                                    pitch=radar,
                                )

                    # Ball on radar (white dot)
                    if ball_xy_px is not None:
                        ball_cm = transformer.transform_points(
                            ball_xy_px[np.newaxis].astype(np.float32))
                        radar = _draw_pts(
                            config=radar_config,
                            xy=ball_cm,
                            face_color=sv.Color.from_hex('#FFFFFF'),
                            radius=15,
                            pitch=radar,
                        )

                    # Tutorial-exact placement: w//2 × h//2, bottom center, 50% opacity
                    radar_resized = sv.resize_image(radar, (fw // 2, fh // 2))
                    rh, rw = radar_resized.shape[:2]
                    rect = sv.Rect(
                        x=fw // 2 - rw // 2,
                        y=fh - rh,
                        width=rw,
                        height=rh,
                    )
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
    # BGR versions of tutorial colours
    color_a = (60, 20, 255)   # #FF1493 in BGR
    color_b = (255, 191, 0)   # #00BFFF in BGR

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
