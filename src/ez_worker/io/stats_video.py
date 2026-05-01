from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ez_worker.io.render import (
    BALL_COLOR,
    GOALKEEPER_COLOR,
    PLAYER_COLOR,
    REFEREE_COLOR,
    _build_sticky_source_labels,
    _draw_track,
)
from ez_worker.schemas import BBox, Event, TrackObservation, VideoMeta


TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    0: (220, 80, 60),    # Team A — blue-ish
    1: (60, 180, 220),   # Team B — red-ish
    2: (160, 160, 160),  # Outlier / unknown
}
TEAM_LABELS: dict[int, str] = {0: "TEAM A", 1: "TEAM B", 2: "OTHER"}

POSSESSION_WINDOW_FRAMES = 75  # ~3 sec look-back at 25fps


def _load_team_classifier(run_dir: Path):
    """Load saved UMAP+KMeans+SigLIP for per-frame prediction."""
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
        print(f"  Warning: could not load team classifier for per-frame prediction: {e}")
        return None


def _predict_frame_teams(clf, frame: np.ndarray, frame_tracks: list, video) -> dict[int, int]:
    """Predict team per player crop in this frame — tutorial's per-frame approach."""
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

    if clf["reducer"] is not None:
        projections = clf["reducer"].transform(embeddings)
    else:
        projections = embeddings

    labels = clf["km"].predict(projections)
    return {tid: int(label) for tid, label in zip(tids, labels)}


def render_stats_video(run_dir: Path) -> Path:
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

    # Static fallback team assignments from clustering
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

    # Load per-frame classifier (tutorial approach — eliminates overlap team switches)
    clf = _load_team_classifier(run_dir)
    if clf:
        print("  Per-frame team prediction enabled (tutorial approach)")
    else:
        print("  Using static team assignments (run cluster-teams to enable per-frame)")

    # Load pitch homography for minimap
    transformer = None
    keypoints_path = run_dir / "pitch_keypoints.json"
    if keypoints_path.exists():
        try:
            from ez_worker.spatial.view_transformer import ViewTransformer
            from ez_worker.spatial.keypoint_detector import KEYPOINT_PITCH_XY_M
            kp_data = json.loads(keypoints_path.read_text(encoding="utf-8"))
            kp_px = kp_data.get("keypoints", {})
            kp_m = kp_data.get("keypoint_pitch_xy_m", {})
            common = [k for k in kp_px if k in kp_m]
            if len(common) >= 4:
                src = np.array([kp_px[k] for k in common], dtype=np.float32)
                dst = np.array([kp_m[k] for k in common], dtype=np.float32)
                transformer = ViewTransformer(src, dst)
                print(f"  Minimap enabled ({len(common)} keypoints)")
        except Exception as e:
            print(f"  Minimap skipped: {e}")

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

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_tracks = tracks_by_frame.get(frame_index, [])

            # Per-frame prediction overrides static assignment — tutorial approach
            if clf and frame_tracks:
                live_teams = _predict_frame_teams(clf, frame, frame_tracks, video)
                frame_team_lookup = {**team_id_by_track, **live_teams}
            else:
                frame_team_lookup = team_id_by_track

            for track in frame_tracks:
                _draw_track_with_team(frame, track, video, sticky_source, frame_team_lookup)
            for event in events_by_frame.get(frame_index, []):
                _draw_event_label(frame, event)
            poss = possession_by_frame.get(frame_index, {})
            _draw_stats_panel(frame, team_stats, poss, has_teams)

            # Minimap overlay
            if transformer is not None:
                try:
                    from ez_worker.spatial.minimap import draw_minimap, overlay_minimap
                    positions_m, t_ids = [], []
                    for track in frame_tracks:
                        if track.label != "player":
                            continue
                        bx = (track.bbox.x1 + track.bbox.x2) / 2 * video.width
                        by = track.bbox.y2 * video.height
                        pt_m = transformer.transform_points(np.array([[bx, by]]))[0]
                        positions_m.append((float(pt_m[0]), float(pt_m[1])))
                        t_ids.append(frame_team_lookup.get(track.track_id))
                    if positions_m:
                        minimap = draw_minimap(positions_m, t_ids)
                        frame = overlay_minimap(frame, minimap)
                except Exception:
                    pass

            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    return output_path


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

    # Semi-transparent dark background
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
    y = y0 + 22

    def text(x: int, row: int, s: str, color: tuple, scale: float = 0.42) -> None:
        cv2.putText(frame, s, (x, y0 + row * row_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    # Headers
    a = team_stats.get(0, {})
    b = team_stats.get(1, {})
    color_a = TEAM_COLORS[0]
    color_b = TEAM_COLORS[1]

    text(col_a, 0, f"TEAM A ({a.get('players', 0)}p)", color_a, 0.44)
    text(col_b, 0, f"TEAM B ({b.get('players', 0)}p)", color_b, 0.44)

    # Divider
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


def _draw_track_with_team(
    frame: np.ndarray,
    track: TrackObservation,
    video: VideoMeta,
    sticky_source: dict[int, str],
    team_id_by_track: dict[int, int | None],
) -> None:
    if track.label == "ball":
        _draw_track(frame, track, video, sticky_source)
        return

    sticky = sticky_source.get(int(track.track_id), track.source_label or track.label or "player")
    if sticky == "goalkeeper":
        color = GOALKEEPER_COLOR
    elif sticky == "referee":
        color = REFEREE_COLOR
    else:
        team_id = team_id_by_track.get(track.track_id)
        color = TEAM_COLORS.get(team_id if team_id is not None else 2, PLAYER_COLOR)

    x1 = int(track.bbox.x1 * video.width)
    y1 = int(track.bbox.y1 * video.height)
    x2 = int(track.bbox.x2 * video.width)
    y2 = int(track.bbox.y2 * video.height)
    lw = 3 if sticky == "goalkeeper" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, lw)

    team_id = team_id_by_track.get(track.track_id)
    team_tag = {0: "A", 1: "B", None: ""}.get(team_id, "?")
    role_tag = {"goalkeeper": "GK", "referee": "REF"}.get(sticky, "")
    label = f"{track.track_id}"
    if role_tag:
        label += f" {role_tag}"
    elif team_tag:
        label += f" {team_tag}"
    cv2.putText(frame, label, (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_event_label(frame: np.ndarray, event: Event) -> None:
    label = event.event_type.replace("_", " ")
    cv2.putText(frame, label, (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


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

    # Convert px distance to rough km (assume ~0.1m per px at broadcast scale)
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
        if e.event_type in {"ball_touch", "pass", "shot"}
        and e.actor_track_id is not None
    ]

    result: dict[int, dict[int, float]] = {}
    for frame_idx in range(0, total_frames):
        window_start = max(0, frame_idx - POSSESSION_WINDOW_FRAMES)
        window_touches = [
            e for e in touch_events
            if window_start <= e.frame_index <= frame_idx
        ]
        counts: dict[int, int] = defaultdict(int)
        for e in window_touches:
            tid = e.actor_track_id
            team_id = team_id_by_track.get(tid)
            if team_id in (0, 1):
                counts[team_id] += 1
        total = sum(counts.values())
        if total == 0:
            result[frame_idx] = {0: 0.5, 1: 0.5}
        else:
            result[frame_idx] = {k: counts[k] / total for k in (0, 1)}
    return result


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
