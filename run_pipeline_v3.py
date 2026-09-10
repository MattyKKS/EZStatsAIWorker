"""One identity/team/event stream, shared by the report and stats video.

Run one clip per process. Existing weights are reused; no training is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def finish_run(run_dir: Path, *, render: bool = True) -> dict:
    from ez_worker.analytics.events import detect_events
    from ez_worker.analytics.stats import build_track_stats
    from ez_worker.config import DEFAULT_CONFIG
    from ez_worker.outputs.writer import write_artifacts
    from ez_worker.postprocess.canonical import prepare_canonical_tracks
    from ez_worker.schemas import AnalysisArtifacts, VideoMeta
    from ez_worker.spatial.event_projection import project_tracks

    video = VideoMeta(**json.loads((run_dir / "video_meta.json").read_text(encoding="utf-8")))
    tracks, decisions = prepare_canonical_tracks(run_dir)
    ball_pitch, player_pitch = project_tracks(run_dir, tracks, video)
    cfg = DEFAULT_CONFIG
    possession_frames = {}
    print("Computing events from final identities and teams...", flush=True)
    events = detect_events(
        tracks, video, cfg.ball_track_id,
        ball_pitch_pos=ball_pitch, player_pitch_pos=player_pitch,
        possession_distance_threshold_px=cfg.possession_distance_threshold_px,
        pass_min_flight_frames=max(1, round(cfg.pass_min_flight_frames * video.fps / 25)),
        owner_min_possession_frames=max(1, round(cfg.owner_min_possession_frames * video.fps / 25)),
        possession_by_frame=possession_frames,
        allow_aerial_contacts=False,
    )
    stats = build_track_stats(tracks, events, video)
    artifacts = AnalysisArtifacts(video=video, tracks=tracks, events=events, stats=stats)
    write_artifacts(artifacts, run_dir)
    held = Counter(t for t in possession_frames.values() if t in (0, 1))
    report_path = run_dir / "match_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["possession"] = {str(t): round(100 * held[t] / sum(held.values()), 1)
                            for t in (0, 1)} if held else {}
    report["possession_known_frames"] = sum(held.values())
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / "possession_by_frame.json").write_text(json.dumps(possession_frames), encoding="utf-8")
    shutil.copy2(run_dir / "tracks.json", run_dir / "tracks_with_teams.json")
    # Compatibility filename only: no second report-only identity/event rewrite.
    shutil.copy2(run_dir / "match_report.json", run_dir / "match_report_merged.json")
    refs = sorted(int(t) for t, role in decisions["roles"].items() if role == "referee")
    (run_dir / "referee_track_ids.json").write_text(json.dumps(refs), encoding="utf-8")
    players = {o.track_id for o in tracks if o.label == "player"}
    quality = {
        "video": str(video.path), "fps": video.fps,
        "player_tracks": len(players), "referee_tracks": len(refs),
        "players_per_team": dict(Counter(str(decisions["teams"].get(t)) for t in players)),
        "events": dict(Counter(e.event_type for e in events)),
        "event_coordinates": "pitch_cm" if ball_pitch is not None else "normalized_pixels",
        "ball_observations": sum(o.label == "ball" and not o.is_interpolated for o in tracks),
        "ball_interpolations": sum(o.label == "ball" and o.is_interpolated for o in tracks),
        "accuracy": None,
        "accuracy_note": "Requires manually labelled identities and events; counts are not accuracy.",
    }
    (run_dir / "quality_summary.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(json.dumps(quality, indent=2), flush=True)
    if render:
        from ez_worker.io.render_from_tracks import render_from_tracks
        render_from_tracks(run_dir)
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/raw/leo_messi_30pass.mp4", type=Path)
    parser.add_argument("--tracker-config", default="configs/botsort_football_v3.yaml", type=Path)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-pitch", action="store_true")
    parser.add_argument("--resume", type=Path, help="Recompute final outputs from preserved raw tracks and crops")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.resume:
        run_dir = args.resume.resolve()
        if not (run_dir / "tracks_raw.json").exists():
            shutil.copy2(run_dir / "tracks.json", run_dir / "tracks_raw.json")
    else:
        import yaml
        from ez_worker.config import PipelineConfig
        from ez_worker.io.video import load_video_meta
        from ez_worker.pipeline import run_analysis

        video = load_video_meta(args.video)
        player = ROOT / "artifacts/training/player_detector_v2/weights/best.pt"
        ball = ROOT / "artifacts/ball/football-ball-detection.pt"
        for path in (player, ball, args.tracker_config):
            if not path.is_file():
                raise FileNotFoundError(path)
        tracker = yaml.safe_load(args.tracker_config.read_text(encoding="utf-8"))
        tracker["track_buffer"] = max(1, round(3.6 * video.fps))
        from datetime import datetime
        setup_dir = ROOT / "outputs" / ("config_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        setup_dir.mkdir(parents=True)
        tracker_path = setup_dir / "tracker.yaml"
        tracker_path.write_text(yaml.safe_dump(tracker), encoding="utf-8")
        cfg = PipelineConfig(
            provider="ultralytics", model_name=str(player), ball_model_name=str(ball),
            tracker_config=str(tracker_path), frame_step=1, render_video=False,
            detection_confidence=0.20, min_player_confidence=0.18, min_ball_confidence=0.15,
            detection_iou=0.45, min_track_length=2, max_players_per_frame=28,
            max_unique_players=10000, min_player_track_frames=max(2, round(video.fps * 0.2)),
            merge_tracklets=False, interpolate_gaps=False,
            max_ball_jump_px=90, ball_reset_gap_frames=round(video.fps * 0.8),
            ball_reset_confidence=0.55, ball_hold_max_gap_frames=max(1, round(video.fps * 0.12)),
            ball_smoothing_alpha=0.35, auto_calibrate=False, export_player_crops=True,
        )
        print(f"Pipeline v3: {args.video}, actual fps={video.fps:.3f}", flush=True)
        run_dir = run_analysis(args.video, cfg).resolve()
        shutil.copy2(run_dir / "tracks.json", run_dir / "tracks_raw.json")
        shutil.copy2(tracker_path, run_dir / "tracker.yaml")
        manifest = {"pipeline": "v3", "config": cfg.model_dump(mode="json"),
                    "tracker": tracker, "weights_sha256": {}, "packages": {}}
        for path in (player, ball):
            with path.open("rb") as stream:
                manifest["weights_sha256"][path.name + ":" + path.parent.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        for package in ("ultralytics", "torch", "numpy", "supervision", "scikit-learn"):
            manifest["packages"][package] = importlib.metadata.version(package)
        manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        manifest["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Run dir: {run_dir}", flush=True)
    if not args.skip_pitch and not (run_dir / "pitch_keypoints_per_frame.json").exists():
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        result = subprocess.run([sys.executable, "-u", "-m", "ez_worker.cli",
            "detect-pitch-keypoints", "--run-dir", str(run_dir),
            "--model-path", "artifacts/pitch/football-pitch-detectionV2.pt",
            "--per-frame-stride", "5"], env=env)
        if result.returncode:
            print("Pitch step failed; events will use pixel coordinates.", flush=True)
    finish_run(run_dir, render=not args.skip_video)
    print(f"PIPELINE v3 complete. Run dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
