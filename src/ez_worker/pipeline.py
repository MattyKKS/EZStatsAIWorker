from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ez_worker.analytics.events import detect_events
from ez_worker.analytics.event_spotter import fuse_events, run_event_spotter
from ez_worker.analytics.stats import build_track_stats
from ez_worker.config import PipelineConfig
from ez_worker.io.crops import export_player_crops
from ez_worker.io.render import render_tracks_video
from ez_worker.io.video import load_video_meta
from ez_worker.outputs.writer import write_artifacts
from ez_worker.postprocess.tracks import cleanup_tracks
from ez_worker.spatial.line_detector import detect_penalty_marks
from ez_worker.providers.base import TrackingProvider
from ez_worker.providers.mock import MockTrackingProvider
from ez_worker.providers.ultralytics_provider import UltralyticsTrackingProvider
from ez_worker.schemas import AnalysisArtifacts


def run_analysis(video_path: Path, config: PipelineConfig) -> Path:
    video = load_video_meta(video_path)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_root / run_id
    provider = _resolve_provider(config.provider)

    penalty_marks: list[tuple[float, float]] = []
    if config.auto_calibrate:
        penalty_marks = detect_penalty_marks(video_path)
        if penalty_marks:
            print(f"Auto-calibrate: detected {len(penalty_marks)} penalty mark(s): {penalty_marks}")
        else:
            print("Auto-calibrate: no penalty marks detected in sample frame.")

    provider_artifacts = provider.run(video=video, config=config, output_dir=output_dir)
    tracks = cleanup_tracks(
        provider_artifacts.tracks,
        video,
        max_players_per_frame=config.max_players_per_frame,
        max_unique_players=config.max_unique_players,
        min_player_track_frames=config.min_player_track_frames,
        merge_tracklets=config.merge_tracklets,
        max_track_gap_frames=config.max_track_gap_frames,
        max_track_merge_distance_px=config.max_track_merge_distance_px,
        interpolate_gaps=config.interpolate_gaps,
        frame_step=config.frame_step,
        max_interpolation_gap_frames=config.max_interpolation_gap_frames,
        clean_ball_path=config.clean_ball_path,
        max_ball_jump_px=config.max_ball_jump_px,
        ball_reset_gap_frames=config.ball_reset_gap_frames,
        ball_reset_confidence=config.ball_reset_confidence,
        ball_hold_max_gap_frames=config.ball_hold_max_gap_frames,
        ball_smoothing_alpha=config.ball_smoothing_alpha,
        drop_ambiguous_ball_frames=config.drop_ambiguous_ball_frames,
        penalty_marks=penalty_marks if penalty_marks else None,
    )
    events = detect_events(
        tracks=tracks,
        video=video,
        ball_track_id=config.ball_track_id,
        possession_distance_threshold_cm=config.possession_distance_threshold_cm,
        possession_distance_threshold_px=config.possession_distance_threshold_px,
        possession_min_seconds=config.possession_min_seconds,
        pass_min_speed_cms=config.pass_min_speed_cms,
        pass_min_speed_px_per_s=config.pass_min_speed_px_per_s,
        shot_min_speed_cms=config.shot_min_speed_cms,
        shot_min_speed_px_per_s=config.shot_min_speed_px_per_s,
        shot_no_catch_seconds=config.shot_no_catch_seconds,
        ball_direction_change_min_deg=config.ball_direction_change_min_deg,
        enable_direct_possession_path=config.enable_direct_possession_path,
        clearance_min_flight_seconds=config.clearance_min_flight_seconds,
        clearance_min_arc_frac=config.clearance_min_arc_frac,
        pass_min_flight_frames=config.pass_min_flight_frames,
        owner_min_possession_frames=config.owner_min_possession_frames,
        pass_min_ball_travel_px=config.pass_min_ball_travel_px,
        reception_decel_fraction=config.reception_decel_fraction,
        touch_min_ball_speed_px_per_s=config.touch_min_ball_speed_px_per_s,
    )
    if config.event_model_name:
        print(f"Running event spotter: {config.event_model_name}")
        spotter_events = run_event_spotter(
            video_path=video_path,
            video=video,
            model_name=config.event_model_name,
        )
        print(f"Event spotter found {len(spotter_events)} events.")
        events = fuse_events(events, spotter_events, video_fps=video.fps)
    stats = build_track_stats(tracks=tracks, events=events, video=video)

    processed_video_path = provider_artifacts.processed_video_path
    if config.render_video:
        processed_video_path = output_dir / "processed_video.mp4"
        render_tracks_video(
            video=video,
            tracks=tracks,
            events=events,
            output_path=processed_video_path,
        )

    artifacts = AnalysisArtifacts(
        video=video,
        tracks=tracks,
        events=events,
        stats=stats,
        processed_video_path=processed_video_path,
    )
    write_artifacts(artifacts, output_dir)
    if config.export_player_crops:
        export_player_crops(
            video,
            tracks,
            output_dir,
            max_crops_per_track=config.max_crops_per_track,
        )
    return output_dir


def _resolve_provider(name: str) -> TrackingProvider:
    if name == "mock":
        return MockTrackingProvider()
    if name == "ultralytics":
        return UltralyticsTrackingProvider()
    raise ValueError(f"Unknown provider '{name}'. Available providers: 'mock', 'ultralytics'.")
