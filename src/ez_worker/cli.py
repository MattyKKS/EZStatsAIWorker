from __future__ import annotations

import argparse
from pathlib import Path

from ez_worker.appearance.prep import prepare_appearance_inputs
from ez_worker.appearance.team_assignment import cluster_teams
from ez_worker.appearance.apply_clusters import apply_team_clusters
from ez_worker.appearance.team_report import build_team_report
from ez_worker.appearance.team_report_analysis_ready import build_analysis_ready_team_report
from ez_worker.appearance.role_hints import build_role_hints
from ez_worker.appearance.role_filter import build_role_filtered_outputs
from ez_worker.analytics.event_refine import build_analysis_ready_events
from ez_worker.analytics.possession_report import build_possession_report
from ez_worker.outputs.integration_bundle import build_integration_bundle
from ez_worker.outputs.output_contract import build_output_contract
from ez_worker.outputs.ui_payload import build_ui_payload
from ez_worker.outputs.api_response import build_api_response
from ez_worker.outputs.mock_api import build_mock_api
from ez_worker.outputs.route_manifest import build_route_manifest
from ez_worker.config import DEFAULT_CONFIG, PipelineConfig
from ez_worker.data_init import init_data_layout
from ez_worker.diagnostics.run_debug import build_run_debug_report
from ez_worker.dataset_utils import prepare_roboflow_dataset_yaml
from ez_worker.pipeline import run_analysis
from ez_worker.spatial.pitch_prep import prepare_pitch_mapping
from ez_worker.spatial.homography import apply_pitch_mapping
from ez_worker.spatial.report import build_spatial_report
from ez_worker.spatial.heatmap_export import build_heatmap_export
from ez_worker.spatial.formation_export import build_formation_export
from ez_worker.io.stats_video import render_stats_video
from ez_worker.train import train_detector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ez-worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Run the AI worker on a video.")
    analyze.add_argument("--video", type=Path, required=True, help="Path to the input video file.")
    analyze.add_argument("--provider", default=DEFAULT_CONFIG.provider, help="Tracking provider name.")
    analyze.add_argument("--frame-step", type=int, default=DEFAULT_CONFIG.frame_step)
    analyze.add_argument("--render-video", action="store_true")
    analyze.add_argument("--export-player-crops", action="store_true")
    analyze.add_argument("--output-root", type=Path, default=DEFAULT_CONFIG.output_root)
    analyze.add_argument("--model-name", default=DEFAULT_CONFIG.model_name)
    analyze.add_argument("--ball-model-name", default=DEFAULT_CONFIG.ball_model_name)
    analyze.add_argument("--tracker-config", default=DEFAULT_CONFIG.tracker_config)
    analyze.add_argument("--detection-confidence", type=float, default=DEFAULT_CONFIG.detection_confidence)
    analyze.add_argument("--detection-iou", type=float, default=DEFAULT_CONFIG.detection_iou)
    analyze.add_argument("--min-player-confidence", type=float, default=DEFAULT_CONFIG.min_player_confidence)
    analyze.add_argument("--min-ball-confidence", type=float, default=DEFAULT_CONFIG.min_ball_confidence)
    analyze.add_argument("--detection-imgsz", type=int, default=DEFAULT_CONFIG.detection_imgsz)
    analyze.add_argument("--ball-detection-imgsz", type=int, default=DEFAULT_CONFIG.ball_detection_imgsz)
    analyze.add_argument("--min-track-length", type=int, default=DEFAULT_CONFIG.min_track_length)
    analyze.add_argument(
        "--max-player-box-area-fraction",
        type=float,
        default=DEFAULT_CONFIG.max_player_box_area_fraction,
    )
    analyze.add_argument("--max-players-per-frame", type=int, default=DEFAULT_CONFIG.max_players_per_frame)
    analyze.add_argument("--max-unique-players", type=int, default=DEFAULT_CONFIG.max_unique_players)
    analyze.add_argument("--min-player-track-frames", type=int, default=DEFAULT_CONFIG.min_player_track_frames)
    analyze.add_argument("--max-track-gap-frames", type=int, default=DEFAULT_CONFIG.max_track_gap_frames)
    analyze.add_argument(
        "--max-track-merge-distance-px",
        type=float,
        default=DEFAULT_CONFIG.max_track_merge_distance_px,
    )
    analyze.add_argument(
        "--max-interpolation-gap-frames",
        type=int,
        default=DEFAULT_CONFIG.max_interpolation_gap_frames,
    )
    analyze.add_argument("--max-ball-jump-px", type=float, default=DEFAULT_CONFIG.max_ball_jump_px)
    analyze.add_argument("--ball-reset-gap-frames", type=int, default=DEFAULT_CONFIG.ball_reset_gap_frames)
    analyze.add_argument("--ball-reset-confidence", type=float, default=DEFAULT_CONFIG.ball_reset_confidence)
    analyze.add_argument("--ball-hold-max-gap-frames", type=int, default=DEFAULT_CONFIG.ball_hold_max_gap_frames)
    analyze.add_argument("--ball-smoothing-alpha", type=float, default=DEFAULT_CONFIG.ball_smoothing_alpha)
    analyze.add_argument("--max-crops-per-track", type=int, default=DEFAULT_CONFIG.max_crops_per_track)
    analyze.add_argument(
        "--possession-distance-threshold-px",
        type=float,
        default=DEFAULT_CONFIG.possession_distance_threshold_px,
    )
    analyze.add_argument(
        "--possession-min-consecutive-frames",
        type=int,
        default=DEFAULT_CONFIG.possession_min_consecutive_frames,
    )
    analyze.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="Detect penalty marks via OpenCV and suppress false ball detections near them.",
    )
    analyze.add_argument(
        "--event-model-name",
        default=DEFAULT_CONFIG.event_model_name,
        help="Path to trained event spotter model (.pt). Falls back to rule-based events if not set.",
    )

    init_data = subparsers.add_parser("init-data", help="Create the next-stage data layout.")
    init_data.add_argument("--root", type=Path, default=Path("."))

    prepare_dataset = subparsers.add_parser(
        "prepare-roboflow",
        help="Create a local dataset YAML with resolved paths for training.",
    )
    prepare_dataset.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/datasets/roboflow/detector"),
    )

    prepare_appearance = subparsers.add_parser(
        "prepare-appearance",
        help="Build the next-stage appearance/team-clustering manifest for a completed run.",
    )
    prepare_appearance.add_argument("--run-dir", type=Path, required=True)

    cluster_appearance = subparsers.add_parser(
        "cluster-teams",
        help="Cluster player tracks into appearance groups from a prepared run folder.",
    )
    cluster_appearance.add_argument("--run-dir", type=Path, required=True)
    cluster_appearance.add_argument(
        "--method",
        default="color",
        choices=["color", "siglip", "siglip-jersey"],
    )
    cluster_appearance.add_argument("--cluster-count", type=int, default=3)
    cluster_appearance.add_argument("--model-name", default="google/siglip-base-patch16-224")

    apply_teams = subparsers.add_parser(
        "apply-team-clusters",
        help="Convert cluster output into provisional team assignments and enriched outputs.",
    )
    apply_teams.add_argument("--run-dir", type=Path, required=True)

    team_report = subparsers.add_parser(
        "build-team-report",
        help="Build a simple team-aware report from provisional team assignments.",
    )
    team_report.add_argument("--run-dir", type=Path, required=True)

    team_report_analysis_ready = subparsers.add_parser(
        "build-analysis-ready-team-report",
        help="Build a cleaner team-aware report from the analysis-ready event and player files.",
    )
    team_report_analysis_ready.add_argument("--run-dir", type=Path, required=True)

    role_hints = subparsers.add_parser(
        "build-role-hints",
        help="Build conservative referee and goalkeeper role hints from current team and spatial outputs.",
    )
    role_hints.add_argument("--run-dir", type=Path, required=True)

    role_filter = subparsers.add_parser(
        "build-analysis-ready",
        help="Build role-aware filtered outputs for later analytics and event work.",
    )
    role_filter.add_argument("--run-dir", type=Path, required=True)

    event_refine = subparsers.add_parser(
        "build-analysis-ready-events",
        help="Filter events through the analysis-ready player set and enrich them with team/role context.",
    )
    event_refine.add_argument("--run-dir", type=Path, required=True)

    possession_report = subparsers.add_parser(
        "build-possession-report",
        help="Build a simple possession and timeline report from the analysis-ready event file.",
    )
    possession_report.add_argument("--run-dir", type=Path, required=True)

    integration_bundle = subparsers.add_parser(
        "build-integration-bundle",
        help="Build one integration-ready bundle from the cleaned outputs for frontend/backend use.",
    )
    integration_bundle.add_argument("--run-dir", type=Path, required=True)

    output_contract = subparsers.add_parser(
        "build-output-contract",
        help="Build a simple output contract file that explains which cleaned outputs frontend/backend should use.",
    )
    output_contract.add_argument("--run-dir", type=Path, required=True)

    ui_payload = subparsers.add_parser(
        "build-ui-payload",
        help="Build one UI-friendly payload from the cleaned integration outputs.",
    )
    ui_payload.add_argument("--run-dir", type=Path, required=True)

    api_response = subparsers.add_parser(
        "build-api-response",
        help="Build one API-friendly response object from the UI payload and output contract.",
    )
    api_response.add_argument("--run-dir", type=Path, required=True)

    mock_api = subparsers.add_parser(
        "build-mock-api",
        help="Split the API-friendly response into mock endpoint files for frontend/backend integration.",
    )
    mock_api.add_argument("--run-dir", type=Path, required=True)

    route_manifest = subparsers.add_parser(
        "build-route-manifest",
        help="Build a simple route manifest that maps mock endpoint files to API paths.",
    )
    route_manifest.add_argument("--run-dir", type=Path, required=True)

    detect_kp = subparsers.add_parser(
        "detect-pitch-keypoints",
        help="Auto-detect pitch keypoints from video using YOLO keypoint model.",
    )
    detect_kp.add_argument("--run-dir", type=Path, required=True)
    detect_kp.add_argument(
        "--model-path",
        type=Path,
        default=None,
    )
    detect_kp.add_argument("--api-key", default=None, help="Roboflow API key for cloud inference.")

    pitch_prep = subparsers.add_parser(
        "prepare-pitch-mapping",
        help="Create a reference frame and manual template for later pitch homography work.",
    )
    pitch_prep.add_argument("--run-dir", type=Path, required=True)

    pitch_apply = subparsers.add_parser(
        "apply-pitch-mapping",
        help="Apply a filled pitch mapping template and project tracks into pitch coordinates.",
    )
    pitch_apply.add_argument("--run-dir", type=Path, required=True)

    spatial_report = subparsers.add_parser(
        "build-spatial-report",
        help="Build simple team and player spatial summaries from projected pitch tracks.",
    )
    spatial_report.add_argument("--run-dir", type=Path, required=True)

    heatmap_export = subparsers.add_parser(
        "build-heatmap-export",
        help="Build heatmap and formation-friendly exports from projected pitch tracks.",
    )
    heatmap_export.add_argument("--run-dir", type=Path, required=True)

    formation_export = subparsers.add_parser(
        "build-formation-export",
        help="Build a simple formation and tactical layout export from heatmap output.",
    )
    formation_export.add_argument("--run-dir", type=Path, required=True)

    run_debug = subparsers.add_parser(
        "build-run-debug-report",
        help="Build a compact diagnostics report for a completed run to debug detector and cleanup quality.",
    )
    run_debug.add_argument("--run-dir", type=Path, required=True)

    stats_video = subparsers.add_parser(
        "render-stats-video",
        help="Re-render the original video with a team stats overlay in the top-right corner.",
    )
    stats_video.add_argument("--run-dir", type=Path, required=True)

    train = subparsers.add_parser("train-detector", help="Train the football detector.")
    train.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/datasets/roboflow/detector"),
    )
    train.add_argument("--model-name", default="yolov8m.pt")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--image-size", type=int, default=1280)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--device", default="0")
    train.add_argument("--project-dir", type=Path, default=Path("artifacts/training"))
    train.add_argument("--run-name", default="roboflow_detector")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        config = PipelineConfig(
            provider=args.provider,
            frame_step=args.frame_step,
            render_video=args.render_video,
            export_player_crops=args.export_player_crops,
            output_root=args.output_root,
            model_name=args.model_name,
            ball_model_name=args.ball_model_name,
            tracker_config=args.tracker_config,
            detection_confidence=args.detection_confidence,
            detection_iou=args.detection_iou,
            min_player_confidence=args.min_player_confidence,
            min_ball_confidence=args.min_ball_confidence,
            detection_imgsz=args.detection_imgsz,
            ball_detection_imgsz=args.ball_detection_imgsz,
            min_track_length=args.min_track_length,
            max_player_box_area_fraction=args.max_player_box_area_fraction,
            max_players_per_frame=args.max_players_per_frame,
            max_unique_players=args.max_unique_players,
            min_player_track_frames=args.min_player_track_frames,
            max_track_gap_frames=args.max_track_gap_frames,
            max_track_merge_distance_px=args.max_track_merge_distance_px,
            max_interpolation_gap_frames=args.max_interpolation_gap_frames,
            max_ball_jump_px=args.max_ball_jump_px,
            ball_reset_gap_frames=args.ball_reset_gap_frames,
            ball_reset_confidence=args.ball_reset_confidence,
            ball_hold_max_gap_frames=args.ball_hold_max_gap_frames,
            ball_smoothing_alpha=args.ball_smoothing_alpha,
            max_crops_per_track=args.max_crops_per_track,
            possession_distance_threshold_px=args.possession_distance_threshold_px,
            possession_min_consecutive_frames=args.possession_min_consecutive_frames,
            auto_calibrate=args.auto_calibrate,
            event_model_name=args.event_model_name,
        )
        output_dir = run_analysis(video_path=args.video, config=config)
        print(f"Analysis complete. Outputs written to: {output_dir}")
        return

    if args.command == "init-data":
        created = init_data_layout(args.root)
        for path in created:
            print(path)
        return

    if args.command == "prepare-roboflow":
        prepared_yaml = prepare_roboflow_dataset_yaml(args.dataset_dir)
        print(prepared_yaml)
        return

    if args.command == "prepare-appearance":
        manifest_path = prepare_appearance_inputs(args.run_dir)
        print(manifest_path)
        return

    if args.command == "cluster-teams":
        output_path = cluster_teams(
            args.run_dir,
            method=args.method,
            cluster_count=args.cluster_count,
            model_name=args.model_name,
        )
        print(output_path)
        return

    if args.command == "apply-team-clusters":
        output_path = apply_team_clusters(args.run_dir)
        print(output_path)
        return

    if args.command == "build-team-report":
        output_path = build_team_report(args.run_dir)
        print(output_path)
        return

    if args.command == "build-analysis-ready-team-report":
        output_path = build_analysis_ready_team_report(args.run_dir)
        print(output_path)
        return

    if args.command == "build-role-hints":
        output_path = build_role_hints(args.run_dir)
        print(output_path)
        return

    if args.command == "build-analysis-ready":
        output_path = build_role_filtered_outputs(args.run_dir)
        print(output_path)
        return

    if args.command == "build-analysis-ready-events":
        output_path = build_analysis_ready_events(args.run_dir)
        print(output_path)
        return

    if args.command == "build-possession-report":
        output_path = build_possession_report(args.run_dir)
        print(output_path)
        return

    if args.command == "build-integration-bundle":
        output_path = build_integration_bundle(args.run_dir)
        print(output_path)
        return

    if args.command == "build-output-contract":
        output_path = build_output_contract(args.run_dir)
        print(output_path)
        return

    if args.command == "build-ui-payload":
        output_path = build_ui_payload(args.run_dir)
        print(output_path)
        return

    if args.command == "build-api-response":
        output_path = build_api_response(args.run_dir)
        print(output_path)
        return

    if args.command == "build-mock-api":
        output_path = build_mock_api(args.run_dir)
        print(output_path)
        return

    if args.command == "build-route-manifest":
        output_path = build_route_manifest(args.run_dir)
        print(output_path)
        return

    if args.command == "detect-pitch-keypoints":
        from ez_worker.spatial.keypoint_detector import detect_pitch_keypoints
        output_path = detect_pitch_keypoints(args.run_dir, args.model_path, api_key=getattr(args, "api_key", None))
        print(output_path)
        return

    if args.command == "prepare-pitch-mapping":
        output_path = prepare_pitch_mapping(args.run_dir)
        print(output_path)
        return

    if args.command == "apply-pitch-mapping":
        output_path = apply_pitch_mapping(args.run_dir)
        print(output_path)
        return

    if args.command == "build-spatial-report":
        output_path = build_spatial_report(args.run_dir)
        print(output_path)
        return

    if args.command == "build-heatmap-export":
        output_path = build_heatmap_export(args.run_dir)
        print(output_path)
        return

    if args.command == "build-formation-export":
        output_path = build_formation_export(args.run_dir)
        print(output_path)
        return

    if args.command == "build-run-debug-report":
        output_path = build_run_debug_report(args.run_dir)
        print(output_path)
        return

    if args.command == "render-stats-video":
        output_path = render_stats_video(args.run_dir)
        print(output_path)
        return

    if args.command == "train-detector":
        save_dir = train_detector(
            dataset_dir=args.dataset_dir,
            model_name=args.model_name,
            epochs=args.epochs,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=args.device,
            project_dir=args.project_dir,
            run_name=args.run_name,
        )
        print(save_dir)


if __name__ == "__main__":
    main()
