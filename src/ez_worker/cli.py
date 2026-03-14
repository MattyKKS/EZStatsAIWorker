from __future__ import annotations

import argparse
from pathlib import Path

from ez_worker.config import DEFAULT_CONFIG, PipelineConfig
from ez_worker.data_init import init_data_layout
from ez_worker.dataset_utils import prepare_roboflow_dataset_yaml
from ez_worker.pipeline import run_analysis
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
    analyze.add_argument("--tracker-config", default=DEFAULT_CONFIG.tracker_config)
    analyze.add_argument("--detection-confidence", type=float, default=DEFAULT_CONFIG.detection_confidence)
    analyze.add_argument("--detection-iou", type=float, default=DEFAULT_CONFIG.detection_iou)
    analyze.add_argument("--min-player-confidence", type=float, default=DEFAULT_CONFIG.min_player_confidence)
    analyze.add_argument("--min-ball-confidence", type=float, default=DEFAULT_CONFIG.min_ball_confidence)
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
            tracker_config=args.tracker_config,
            detection_confidence=args.detection_confidence,
            detection_iou=args.detection_iou,
            min_player_confidence=args.min_player_confidence,
            min_ball_confidence=args.min_ball_confidence,
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
            max_crops_per_track=args.max_crops_per_track,
            possession_distance_threshold_px=args.possession_distance_threshold_px,
            possession_min_consecutive_frames=args.possession_min_consecutive_frames,
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
