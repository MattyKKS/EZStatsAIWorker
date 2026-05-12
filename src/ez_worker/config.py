from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    provider: str = Field(default="mock")
    frame_step: int = Field(default=5, ge=1)
    render_video: bool = Field(default=False)
    output_root: Path = Field(default=Path("outputs"))
    model_name: str = Field(default="yolov8n.pt")
    ball_model_name: Optional[str] = Field(
        default=None,
        description=(
            "Optional separate model used exclusively for the dedicated ball pass. "
            "When set, the main model (model_name) handles player/goalkeeper/referee tracking "
            "and this model handles ball detection at ball_detection_imgsz resolution. "
            "Train a ball-only YOLO model on detector_ball_only dataset and point here."
        ),
    )
    tracker_config: str = Field(default="configs/bytetrack_football.yaml")
    detection_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    detection_iou: float = Field(default=0.45, ge=0.0, le=1.0)
    min_player_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    min_ball_confidence: float = Field(default=0.15, ge=0.0, le=1.0)
    detection_imgsz: int = Field(default=1280, ge=320)
    dedicated_ball_pass: bool = Field(default=True)
    ball_detection_imgsz: int = Field(default=1280, ge=320)
    min_track_length: int = Field(default=4, ge=1)
    max_player_box_area_fraction: float = Field(default=0.12, gt=0.0, lt=1.0)
    max_players_per_frame: int = Field(default=30, ge=1)
    max_unique_players: int = Field(default=30, ge=1)
    min_player_track_frames: int = Field(default=12, ge=1)
    merge_tracklets: bool = Field(default=True)
    max_track_gap_frames: int = Field(default=15, ge=0)
    max_track_merge_distance_px: float = Field(default=80.0, gt=0)
    interpolate_gaps: bool = Field(default=True)
    max_interpolation_gap_frames: int = Field(default=12, ge=0)
    clean_ball_path: bool = Field(default=True)
    max_ball_jump_px: float = Field(default=120.0, gt=0)
    ball_reset_gap_frames: int = Field(default=20, ge=0)
    ball_reset_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    ball_hold_max_gap_frames: int = Field(default=6, ge=0)
    ball_smoothing_alpha: float = Field(default=0.35, ge=0.0, le=1.0)
    drop_ambiguous_ball_frames: bool = Field(default=True)
    export_player_crops: bool = Field(default=False)
    max_crops_per_track: int = Field(default=20, ge=1)
    ball_track_id: int = Field(default=0)
    # ── Possession ──────────────────────────────────────────────────────────
    possession_distance_threshold_cm: float = Field(default=200.0, gt=0,
        description="Ball ownership radius in cm (used when pitch coords available)")
    possession_distance_threshold_px: float = Field(default=50.0, gt=0,
        description="Pixel-space fallback when homography is unavailable")
    possession_min_seconds: float = Field(default=0.15, gt=0,
        description="Seconds ball must stay near a player to confirm ownership")
    # ── Pass ────────────────────────────────────────────────────────────────
    pass_min_speed_cms: float = Field(default=250.0, gt=0,
        description="Minimum ball speed in cm/s to enter IN_FLIGHT (real pass vs slow drift)")
    pass_min_speed_px_per_s: float = Field(default=200.0, gt=0,
        description="Pixel fallback for pass_min_speed_cms")
    ball_direction_change_min_deg: float = Field(default=35.0, gt=0, lt=180.0,
        description="Direction change required for slow direct-possession pass detection")
    # ── Shot ────────────────────────────────────────────────────────────────
    shot_min_speed_cms: float = Field(default=1200.0, gt=0,
        description="Minimum ball launch speed in cm/s to classify as shot (12 m/s)")
    shot_min_speed_px_per_s: float = Field(default=600.0, gt=0,
        description="Pixel fallback for shot_min_speed_cms")
    shot_no_catch_seconds: float = Field(default=1.0, gt=0,
        description="Seconds with no reception after launch to confirm shot")
    # ── Clearance / high ball ────────────────────────────────────────────────
    clearance_min_flight_seconds: float = Field(default=1.2, gt=0,
        description="Flight longer than this with arc = clearance/long_ball, not shot")
    clearance_min_arc_frac: float = Field(default=0.04, gt=0,
        description="Ball y-range > this fraction of frame height to count as high arc")
    auto_calibrate: bool = Field(default=False)
    event_model_name: Optional[str] = Field(default=None)


DEFAULT_CONFIG = PipelineConfig()
