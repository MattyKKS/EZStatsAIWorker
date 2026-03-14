from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class VideoMeta(BaseModel):
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


class TrackObservation(BaseModel):
    frame_index: int
    track_id: int
    label: str
    confidence: float = Field(default=1.0)
    bbox: BBox
    team_id: int | None = None


class Event(BaseModel):
    frame_index: int
    time_seconds: float
    event_type: str
    actor_track_id: int | None = None
    target_track_id: int | None = None
    details: dict = Field(default_factory=dict)


class TrackStats(BaseModel):
    track_id: int
    label: str
    frame_count: int
    approx_distance_px: float
    avg_speed_px_per_frame: float
    touch_count: int = 0
    pass_count: int = 0
    shot_count: int = 0


class AnalysisArtifacts(BaseModel):
    video: VideoMeta
    tracks: list[TrackObservation]
    events: list[Event]
    stats: list[TrackStats]
    processed_video_path: Path | None = None
