from __future__ import annotations

import math
from pathlib import Path

from ez_worker.config import PipelineConfig
from ez_worker.providers.base import ProviderArtifacts, TrackingProvider
from ez_worker.schemas import BBox, TrackObservation, VideoMeta


class MockTrackingProvider(TrackingProvider):
    """Synthetic provider so the pipeline can run before real CV models are wired in."""

    def run(
        self,
        video: VideoMeta,
        config: PipelineConfig,
        output_dir: Path,
    ) -> ProviderArtifacts:
        observations: list[TrackObservation] = []
        max_frames = min(video.frame_count, int(video.fps * 60 * 2))

        for frame_index in range(0, max_frames, config.frame_step):
            t = frame_index / max(video.fps, 1.0)
            observations.extend(
                [
                    self._player_obs(frame_index, 1, 0.22 + 0.08 * math.sin(t / 3), 0.55),
                    self._player_obs(frame_index, 2, 0.54 + 0.06 * math.sin(t / 4), 0.48),
                    self._player_obs(frame_index, 3, 0.76 + 0.04 * math.sin(t / 5), 0.40),
                    self._ball_obs(frame_index, 0.24 + 0.52 * min(t / 60.0, 1.0), 0.52),
                ]
            )
        return ProviderArtifacts(tracks=observations)

    def _player_obs(self, frame_index: int, track_id: int, x: float, y: float) -> TrackObservation:
        w = 0.035
        h = 0.14
        return TrackObservation(
            frame_index=frame_index,
            track_id=track_id,
            label="player",
            source_label="player",
            bbox=BBox(x1=x - w, y1=y - h, x2=x + w, y2=y + h),
        )

    def _ball_obs(self, frame_index: int, x: float, y: float) -> TrackObservation:
        r = 0.012
        return TrackObservation(
            frame_index=frame_index,
            track_id=0,
            label="ball",
            source_label="ball",
            bbox=BBox(x1=x - r, y1=y - r, x2=x + r, y2=y + r),
        )
