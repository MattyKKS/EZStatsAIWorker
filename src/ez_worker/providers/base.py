from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ez_worker.config import PipelineConfig
from ez_worker.schemas import TrackObservation, VideoMeta


@dataclass
class ProviderArtifacts:
    tracks: list[TrackObservation]
    processed_video_path: Path | None = None


class TrackingProvider(ABC):
    @abstractmethod
    def run(
        self,
        video: VideoMeta,
        config: PipelineConfig,
        output_dir: Path,
    ) -> ProviderArtifacts:
        raise NotImplementedError
