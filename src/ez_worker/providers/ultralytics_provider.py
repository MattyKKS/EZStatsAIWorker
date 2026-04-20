from __future__ import annotations

from collections import Counter
from pathlib import Path
import warnings

import cv2

from ez_worker.config import PipelineConfig
from ez_worker.providers.base import ProviderArtifacts, TrackingProvider
from ez_worker.schemas import BBox, TrackObservation, VideoMeta


class UltralyticsTrackingProvider(TrackingProvider):
    """Football-oriented provider using one tracked stream with separate ball handling per frame."""

    def run(
        self,
        video: VideoMeta,
        config: PipelineConfig,
        output_dir: Path,
    ) -> ProviderArtifacts:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run 'pip install -e .[ml]' before using "
                "the ultralytics provider."
            ) from exc

        model = YOLO(config.model_name)
        model_info = self._build_model_info(model)
        if not model_info["has_goalkeeper_referee_classes"]:
            warnings.warn(
                (
                    f"Model '{config.model_name}' does not expose both 'goalkeeper' and 'referee' classes. "
                    "This run will treat most humans as generic players (tutorial-style GK/REF separation "
                    "requires football-trained weights like a Roboflow-trained best.pt)."
                ),
                stacklevel=2,
            )

        if config.ball_model_name:
            ball_model = YOLO(config.ball_model_name)
            ball_model_info = self._build_model_info(ball_model)
        else:
            ball_model = model
            ball_model_info = model_info

        results = model.track(
            source=str(video.path),
            stream=True,
            conf=config.detection_confidence,
            iou=config.detection_iou,
            imgsz=config.detection_imgsz,
            tracker=config.tracker_config,
            persist=True,
            vid_stride=config.frame_step,
            verbose=False,
        )

        tracks: list[TrackObservation] = []
        for result_index, result in enumerate(results):
            source_frame_index = result_index * config.frame_step
            frame_tracks = self._result_to_tracks(
                result,
                source_frame_index,
                video,
                model_info["label_map"],
            )
            frame_tracks = self._normalize_frame_tracks(frame_tracks, config)
            tracks.extend(frame_tracks)

        if config.dedicated_ball_pass:
            tracks = self._replace_ball_tracks_with_dedicated_pass(
                model=ball_model,
                tracks=tracks,
                video=video,
                config=config,
                model_label_map=ball_model_info["label_map"],
            )

        tracks = self._filter_tracks(tracks, config)
        return ProviderArtifacts(tracks=tracks, processed_video_path=None)

    def _result_to_tracks(
        self,
        result,
        frame_index: int,
        video: VideoMeta,
        model_label_map: dict[int, tuple[str, str]],
    ) -> list[TrackObservation]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        class_ids = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
        confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else []
        xyxy_values = boxes.xyxy.cpu().tolist()
        track_ids = (
            boxes.id.int().cpu().tolist()
            if getattr(boxes, "id", None) is not None
            else list(range(len(xyxy_values)))
        )

        observations: list[TrackObservation] = []
        for xyxy, track_id, class_id, confidence in zip(
            xyxy_values, track_ids, class_ids, confidences, strict=False
        ):
            mapped_info = model_label_map.get(class_id)
            if mapped_info is None:
                continue
            label, source_label = mapped_info

            x1, y1, x2, y2 = xyxy
            observations.append(
                TrackObservation(
                    frame_index=frame_index,
                    track_id=int(track_id),
                    label=label,
                    source_label=source_label,
                    confidence=float(confidence),
                    bbox=BBox(
                        x1=max(0.0, min(1.0, x1 / video.width)),
                        y1=max(0.0, min(1.0, y1 / video.height)),
                        x2=max(0.0, min(1.0, x2 / video.width)),
                        y2=max(0.0, min(1.0, y2 / video.height)),
                    ),
                )
            )
        return observations

    def _normalize_frame_tracks(
        self,
        tracks: list[TrackObservation],
        config: PipelineConfig,
    ) -> list[TrackObservation]:
        players = [
            track
            for track in tracks
            if track.label == "player" and track.confidence >= config.min_player_confidence
        ]
        balls = [
            track
            for track in tracks
            if track.label == "ball" and track.confidence >= config.min_ball_confidence
        ]
        if balls:
            balls.sort(key=lambda track: track.confidence, reverse=True)
            balls = [balls[0].model_copy(update={"track_id": 0})]
        return players + balls

    def _build_model_info(self, model) -> dict[str, object]:
        names = getattr(model.model, "names", None) or {}
        normalized = {int(idx): str(name).strip().lower() for idx, name in names.items()}

        mapped: dict[int, tuple[str, str]] = {}
        for class_id, name in normalized.items():
            if name in {"person", "player"}:
                mapped[class_id] = ("player", "player")
            elif name == "goalkeeper":
                mapped[class_id] = ("player", "goalkeeper")
            elif name == "referee":
                mapped[class_id] = ("player", "referee")
            elif name in {"sports ball", "ball"}:
                mapped[class_id] = ("ball", "ball")
        return {
            "label_map": mapped,
            "has_goalkeeper_referee_classes": ("goalkeeper" in normalized.values() and "referee" in normalized.values()),
        }

    def _filter_tracks(
        self,
        tracks: list[TrackObservation],
        config: PipelineConfig,
    ) -> list[TrackObservation]:
        filtered: list[TrackObservation] = []
        for track in tracks:
            if track.label == "player":
                if track.confidence < config.min_player_confidence:
                    continue
                if _bbox_area_fraction(track.bbox) > config.max_player_box_area_fraction:
                    continue
            elif track.label == "ball":
                if track.confidence < config.min_ball_confidence:
                    continue
            filtered.append(track)

        counts = Counter((track.label, track.track_id) for track in filtered)
        return [
            track
            for track in filtered
            if track.label == "ball" or counts[(track.label, track.track_id)] >= config.min_track_length
        ]

    def _replace_ball_tracks_with_dedicated_pass(
        self,
        *,
        model,
        tracks: list[TrackObservation],
        video: VideoMeta,
        config: PipelineConfig,
        model_label_map: dict[int, tuple[str, str]],
    ) -> list[TrackObservation]:
        player_tracks = [track for track in tracks if track.label != "ball"]
        tracked_ball_tracks = [track.model_copy(update={"track_id": 0}) for track in tracks if track.label == "ball"]
        dedicated_ball_tracks = self._run_dedicated_ball_pass(
            model=model,
            video=video,
            config=config,
            model_label_map=model_label_map,
        )
        ball_tracks = self._merge_ball_sources(
            tracked_ball_tracks=tracked_ball_tracks,
            dedicated_ball_tracks=dedicated_ball_tracks,
        )
        return player_tracks + ball_tracks

    def _run_dedicated_ball_pass(
        self,
        *,
        model,
        video: VideoMeta,
        config: PipelineConfig,
        model_label_map: dict[int, tuple[str, str]],
    ) -> list[TrackObservation]:
        cap = cv2.VideoCapture(str(video.path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video for dedicated ball pass: {video.path}")

        class_ids = [idx for idx, mapped in model_label_map.items() if mapped[0] == "ball"]
        if not class_ids:
            cap.release()
            return []

        ball_tracks: list[TrackObservation] = []
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % config.frame_step != 0:
                    frame_index += 1
                    continue

                results = model.predict(
                    source=frame,
                    conf=max(0.12, min(config.detection_confidence, config.min_ball_confidence)),
                    iou=config.detection_iou,
                    imgsz=config.ball_detection_imgsz,
                    verbose=False,
                    classes=class_ids,
                )
                if results:
                    frame_tracks = self._result_to_tracks(
                        results[0],
                        frame_index,
                        video,
                        model_label_map,
                    )
                    balls = [track for track in frame_tracks if track.label == "ball"]
                    if config.drop_ambiguous_ball_frames and len(balls) > 1:
                        frame_index += 1
                        continue
                    if balls:
                        balls.sort(key=lambda track: track.confidence, reverse=True)
                        ball_tracks.append(balls[0].model_copy(update={"track_id": 0}))
                frame_index += 1
        finally:
            cap.release()

        return ball_tracks

    def _merge_ball_sources(
        self,
        *,
        tracked_ball_tracks: list[TrackObservation],
        dedicated_ball_tracks: list[TrackObservation],
    ) -> list[TrackObservation]:
        merged_by_frame: dict[int, TrackObservation] = {}

        for track in tracked_ball_tracks:
            current = merged_by_frame.get(track.frame_index)
            if current is None or track.confidence > current.confidence:
                merged_by_frame[track.frame_index] = track

        for track in dedicated_ball_tracks:
            current = merged_by_frame.get(track.frame_index)
            if current is None or track.confidence >= current.confidence:
                merged_by_frame[track.frame_index] = track

        return [merged_by_frame[frame_index] for frame_index in sorted(merged_by_frame)]


def _bbox_area_fraction(bbox: BBox) -> float:
    return max(0.0, bbox.x2 - bbox.x1) * max(0.0, bbox.y2 - bbox.y1)
