from pathlib import Path

from ez_worker.schemas import VideoMeta


def load_video_meta(video_path: Path) -> VideoMeta:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Install the 'cv' extras before reading videos."
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Video metadata could not be read from: {video_path}")

    return VideoMeta(
        path=video_path,
        fps=fps if fps > 0 else 25.0,
        frame_count=frame_count,
        width=width,
        height=height,
    )
