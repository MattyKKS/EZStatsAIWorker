import numpy as np

from ez_worker.spatial.field_positions import FieldProjector, export_field_positions
from ez_worker.schemas import BBox, TrackObservation, VideoMeta


VERTICES = [(0, 0), (100, 0), (0, 50), (100, 50), (50, 0), (50, 50), (0, 25), (100, 25)]


def data():
    points = {str(i): [x * 2 + 10, y * 2 + 20] for i, (x, y) in enumerate(VERTICES)}
    return {"stride": 1, "video_wh": [300, 200], "frames": {"0": {"keypoints": points}}}


def test_projection_known_transform_and_expiry():
    p = FieldProjector(data(), VERTICES)
    assert np.allclose(p.point(0, 110, 70), [50, 25])
    assert p.point(4, 110, 70) is None
    assert p.point(0, 10000, 70) is None


def test_no_projection_across_camera_cut():
    assert FieldProjector(data(), VERTICES, cut_frames=[2]).point(2, 110, 70) is None


def test_partial_calibration_does_not_discard_good_positions():
    video = VideoMeta(path="unused.mp4", fps=25, frame_count=20, width=300, height=200)
    tracks = [TrackObservation(frame_index=f, track_id=1, label="player",
                              bbox=BBox(x1=100/300, x2=120/300, y1=.2, y2=.35)) for f in (0, 19)]
    result = export_field_positions(tracks, video, data(), VERTICES)
    assert result["positions"][0]["available"]
    assert not result["positions"][1]["available"]
    assert result["player_projection_coverage"] == .5


def test_degenerate_landmarks_are_rejected():
    d = data()
    d["frames"]["0"]["keypoints"] = {str(i): [i, 0] for i in range(8)}
    assert FieldProjector(d, VERTICES).point(0, 1, 1) is None
