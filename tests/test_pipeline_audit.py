from types import SimpleNamespace

import pytest


def test_event_checkpoint_mismatch_has_actionable_error():
    from ez_worker.analytics.event_spotter import _validate_checkpoint_classes
    with pytest.raises(ValueError, match="18 output classes.*expects 14"):
        _validate_checkpoint_classes({"head.2.weight": SimpleNamespace(shape=(18, 128))}, "model.pt")


def test_event_checkpoint_matching_shape_is_accepted():
    from ez_worker.analytics.event_spotter import _validate_checkpoint_classes
    _validate_checkpoint_classes({"head.2.weight": SimpleNamespace(shape=(14, 128))}, "model.pt")


def test_event_checkpoint_missing_head_is_rejected():
    from ez_worker.analytics.event_spotter import _validate_checkpoint_classes
    with pytest.raises(ValueError, match="unknown output classes"):
        _validate_checkpoint_classes({}, "model.pt")


def test_radar_keypoints_expire():
    from ez_worker.io.render_from_tracks import _keypoints_for_frame
    data = {"stride": 5, "frames": {"0": {"keypoints": {
        "0": [10, 10], "1": [20, 10], "2": [10, 20], "3": [20, 20]}}}}
    assert _keypoints_for_frame(data, 0, (100, 100)) is not None
    assert _keypoints_for_frame(data, 15, (100, 100)) is not None
    assert _keypoints_for_frame(data, 16, (100, 100)) is None
