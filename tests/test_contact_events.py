import pytest

from ez_worker.analytics.contacts import detect_contact_events
from ez_worker.schemas import BBox, TrackObservation, VideoMeta


def scene(fps=25, target_team=0, pan=0, flyby=False, interpolated=False):
    video = VideoMeta(path="test.mp4", fps=fps, frame_count=fps * 3, width=1000, height=500)
    tracks = []
    for f in range(video.frame_count):
        t = f / fps
        shift = pan * t
        for tid, x, team in ((1, .2, 0), (2, .6, target_team), (3, .8, 1), (4, .9, 1)):
            tracks.append(TrackObservation(frame_index=f, track_id=tid, label="player", team_id=team,
                          bbox=BBox(x1=x-.02+shift, x2=x+.02+shift, y1=.4, y2=.6)))
        x = .2 if t < 1 else min(.6, .2 + (t-1) * .4)
        if flyby:
            x = .2 + max(0, t-1) * .4
        tracks.append(TrackObservation(frame_index=f, track_id=0, label="ball", is_interpolated=interpolated,
                      bbox=BBox(x1=x-.003+shift, x2=x+.003+shift, y1=.588, y2=.6)))
    return tracks, video


@pytest.mark.parametrize("fps", [25, 60])
@pytest.mark.parametrize("pan", [0, .05])
def test_controlled_pass_without_absolute_speed_gate(fps, pan):
    tracks, video = scene(fps=fps, pan=pan)
    events, evidence, possession = detect_contact_events(tracks, video)
    assert [(e.event_type, e.actor_track_id, e.target_track_id) for e in events] == [("pass", 1, 2)]
    assert evidence["experimental"] and possession


def test_opponent_contact_is_not_automatically_interception():
    tracks, video = scene(target_team=1)
    events, _, _ = detect_contact_events(tracks, video)
    assert [e.event_type for e in events] == ["ball_transfer"]


def test_interpolated_ball_cannot_confirm_contact():
    tracks, video = scene(interpolated=True)
    events, _, possession = detect_contact_events(tracks, video)
    assert not events and not possession


def test_straight_through_bystander_not_pass():
    tracks, video = scene(flyby=True)
    events, _, _ = detect_contact_events(tracks, video)
    assert not events


def test_cut_does_not_join_two_possessions():
    tracks, video = scene()
    events, evidence, _ = detect_contact_events(tracks, video, cut_frames=[35])
    assert not events
    assert 35 in evidence["cut_frames"]


def test_sequential_id_fragment_is_not_pass():
    tracks, video = scene()
    for o in tracks:
        if o.label == "ball":
            o.bbox = BBox(x1=.197, x2=.203, y1=.588, y2=.6)
        if o.track_id == 1 and o.frame_index > 30:
            o.track_id = 10
    events, _, _ = detect_contact_events(tracks, video)
    assert not events


def test_unknown_teams_not_pass():
    tracks, video = scene(target_team=None)
    events, _, _ = detect_contact_events(tracks, video)
    assert all(e.event_type == "ball_transfer" for e in events)


def test_airborne_ball_cannot_confirm_foot_contact():
    tracks, video = scene()
    for o in tracks:
        if o.label == "ball":
            o.bbox.y1, o.bbox.y2 = .3, .312
    events, _, _ = detect_contact_events(tracks, video)
    assert not events


def test_overlapping_players_do_not_confirm_contact():
    tracks, video = scene()
    duplicate = [o.model_copy(update={"track_id": o.track_id + 10, "team_id": 1})
                 for o in tracks if o.label == "player"]
    events, _, possession = detect_contact_events(tracks + duplicate, video)
    assert not events and not possession


def test_missing_flight_is_not_confirmed_pass():
    tracks, video = scene()
    tracks = [o for o in tracks if not (o.label == "ball" and 28 <= o.frame_index < 45)]
    events, evidence, _ = detect_contact_events(tracks, video)
    assert not events
    assert any(t["reason"] == "insufficient_ball_coverage" for t in evidence["transfers"])


def test_replay_preserves_source_and_marks_unsupported_events(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("replay", Path(__file__).resolve().parents[1] / "scripts/replay_events.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    source.mkdir()
    tracks, video = scene()
    payload = json.dumps([t.model_dump(mode="json") for t in tracks])
    (source / "tracks.json").write_text(payload, encoding="utf-8")
    (source / "video_meta.json").write_text(video.model_dump_json(), encoding="utf-8")
    destination = tmp_path / "result"
    module.main(["--run-dir", str(source), "--output-dir", str(destination), "--skip-video"])
    assert (source / "tracks.json").read_text(encoding="utf-8") == payload
    assert not (source / "events.json").exists()
    assert json.loads((destination / "tracks.json").read_text()) == json.loads(payload)
    report = json.loads((destination / "match_report.json").read_text())
    assert report["summary"]["total_passes"] == 1
    assert "goal" in report["event_detection"]["unsupported"]
    with pytest.raises(FileExistsError):
        module.main(["--run-dir", str(source), "--output-dir", str(destination), "--skip-video"])
