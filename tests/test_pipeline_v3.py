"""Regression checks for identity, event, and Colab output consistency."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ez_worker.analytics.events import detect_events
from ez_worker.analytics.stats import build_track_stats
from ez_worker.postprocess.canonical import assign_roles_and_teams
from ez_worker.postprocess.tracks import _interpolate_ball_short_gaps, _refine_ball_timeseries_tutorial
from ez_worker.providers.ultralytics_provider import UltralyticsTrackingProvider
from ez_worker.schemas import BBox, Event, TrackObservation, VideoMeta


def video():
    return VideoMeta(path=Path("clip.mp4"), fps=25, frame_count=30, width=1000, height=700)


def observation(frame=0, tid=1, role="player", x=0.3, interpolated=False):
    return TrackObservation(frame_index=frame, track_id=tid, label="ball" if role == "ball" else "player",
                            source_label=role, confidence=0.9, team_id=0,
                            is_interpolated=interpolated,
                            bbox=BBox(x1=x - 0.01, x2=x + 0.01, y1=0.4, y2=0.5))


def test_prediction_boxes_do_not_get_fake_player_ids():
    import torch
    boxes = SimpleNamespace(xyxy=torch.tensor([[10, 10, 20, 30], [30, 30, 33, 33]]),
                            cls=torch.tensor([0, 1]), conf=torch.tensor([0.9, 0.8]), id=None)
    tracks = UltralyticsTrackingProvider()._result_to_tracks(
        SimpleNamespace(boxes=boxes), 0, video(), {0: ("player", "player"), 1: ("ball", "ball")})
    assert len(tracks) == 1 and tracks[0].label == "ball" and tracks[0].track_id == 0


def test_ball_gap_limit_and_provenance():
    balls = [observation(0, 0, "ball"), observation(4, 0, "ball", x=0.4)]
    filled = _interpolate_ball_short_gaps(balls, frame_step=1, max_gap_frames=4)
    assert len(filled) == 5
    assert all(o.is_interpolated for o in filled[1:-1])
    assert not filled[0].is_interpolated and not filled[-1].is_interpolated
    refined = _refine_ball_timeseries_tutorial(balls, video=video(), frame_step=1,
                                              max_gap_frames=3, suspicious_hotspots=[])
    assert [o.frame_index for o in refined] == [0, 4]


def test_interpolation_cannot_create_possession_or_touches(monkeypatch):
    import ez_worker.analytics.events as module
    monkeypatch.setattr(module, "_compute_ball_velocity", lambda *args: {i: (5, 0) for i in range(20)})
    tracks = [o for i in range(20) for o in (observation(i), observation(i, 0, "ball", interpolated=True))]
    held = {}
    assert detect_events(tracks, video(), 0, possession_by_frame=held) == []
    assert not any(t is not None for t in held.values())


def test_single_frame_contact_does_not_confirm_touch(monkeypatch):
    import ez_worker.analytics.events as module
    monkeypatch.setattr(module, "_compute_ball_velocity", lambda *args: {1: (5, 0)})
    assert detect_events([observation(1), observation(1, 0, "ball")], video(), 0) == []


def test_ground_ball_not_airborne_because_of_foreground_players():
    from ez_worker.analytics.events import _is_ball_airborne
    near = observation()
    foreground = observation(tid=2).model_copy(update={"bbox": BBox(x1=0.29, x2=0.31, y1=0.7, y2=0.9)})
    assert not _is_ball_airborne(observation(tid=0, role="ball"), [near, foreground], video())


def test_roles_excluded_from_team_clustering_and_locked():
    tracks = [observation(i, tid, role) for i in range(10)
              for tid, role in [(1, "player"), (2, "player"), (3, "referee")]]
    tracks.append(observation(10, 3, "player"))
    colors = {1: np.array([60, 210, 70]), 2: np.array([230, 230, 230]), 3: np.array([20, 20, 20])}
    final, decisions = assign_roles_and_teams(tracks, colors)
    assert decisions["teams"][1] != decisions["teams"][2]
    assert 3 not in decisions["teams"]
    assert all(o.label == "referee" and o.team_id is None for o in final if o.track_id == 3)
    assert len(build_track_stats(final, [], video())) == 2


def test_shot_attempts_and_goals_count_in_stats():
    events = [Event(frame_index=i, time_seconds=i / 25, event_type=kind, actor_track_id=1)
              for i, kind in enumerate(("shot", "shot_attempt", "goal"))]
    assert build_track_stats([observation()], events, video())[0].shot_count == 3


def test_fragment_reconnection_rejects_ambiguous_and_concurrent_players():
    from ez_worker.postprocess.canonical import reconnect_fragments
    colours = {t: np.array([60, 210, 70]) for t in (1, 2, 3)}
    first = [observation(i, 1) for i in range(5)]
    second = [observation(i, 2) for i in range(6, 10)]
    result, remap = reconnect_fragments(first + second, colours, video())
    assert remap == {2: 1}
    assert {o.track_id for o in result} == {1}
    ambiguous = [observation(i, 3, x=0.301) for i in range(6, 10)]
    assert reconnect_fragments(first + second + ambiguous, colours, video())[1] == {}
    concurrent = [observation(i, 2) for i in range(5)]
    assert reconnect_fragments(first + concurrent, colours, video())[1] == {}


def test_final_report_tracks_and_stats_use_same_ids(tmp_path, monkeypatch):
    import run_pipeline_v3
    import ez_worker.postprocess.canonical as canonical
    import ez_worker.spatial.event_projection as projection
    tracks = [o for i in range(10) for o in (observation(i, 1), observation(i, 2, "referee"))]
    (tmp_path / "tracks_raw.json").write_text(json.dumps([o.model_dump(mode="json") for o in tracks]))
    (tmp_path / "video_meta.json").write_text(json.dumps(video().model_dump(mode="json")))
    monkeypatch.setattr(canonical, "track_jersey_colors", lambda *args: {})
    monkeypatch.setattr(projection, "project_tracks", lambda *args: (None, None))
    run_pipeline_v3.finish_run(tmp_path, render=False)
    report = json.loads((tmp_path / "match_report.json").read_text())
    assert [p["track_id"] for p in report["players"]] == [1]
    assert (tmp_path / "match_report.json").read_bytes() == (tmp_path / "match_report_merged.json").read_bytes()
    assert (tmp_path / "tracks.json").read_bytes() == (tmp_path / "tracks_with_teams.json").read_bytes()
    assert json.loads((tmp_path / "referee_track_ids.json").read_text()) == [2]


def test_colab_cells_compile_and_have_independent_runs():
    path = Path(__file__).resolve().parents[1] / "docs/run_on_colab_v3.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for i, source in enumerate(code):
        compile(source, f"cell_{i}", "exec")
    runs = [s for s in code if "run_colab_clip.py" in s]
    assert len(runs) == 4
    assert "leo_messi_30pass.mp4" in runs[0]
    assert "08fd33_4.mp4" in runs[1]
    assert all("runpy.run_path" in s and "subprocess.run" not in s for s in runs)


def test_notebook_setup_forwards_child_output(capsys):
    import ast
    path = Path(__file__).resolve().parents[1] / "docs/run_on_colab_v3.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][6]["source"])
    parsed = ast.parse(source)
    definitions = ast.Module(body=[node for node in parsed.body
                                   if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))],
                             type_ignores=[])
    namespace = {}
    exec(compile(definitions, "colab_setup", "exec"), namespace)
    namespace["run_checked"]([sys.executable, "-c", "print('setup progress visible')"])
    assert "setup progress visible" in capsys.readouterr().out
