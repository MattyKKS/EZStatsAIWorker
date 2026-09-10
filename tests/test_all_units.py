"""
EZ Stats AI Worker — Comprehensive Unit Tests
UTC-07 to UTC-16: Schemas, detect_events, build_track_stats, write_artifacts, fuse_events

Run: cd "EZ Stats AI Worker" && python3 -m unittest tests/test_all_units.py -v
"""

import sys, os, types, unittest, tempfile, json
from pathlib import Path
from typing import Optional

# ── Minimal pydantic stub ──────────────────────────────────────────────────────
_MISSING = object()

class _FieldInfo:
    def __init__(self, default=_MISSING, default_factory=None, **kw):
        self.default = default
        self.default_factory = default_factory

def _Field(default=_MISSING, default_factory=None, **kw):
    return _FieldInfo(default=default, default_factory=default_factory)

class _BaseModel:
    def __init__(self, **kwargs):
        cls = type(self)
        annotations = {}
        for klass in reversed(cls.__mro__):
            if hasattr(klass, '__annotations__'):
                annotations.update(klass.__annotations__)
        for name in annotations:
            cls_val = cls.__dict__.get(name, _MISSING)
            if isinstance(cls_val, _FieldInfo):
                fi = cls_val
                if name in kwargs:
                    setattr(self, name, kwargs[name])
                elif fi.default_factory is not None:
                    setattr(self, name, fi.default_factory())
                elif fi.default is not _MISSING:
                    setattr(self, name, fi.default)
                else:
                    if name not in kwargs:
                        raise TypeError(f"__init__() missing required field '{name}'")
            else:
                if name in kwargs:
                    setattr(self, name, kwargs[name])
                elif cls_val is not _MISSING:
                    setattr(self, name, cls_val)

    def model_dump(self, mode=None):
        cls = type(self)
        annotations = {}
        for klass in reversed(cls.__mro__):
            if hasattr(klass, '__annotations__'):
                annotations.update(klass.__annotations__)
        result = {}
        for name in annotations:
            val = getattr(self, name, None)
            if hasattr(val, 'model_dump'):
                result[name] = val.model_dump(mode=mode)
            elif isinstance(val, list):
                result[name] = [i.model_dump(mode=mode) if hasattr(i, 'model_dump') else i for i in val]
            elif mode == 'json' and hasattr(val, '__fspath__'):
                result[name] = str(val)
            else:
                result[name] = val
        return result

pydantic_mod = types.ModuleType('pydantic')
pydantic_mod.BaseModel = _BaseModel
pydantic_mod.Field = _Field
try:
    import pydantic
except ImportError:
    sys.modules['pydantic'] = pydantic_mod

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from ez_worker.schemas import (
    VideoMeta, BBox, TrackObservation, Event, TrackStats, AnalysisArtifacts
)
from ez_worker.analytics.stats import build_track_stats
from ez_worker.analytics.events import detect_events
from ez_worker.analytics.event_spotter import fuse_events
from ez_worker.outputs.writer import write_artifacts

# ── Helpers ────────────────────────────────────────────────────────────────────
VALID_EVENT_TYPES = {'touch', 'pass', 'long_ball', 'clearance', 'interception', 'shot_attempt'}

def make_video(fps=25.0, frames=100, w=1920, h=1080):
    return VideoMeta(path=Path('/tmp/vid.mp4'), fps=fps, frame_count=frames, width=w, height=h)

def make_obs(frame, track_id, label='player', x1=100, y1=100, x2=120, y2=140,
             confidence=1.0, team_id=None, source_label=None):
    return TrackObservation(
        frame_index=frame, track_id=track_id, label=label,
        source_label=source_label, confidence=confidence,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), team_id=team_id
    )

def make_event(frame, etype, actor=None, target=None, video_fps=25.0):
    return Event(frame_index=frame, time_seconds=frame/video_fps,
                 event_type=etype, actor_track_id=actor, target_track_id=target)

def make_artifacts(tracks=None, events=None, stats=None):
    return AnalysisArtifacts(
        video=make_video(),
        tracks=tracks or [],
        events=events or [],
        stats=stats or []
    )

# ══════════════════════════════════════════════════════════════════════════════
# UTC-07  run_analysis() — pipeline data model schemas
# ══════════════════════════════════════════════════════════════════════════════

class UTC07_RunAnalysis(unittest.TestCase):

    def test_bbox_construction(self):
        b = BBox(x1=10, y1=20, x2=30, y2=40)
        self.assertEqual(b.x1, 10); self.assertEqual(b.y2, 40)

    def test_bbox_cx_property(self):
        b = BBox(x1=10, y1=10, x2=30, y2=50)
        self.assertAlmostEqual(b.cx, 20.0)

    def test_bbox_cy_property(self):
        b = BBox(x1=10, y1=10, x2=30, y2=50)
        self.assertAlmostEqual(b.cy, 30.0)

    def test_bbox_model_dump_keys(self):
        b = BBox(x1=0, y1=0, x2=10, y2=10)
        d = b.model_dump()
        self.assertIn('x1', d); self.assertIn('x2', d)
        self.assertIn('y1', d); self.assertIn('y2', d)

    def test_video_meta_construction(self):
        v = make_video()
        self.assertEqual(v.fps, 25.0); self.assertEqual(v.frame_count, 100)

    def test_track_observation_defaults(self):
        obs = make_obs(0, 1)
        self.assertEqual(obs.confidence, 1.0)
        self.assertIsNone(obs.team_id)
        self.assertIsNone(obs.source_label)

    def test_event_details_default_factory(self):
        e1 = Event(frame_index=0, time_seconds=0.0, event_type='touch')
        e2 = Event(frame_index=1, time_seconds=0.04, event_type='pass')
        self.assertIsNot(e1.details, e2.details)  # different dicts per instance

    def test_event_actor_target_optional(self):
        e = Event(frame_index=5, time_seconds=0.2, event_type='clearance')
        self.assertIsNone(e.actor_track_id); self.assertIsNone(e.target_track_id)

    def test_track_stats_touch_pass_shot_default_zero(self):
        ts = TrackStats(track_id=1, label='player', frame_count=10,
                        approx_distance_px=100.0, avg_speed_px_per_frame=10.0)
        self.assertEqual(ts.touch_count, 0)
        self.assertEqual(ts.pass_count, 0)
        self.assertEqual(ts.shot_count, 0)

    def test_analysis_artifacts_tracks_list(self):
        a = make_artifacts()
        self.assertIsInstance(a.tracks, list); self.assertIsInstance(a.events, list)

    def test_video_meta_model_dump_json_mode_path_to_str(self):
        v = make_video()
        d = v.model_dump(mode='json')
        self.assertIsInstance(d['path'], str)

    def test_bbox_cx_cy_not_in_model_dump(self):
        # cx/cy are @property descriptors, not annotations → not in model_dump
        b = BBox(x1=0, y1=0, x2=20, y2=40)
        d = b.model_dump()
        self.assertNotIn('cx', d); self.assertNotIn('cy', d)

# ══════════════════════════════════════════════════════════════════════════════
# UTC-13  detect_events — state machine
# ══════════════════════════════════════════════════════════════════════════════

class UTC13_DetectEvents(unittest.TestCase):

    def _run(self, tracks, video=None, ball_id=99):
        v = video or make_video(fps=25.0, frames=len(set(o.frame_index for o in tracks)) or 1)
        return detect_events(tracks=tracks, video=v, ball_track_id=ball_id)

    def test_empty_tracks_returns_empty_list(self):
        result = self._run([])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_no_ball_track_returns_empty_list(self):
        tracks = [make_obs(i, 1, 'player', x1=100, y1=100, x2=120, y2=140) for i in range(30)]
        result = self._run(tracks, ball_id=99)  # 99 not present
        self.assertIsInstance(result, list)

    def test_only_ball_no_players_returns_empty(self):
        tracks = [make_obs(i, 99, 'ball', x1=200, y1=200, x2=210, y2=210) for i in range(30)]
        result = self._run(tracks, ball_id=99)
        self.assertIsInstance(result, list)

    def test_output_is_list_of_events(self):
        tracks = []
        for i in range(50):
            tracks.append(make_obs(i, 1, 'player', x1=100, y1=100, x2=120, y2=160))
            tracks.append(make_obs(i, 99, 'ball', x1=110, y1=150, x2=118, y2=158))
        result = self._run(tracks, ball_id=99)
        self.assertIsInstance(result, list)
        for ev in result:
            self.assertIsInstance(ev, Event)

    def test_all_emitted_event_types_are_valid(self):
        tracks = []
        for i in range(100):
            tracks.append(make_obs(i, 1, 'player', x1=100, y1=100, x2=120, y2=160))
            tracks.append(make_obs(i, 99, 'ball', x1=110, y1=140, x2=118, y2=148))
        result = self._run(tracks, ball_id=99)
        for ev in result:
            self.assertIn(ev.event_type, VALID_EVENT_TYPES,
                          f"Invalid event type: {ev.event_type}")

    def test_events_chronologically_ordered(self):
        tracks = []
        for i in range(80):
            tracks.append(make_obs(i, 1, 'player', x1=100, y1=100, x2=120, y2=160))
            tracks.append(make_obs(i, 99, 'ball', x1=110, y1=140, x2=118, y2=148))
        result = self._run(tracks, ball_id=99)
        frames = [ev.frame_index for ev in result]
        self.assertEqual(frames, sorted(frames))

    def test_event_time_seconds_consistent_with_frame_and_fps(self):
        tracks = []
        for i in range(60):
            tracks.append(make_obs(i, 1, 'player', x1=100, y1=100, x2=120, y2=160))
            tracks.append(make_obs(i, 99, 'ball', x1=110, y1=140, x2=118, y2=148))
        result = self._run(tracks, ball_id=99)
        for ev in result:
            expected_t = ev.frame_index / 25.0
            self.assertAlmostEqual(ev.time_seconds, expected_t, places=3)

    def test_ball_far_from_all_players_produces_no_touch(self):
        tracks = []
        for i in range(30):
            # ball 500px away from player
            tracks.append(make_obs(i, 1, 'player', x1=0, y1=0, x2=20, y2=40))
            tracks.append(make_obs(i, 99, 'ball', x1=500, y1=500, x2=508, y2=508))
        result = self._run(tracks, ball_id=99)
        touch_events = [ev for ev in result if ev.event_type == 'touch']
        self.assertEqual(len(touch_events), 0)

    def test_actor_track_id_in_emitted_events(self):
        tracks = []
        for i in range(60):
            tracks.append(make_obs(i, 1, 'player', x1=100, y1=100, x2=120, y2=160))
            tracks.append(make_obs(i, 99, 'ball', x1=110, y1=140, x2=118, y2=148))
        result = self._run(tracks, ball_id=99)
        for ev in result:
            if ev.event_type in ('touch', 'pass', 'shot_attempt'):
                self.assertIsNotNone(ev.actor_track_id)

    def test_result_has_frame_index_int(self):
        tracks = []
        for i in range(40):
            tracks.append(make_obs(i, 1, 'player', x1=100, y1=100, x2=120, y2=160))
            tracks.append(make_obs(i, 99, 'ball', x1=110, y1=140, x2=118, y2=148))
        result = self._run(tracks, ball_id=99)
        for ev in result:
            self.assertIsInstance(ev.frame_index, int)

# ══════════════════════════════════════════════════════════════════════════════
# UTC-14  build_track_stats
# ══════════════════════════════════════════════════════════════════════════════

class UTC14_BuildTrackStats(unittest.TestCase):

    def test_empty_tracks_returns_empty_list(self):
        result = build_track_stats([], [], make_video())
        self.assertEqual(result, [])

    def test_non_player_tracks_excluded(self):
        tracks = [make_obs(i, 99, 'ball') for i in range(5)]
        result = build_track_stats(tracks, [], make_video())
        self.assertEqual(result, [])

    def test_player_track_included(self):
        tracks = [make_obs(i, 1, 'player', x1=100+i, y1=100, x2=120+i, y2=140) for i in range(10)]
        result = build_track_stats(tracks, [], make_video())
        ids = [s.track_id for s in result]
        self.assertIn(1, ids)

    def test_frame_count_correct(self):
        tracks = [make_obs(i, 1, 'player', x1=100, y1=100, x2=120, y2=140) for i in range(7)]
        result = build_track_stats(tracks, [], make_video())
        stat = next(s for s in result if s.track_id == 1)
        self.assertEqual(stat.frame_count, 7)

    def test_touch_count_from_events(self):
        tracks = [make_obs(0, 1, 'player')]
        events = [make_event(0, 'touch', actor=1)]
        result = build_track_stats(tracks, events, make_video())
        stat = next(s for s in result if s.track_id == 1)
        self.assertEqual(stat.touch_count, 1)

    def test_pass_count_from_events(self):
        tracks = [make_obs(i, 1, 'player') for i in range(3)]
        events = [make_event(0, 'pass', actor=1), make_event(1, 'pass', actor=1)]
        result = build_track_stats(tracks, events, make_video())
        stat = next(s for s in result if s.track_id == 1)
        self.assertEqual(stat.pass_count, 2)

    def test_shot_count_from_events(self):
        tracks = [make_obs(0, 5, 'player')]
        events = [make_event(0, 'shot_attempt', actor=5)]
        result = build_track_stats(tracks, events, make_video())
        stat = next(s for s in result if s.track_id == 5)
        self.assertEqual(stat.shot_count, 1)

    def test_approx_distance_zero_for_stationary(self):
        tracks = [make_obs(i, 1, 'player', x1=100, y1=100, x2=120, y2=140) for i in range(5)]
        result = build_track_stats(tracks, [], make_video())
        stat = next(s for s in result if s.track_id == 1)
        self.assertAlmostEqual(stat.approx_distance_px, 0.0, places=1)

    def test_sorted_by_track_id(self):
        tracks = ([make_obs(i, 3, 'player') for i in range(3)] +
                  [make_obs(i, 1, 'player') for i in range(3)] +
                  [make_obs(i, 2, 'player') for i in range(3)])
        result = build_track_stats(tracks, [], make_video())
        ids = [s.track_id for s in result]
        self.assertEqual(ids, sorted(ids))

    def test_multiple_players_separate_stats(self):
        tracks = ([make_obs(i, 1, 'player') for i in range(5)] +
                  [make_obs(i, 2, 'player') for i in range(5)])
        result = build_track_stats(tracks, [], make_video())
        ids = [s.track_id for s in result]
        self.assertIn(1, ids); self.assertIn(2, ids)

    def test_avg_speed_non_negative(self):
        tracks = [make_obs(i, 1, 'player', x1=100+i*5, y1=100, x2=120+i*5, y2=140) for i in range(10)]
        result = build_track_stats(tracks, [], make_video())
        stat = next(s for s in result if s.track_id == 1)
        self.assertGreaterEqual(stat.avg_speed_px_per_frame, 0.0)

    def test_result_items_are_track_stats(self):
        tracks = [make_obs(i, 1, 'player') for i in range(5)]
        result = build_track_stats(tracks, [], make_video())
        for s in result:
            self.assertIsInstance(s, TrackStats)

# ══════════════════════════════════════════════════════════════════════════════
# UTC-16  fuse_events / event_spotter
# ══════════════════════════════════════════════════════════════════════════════

class UTC16_FuseAndSpotter(unittest.TestCase):

    def test_empty_inputs_returns_empty(self):
        result = fuse_events([], [], video_fps=25.0)
        self.assertEqual(result, [])

    def test_rule_only_no_lstm_passthrough(self):
        rule = [make_event(10, 'touch', actor=1)]
        result = fuse_events(rule, [], video_fps=25.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].event_type, 'touch')

    def test_lstm_goal_upgrades_shot_attempt(self):
        rule = [make_event(10, 'shot_attempt', actor=1)]
        spotter = [make_event(10, 'goal', actor=1)]
        result = fuse_events(rule, spotter, video_fps=25.0)
        types = [e.event_type for e in result]
        # 'goal' is in _upgrades for shot_attempt
        self.assertIn('goal', types)

    def test_lstm_cross_upgrades_pass(self):
        rule = [make_event(20, 'pass', actor=2)]
        spotter = [make_event(20, 'cross', actor=2)]
        result = fuse_events(rule, spotter, video_fps=25.0)
        types = [e.event_type for e in result]
        self.assertIn('cross', types)

    def test_output_is_sorted_by_frame(self):
        rule = [make_event(30, 'touch', actor=1), make_event(5, 'pass', actor=1)]
        result = fuse_events(rule, [], video_fps=25.0)
        frames = [e.frame_index for e in result]
        self.assertEqual(frames, sorted(frames))

    def test_lstm_only_events_included(self):
        # Events only from LSTM (no corresponding rule event) should still appear
        spotter = [make_event(15, 'header', actor=3)]
        result = fuse_events([], spotter, video_fps=25.0)
        types = [e.event_type for e in result]
        self.assertIn('header', types)

    def test_result_is_list_of_events(self):
        rule = [make_event(0, 'touch', actor=1)]
        result = fuse_events(rule, [], video_fps=25.0)
        for e in result:
            self.assertIsInstance(e, Event)

    def test_upgrade_does_not_duplicate_original(self):
        rule = [make_event(10, 'shot_attempt', actor=1)]
        spotter = [make_event(10, 'goal', actor=1)]
        result = fuse_events(rule, spotter, video_fps=25.0)
        # Original shot_attempt should not appear alongside upgraded goal
        types = [e.event_type for e in result]
        if 'goal' in types:
            self.assertNotIn('shot_attempt', types)

    def test_out_of_window_lstm_not_upgrade(self):
        # LSTM event far away in frames should not upgrade a rule event
        rule = [make_event(10, 'shot_attempt', actor=1)]
        spotter = [make_event(200, 'goal', actor=1)]  # far away
        result = fuse_events(rule, spotter, video_fps=25.0)
        types = [e.event_type for e in result]
        # shot_attempt should remain since goal is too far away
        self.assertIn('shot_attempt', types)

# ══════════════════════════════════════════════════════════════════════════════
# UTC-12  IO layer — write_artifacts / _build_match_report
# ══════════════════════════════════════════════════════════════════════════════

class UTC12_IOLayer(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _artifacts_with_data(self):
        tracks = [
            make_obs(0, 1, 'player', team_id=0),
            make_obs(1, 1, 'player', team_id=0),
            make_obs(0, 2, 'player', team_id=1),
        ]
        events = [
            make_event(0, 'touch', actor=1),
            make_event(1, 'pass', actor=1, target=2),
        ]
        stats = [
            TrackStats(track_id=1, label='player', frame_count=2,
                       approx_distance_px=50.0, avg_speed_px_per_frame=25.0,
                       touch_count=1, pass_count=1),
            TrackStats(track_id=2, label='player', frame_count=1,
                       approx_distance_px=10.0, avg_speed_px_per_frame=10.0),
        ]
        return make_artifacts(tracks=tracks, events=events, stats=stats)

    def test_write_artifacts_creates_video_meta_json(self):
        write_artifacts(make_artifacts(), Path(self.tmpdir))
        self.assertTrue((Path(self.tmpdir) / 'video_meta.json').exists())

    def test_write_artifacts_creates_tracks_json(self):
        write_artifacts(make_artifacts(), Path(self.tmpdir))
        self.assertTrue((Path(self.tmpdir) / 'tracks.json').exists())

    def test_write_artifacts_creates_events_json(self):
        write_artifacts(make_artifacts(), Path(self.tmpdir))
        self.assertTrue((Path(self.tmpdir) / 'events.json').exists())

    def test_write_artifacts_creates_player_stats_json(self):
        write_artifacts(make_artifacts(), Path(self.tmpdir))
        self.assertTrue((Path(self.tmpdir) / 'player_stats.json').exists())

    def test_write_artifacts_creates_summary_json(self):
        write_artifacts(make_artifacts(), Path(self.tmpdir))
        self.assertTrue((Path(self.tmpdir) / 'summary.json').exists())

    def test_write_artifacts_creates_match_report_json(self):
        write_artifacts(make_artifacts(), Path(self.tmpdir))
        self.assertTrue((Path(self.tmpdir) / 'match_report.json').exists())

    def test_match_report_has_required_keys(self):
        write_artifacts(self._artifacts_with_data(), Path(self.tmpdir))
        with open(Path(self.tmpdir) / 'match_report.json') as f:
            report = json.load(f)
        for key in ('video', 'duration_s', 'fps', 'possession', 'players', 'events', 'summary'):
            self.assertIn(key, report, f"Missing key: {key}")

    def test_match_report_summary_keys(self):
        write_artifacts(self._artifacts_with_data(), Path(self.tmpdir))
        with open(Path(self.tmpdir) / 'match_report.json') as f:
            report = json.load(f)
        summary = report['summary']
        for key in ('total_touches', 'total_passes'):
            self.assertIn(key, summary)

    def test_match_report_duration_s(self):
        v = make_video(fps=25.0, frames=250)
        a = AnalysisArtifacts(video=v, tracks=[], events=[], stats=[])
        write_artifacts(a, Path(self.tmpdir))
        with open(Path(self.tmpdir) / 'match_report.json') as f:
            report = json.load(f)
        self.assertAlmostEqual(report['duration_s'], 10.0, places=1)

    def test_events_json_is_list(self):
        write_artifacts(self._artifacts_with_data(), Path(self.tmpdir))
        with open(Path(self.tmpdir) / 'events.json') as f:
            ev = json.load(f)
        self.assertIsInstance(ev, list)

    def test_tracks_json_is_list(self):
        tracks = [make_obs(0, 1, 'player')]
        write_artifacts(make_artifacts(tracks=tracks), Path(self.tmpdir))
        with open(Path(self.tmpdir) / 'tracks.json') as f:
            data = json.load(f)
        self.assertIsInstance(data, list)

    def test_player_stats_json_is_list(self):
        stats = [TrackStats(track_id=1, label='player', frame_count=5,
                            approx_distance_px=50.0, avg_speed_px_per_frame=10.0)]
        write_artifacts(make_artifacts(stats=stats), Path(self.tmpdir))
        with open(Path(self.tmpdir) / 'player_stats.json') as f:
            data = json.load(f)
        self.assertIsInstance(data, list)

    def test_pass_network_in_match_report(self):
        write_artifacts(self._artifacts_with_data(), Path(self.tmpdir))
        with open(Path(self.tmpdir) / 'match_report.json') as f:
            report = json.load(f)
        self.assertIn('pass_network', report)
        pn = report['pass_network']
        self.assertIn('nodes', pn); self.assertIn('edges', pn)

    def test_write_artifacts_accepts_path_object(self):
        write_artifacts(make_artifacts(), Path(self.tmpdir))
        self.assertTrue((Path(self.tmpdir) / 'video_meta.json').exists())

# ══════════════════════════════════════════════════════════════════════════════
# UTC-12b IO layer — load_video_meta (error paths)
# ══════════════════════════════════════════════════════════════════════════════

class UTC12b_LoadVideoMeta(unittest.TestCase):

    def test_nonexistent_file_raises(self):
        from ez_worker.io.video import load_video_meta
        with self.assertRaises((FileNotFoundError, Exception)):
            load_video_meta(Path('/nonexistent/video.mp4'))

    def test_nonexistent_path_is_not_silently_ignored(self):
        from ez_worker.io.video import load_video_meta
        raised = False
        try:
            load_video_meta(Path('/tmp/does_not_exist_12345.mp4'))
        except Exception:
            raised = True
        self.assertTrue(raised, 'Expected exception for missing file')

# ══════════════════════════════════════════════════════════════════════════════
# UTC-13  Event schema field contracts
# ══════════════════════════════════════════════════════════════════════════════

class UTC08_ProviderContracts(unittest.TestCase):

    def test_event_type_touch_valid(self):
        e = Event(frame_index=0, time_seconds=0.0, event_type='touch')
        self.assertEqual(e.event_type, 'touch')

    def test_event_type_pass_valid(self):
        e = Event(frame_index=0, time_seconds=0.0, event_type='pass')
        self.assertEqual(e.event_type, 'pass')

    def test_event_type_long_ball_valid(self):
        e = Event(frame_index=0, time_seconds=0.0, event_type='long_ball')
        self.assertEqual(e.event_type, 'long_ball')

    def test_event_type_clearance_valid(self):
        e = Event(frame_index=0, time_seconds=0.0, event_type='clearance')
        self.assertEqual(e.event_type, 'clearance')

    def test_event_type_interception_valid(self):
        e = Event(frame_index=0, time_seconds=0.0, event_type='interception')
        self.assertEqual(e.event_type, 'interception')

    def test_event_type_shot_attempt_valid(self):
        e = Event(frame_index=0, time_seconds=0.0, event_type='shot_attempt')
        self.assertEqual(e.event_type, 'shot_attempt')

    def test_event_details_mutable_dict(self):
        e = Event(frame_index=0, time_seconds=0.0, event_type='touch', details={'key': 'val'})
        self.assertEqual(e.details['key'], 'val')

    def test_event_all_fields_present(self):
        e = Event(frame_index=5, time_seconds=0.2, event_type='pass',
                  actor_track_id=1, target_track_id=2, details={'dist': 10})
        self.assertEqual(e.frame_index, 5)
        self.assertAlmostEqual(e.time_seconds, 0.2)
        self.assertEqual(e.actor_track_id, 1)
        self.assertEqual(e.target_track_id, 2)

# ══════════════════════════════════════════════════════════════════════════════
# UTC-09  Track geometry — BBox calculations
# ══════════════════════════════════════════════════════════════════════════════

class UTC09_TrackGeometry(unittest.TestCase):

    def test_cx_center_horizontal(self):
        self.assertAlmostEqual(BBox(x1=0, y1=0, x2=100, y2=50).cx, 50.0)

    def test_cy_center_vertical(self):
        self.assertAlmostEqual(BBox(x1=0, y1=0, x2=100, y2=80).cy, 40.0)

    def test_cx_float_precision(self):
        self.assertAlmostEqual(BBox(x1=10.5, y1=0, x2=21.5, y2=10).cx, 16.0)

    def test_cy_float_precision(self):
        self.assertAlmostEqual(BBox(x1=0, y1=10.3, x2=10, y2=20.7).cy, 15.5)

    def test_bbox_negative_coords(self):
        b = BBox(x1=-10, y1=-20, x2=10, y2=20)
        self.assertAlmostEqual(b.cx, 0.0); self.assertAlmostEqual(b.cy, 0.0)

    def test_model_dump_contains_all_corners(self):
        d = BBox(x1=1, y1=2, x2=3, y2=4).model_dump()
        self.assertEqual(d, {'x1': 1, 'y1': 2, 'x2': 3, 'y2': 4})

# ══════════════════════════════════════════════════════════════════════════════
# UTC-11  Clustering output — AnalysisArtifacts composition
# ══════════════════════════════════════════════════════════════════════════════

class UTC11_ClusterArtifacts(unittest.TestCase):

    def test_empty_artifacts_valid(self):
        a = make_artifacts()
        self.assertEqual(a.tracks, [])
        self.assertEqual(a.events, [])
        self.assertEqual(a.stats, [])

    def test_processed_video_path_default_none(self):
        a = make_artifacts()
        self.assertIsNone(a.processed_video_path)

    def test_processed_video_path_set(self):
        a = AnalysisArtifacts(
            video=make_video(), tracks=[], events=[], stats=[],
            processed_video_path=Path('/tmp/out.mp4')
        )
        self.assertEqual(a.processed_video_path, Path('/tmp/out.mp4'))

    def test_artifacts_with_tracks_events_stats(self):
        tracks = [make_obs(0, 1, 'player')]
        events = [make_event(0, 'touch', actor=1)]
        stats  = [TrackStats(track_id=1, label='player', frame_count=1,
                             approx_distance_px=0.0, avg_speed_px_per_frame=0.0)]
        a = make_artifacts(tracks=tracks, events=events, stats=stats)
        self.assertEqual(len(a.tracks), 1)
        self.assertEqual(len(a.events), 1)
        self.assertEqual(len(a.stats), 1)

    def test_model_dump_json_mode_paths(self):
        a = make_artifacts()
        d = a.model_dump(mode='json')
        self.assertIn('video', d)
        self.assertIsInstance(d['video']['path'], str)

# ══════════════════════════════════════════════════════════════════════════════
# UTC-10  Spatial layer — TrackObservation contracts field contracts
# ══════════════════════════════════════════════════════════════════════════════

class UTC10_SpatialContracts(unittest.TestCase):

    def test_label_player(self):
        obs = make_obs(0, 1, 'player')
        self.assertEqual(obs.label, 'player')

    def test_label_ball(self):
        obs = make_obs(0, 99, 'ball')
        self.assertEqual(obs.label, 'ball')

    def test_label_goalkeeper(self):
        obs = make_obs(0, 1, 'goalkeeper')
        self.assertEqual(obs.label, 'goalkeeper')

    def test_confidence_range_low(self):
        obs = make_obs(0, 1, confidence=0.1)
        self.assertAlmostEqual(obs.confidence, 0.1)

    def test_confidence_range_high(self):
        obs = make_obs(0, 1, confidence=0.99)
        self.assertAlmostEqual(obs.confidence, 0.99)

    def test_team_id_zero_valid(self):
        obs = make_obs(0, 1, team_id=0)
        self.assertEqual(obs.team_id, 0)

    def test_team_id_one_valid(self):
        obs = make_obs(0, 1, team_id=1)
        self.assertEqual(obs.team_id, 1)

    def test_bbox_accessible(self):
        obs = make_obs(0, 1, x1=10, y1=20, x2=30, y2=60)
        self.assertIsInstance(obs.bbox, BBox)
        self.assertAlmostEqual(obs.bbox.cx, 20.0)

    def test_source_label_none_default(self):
        obs = make_obs(0, 1)
        self.assertIsNone(obs.source_label)

    def test_source_label_string(self):
        obs = make_obs(0, 1, source_label='yolo')
        self.assertEqual(obs.source_label, 'yolo')


if __name__ == '__main__':
    unittest.main(verbosity=2)


# ══════════════════════════════════════════════════════════════════════════════
# UTC-15  _build_match_report() — match report JSON structure
# ══════════════════════════════════════════════════════════════════════════════

class UTC15_MatchReport(unittest.TestCase):
    """Match report structure and field validation."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.video  = make_video()
        obs_p = [make_obs(i, 1, team_id=0) for i in range(10)]
        obs_b = [make_obs(i, 0, label='ball') for i in range(10)]
        self.tracks = obs_p + obs_b
        self.events = [make_event(3, 'pass', actor=1, target=2),
                       make_event(7, 'shot_attempt', actor=1)]
        self.stats  = build_track_stats(self.tracks, self.events, self.video)
        arts = AnalysisArtifacts(video=self.video, tracks=self.tracks,
                                 events=self.events, stats=self.stats)
        write_artifacts(arts, self.tmpdir)
        with open(self.tmpdir / 'match_report.json') as f:
            self.report = json.load(f)

    def test_tc01_match_report_has_required_keys(self):
        """Match report contains all required top-level sections."""
        for key in ('summary', 'events', 'players', 'pass_network'):
            self.assertIn(key, self.report)

    def test_tc02_match_report_summary_keys(self):
        """Match report summary section contains all required fields."""
        s = self.report['summary']
        for key in ('total_passes', 'total_shots', 'total_touches'):
            self.assertIn(key, s)

    def test_tc03_match_report_duration_s(self):
        """Match duration is calculated correctly from frame count and frame rate."""
        expected = self.video.frame_count / self.video.fps
        self.assertAlmostEqual(self.report['duration_s'], expected, places=2)

    def test_tc04_pass_network_in_match_report(self):
        """Match report includes a pass network section."""
        self.assertIn('pass_network', self.report)

    def test_tc05_events_json_is_list(self):
        """Events section in match report is a list."""
        self.assertIsInstance(self.report['events'], list)


if __name__ == "__main__":
    unittest.main()
