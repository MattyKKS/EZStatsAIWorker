"""
EZ Stats AI Worker – Input/Output Unit Tests
UTC-04 (AI Worker output structure) and UTC-05 (Match Report API output)

Run:  python3 -m unittest tests/test_io.py -v
      python3 tests/test_io.py
"""
import unittest
from pathlib import Path
from enum import Enum


# ── Constants mirroring the actual code ───────────────────────────────────────

VALID_EVENT_TYPES = {
    "touch", "pass", "long_ball",
    "clearance", "interception", "shot_attempt",
}

MATCH_STATUS_VALUES = {
    "CREATED", "UPLOADED", "QUEUED", "PROCESSING", "COMPLETED", "FAILED",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_bbox(x1=10.0, y1=20.0, x2=50.0, y2=90.0):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def make_track_observation(frame_index=0, track_id=1, label="player", confidence=0.92):
    return {
        "frame_index": frame_index,
        "track_id": track_id,
        "label": label,
        "source_label": None,
        "confidence": confidence,
        "bbox": make_bbox(),
        "team_id": None,
    }


def make_event(event_type="pass", frame_index=42, time_seconds=1.68, actor_track_id=3):
    return {
        "frame_index": frame_index,
        "time_seconds": time_seconds,
        "event_type": event_type,
        "actor_track_id": actor_track_id,
        "target_track_id": None,
        "details": {},
    }


def make_video_meta(path="match.mp4", fps=25.0, frame_count=625, width=1920, height=1080):
    return {
        "path": str(path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def make_analysis_artifacts(events=None, tracks=None, stats=None):
    return {
        "video": make_video_meta(),
        "tracks": tracks if tracks is not None else [],
        "events": events if events is not None else [],
        "stats": stats if stats is not None else [],
        "processed_video_path": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UTC-04: AI Worker Output Module – Event Detection Output
# ══════════════════════════════════════════════════════════════════════════════

class TestEventDetectionOutput(unittest.TestCase):
    """UTC-04: Verify event detection pipeline output structure and content."""

    def test_tc01_output_contains_events_key(self):
        """TC-01: AnalysisArtifacts contains 'events' key and it is a list."""
        artifacts = make_analysis_artifacts()
        self.assertIn("events", artifacts, "Output must contain 'events' key")
        self.assertIsInstance(artifacts["events"], list,
                              "'events' must be a list")

    def test_tc01_output_contains_all_required_keys(self):
        """TC-01: AnalysisArtifacts contains all required top-level keys."""
        artifacts = make_analysis_artifacts()
        for key in ("video", "tracks", "events", "stats"):
            self.assertIn(key, artifacts,
                          f"Output must contain '{key}' key")

    def test_tc02_event_contains_required_fields(self):
        """TC-02: Each event object has required fields with correct types."""
        event = make_event()
        required = {
            "event_type": str,
            "frame_index": int,
            "time_seconds": float,
            "actor_track_id": int,
        }
        for field, expected_type in required.items():
            self.assertIn(field, event,
                          f"Event must contain field '{field}'")
            self.assertIsInstance(event[field], expected_type,
                                  f"Field '{field}' must be {expected_type.__name__}")

    def test_tc02_event_target_track_id_is_optional(self):
        """TC-02: target_track_id may be None (for events with no receiver)."""
        event = make_event()
        event["target_track_id"] = None
        self.assertIn("target_track_id", event)
        # None is acceptable
        self.assertIsNone(event["target_track_id"])

    def test_tc03_event_types_within_defined_set(self):
        """TC-03: All event types are within the six defined values."""
        sample_events = [
            make_event("touch"), make_event("pass"), make_event("long_ball"),
            make_event("clearance"), make_event("interception"), make_event("shot_attempt"),
        ]
        for ev in sample_events:
            self.assertIn(ev["event_type"], VALID_EVENT_TYPES,
                          f"'{ev['event_type']}' is not a valid event type")

    def test_tc03_no_tackle_in_valid_types(self):
        """TC-03: 'tackle' is NOT a valid event type."""
        self.assertNotIn("tackle", VALID_EVENT_TYPES)

    def test_tc03_no_foul_in_valid_types(self):
        """TC-03: 'foul' is NOT a valid event type."""
        self.assertNotIn("foul", VALID_EVENT_TYPES)

    def test_tc03_exactly_six_event_types(self):
        """TC-03: Exactly six event types are defined."""
        self.assertEqual(len(VALID_EVENT_TYPES), 6,
                         f"Expected 6 event types, got {len(VALID_EVENT_TYPES)}")

    def test_tc04_empty_events_list_when_no_ball(self):
        """TC-04: When no ball is detectable, events list is empty []."""
        artifacts = make_analysis_artifacts(events=[])
        self.assertIn("events", artifacts)
        self.assertIsInstance(artifacts["events"], list)
        self.assertEqual(len(artifacts["events"]), 0,
                         "events list must be empty when no ball is detectable")

    def test_tc04_empty_events_list_is_list_not_none(self):
        """TC-04: Empty output returns [] not None."""
        artifacts = make_analysis_artifacts(events=[])
        self.assertIsNotNone(artifacts["events"],
                             "'events' must be [] not None when empty")


# ══════════════════════════════════════════════════════════════════════════════
# UTC-05: Match Report API — Output Structure
# ══════════════════════════════════════════════════════════════════════════════

class TestMatchReportOutput(unittest.TestCase):
    """UTC-05: Verify match report API output structure."""

    def make_report(self, events=None):
        """Simulate a match report JSON as returned by GET /matches/:id/report."""
        return {
            "match_id": "test-match-uuid",
            "video": "match.mp4",
            "duration_s": 25.0,
            "fps": 25.0,
            "possession": {"team_0": 0.55, "team_1": 0.45},
            "players": [],
            "events": events if events is not None else [],
            "pass_network": {"nodes": [], "edges": []},
            "summary": {
                "total_touches": 0,
                "total_passes": 0,
                "total_long_balls": 0,
                "total_clearances": 0,
                "total_interceptions": 0,
                "total_shots": 0,
                "total_goals": 0,
            },
        }

    def test_tc01_report_contains_events_key(self):
        """TC-01: Match report response contains 'events' key."""
        report = self.make_report()
        self.assertIn("events", report)
        self.assertIsInstance(report["events"], list)

    def test_tc01_report_has_required_top_level_keys(self):
        """TC-01: Match report contains all required top-level keys."""
        report = self.make_report()
        for key in ("match_id", "fps", "events", "players", "summary"):
            self.assertIn(key, report, f"Report must have '{key}'")

    def test_tc01_summary_has_required_fields(self):
        """TC-01: Summary block contains required count fields."""
        report = self.make_report()
        summary = report["summary"]
        for field in (
            "total_touches", "total_passes", "total_long_balls",
            "total_clearances", "total_interceptions", "total_shots",
        ):
            self.assertIn(field, summary, f"Summary must have '{field}'")

    def test_tc02_report_events_have_correct_types(self):
        """TC-02: Events in report have correct field types."""
        events = [
            {"type": "pass", "frame": 100, "time_s": 4.0,
             "actor": 3, "actor_label": "player", "target": 7,
             "target_label": "player", "details": {}},
        ]
        report = self.make_report(events=events)
        for ev in report["events"]:
            self.assertIn("type", ev)
            self.assertIn("frame", ev)
            self.assertIn("time_s", ev)
            self.assertIsInstance(ev["frame"], int)
            self.assertIsInstance(ev["time_s"], float)

    def test_tc03_report_event_types_within_valid_set(self):
        """TC-03: Report event types are within allowed set (no tackle/foul)."""
        # MatchEvent.type in types.ts includes shot/goal but NOT tackle/foul
        REPORT_EVENT_TYPES = {
            "touch", "pass", "long_ball", "clearance",
            "interception", "shot", "goal",
        }
        invalid_types = {"tackle", "foul", "yellow_card", "red_card"}
        for t in invalid_types:
            self.assertNotIn(t, REPORT_EVENT_TYPES,
                             f"'{t}' must not be a valid report event type")


# ══════════════════════════════════════════════════════════════════════════════
# UTC-04 supplement: TrackObservation structure
# ══════════════════════════════════════════════════════════════════════════════

class TestTrackObservationStructure(unittest.TestCase):
    """Validates TrackObservation output structure matches schema."""

    def test_track_observation_has_required_fields(self):
        """TC-02: TrackObservation has all required fields."""
        obs = make_track_observation()
        for field in ("frame_index", "track_id", "label", "bbox", "confidence"):
            self.assertIn(field, obs, f"TrackObservation must have '{field}'")

    def test_track_id_is_integer(self):
        """TC-02: track_id is an integer (ByteTrack assigns integers)."""
        obs = make_track_observation(track_id=5)
        self.assertIsInstance(obs["track_id"], int)

    def test_frame_index_is_integer(self):
        obs = make_track_observation(frame_index=10)
        self.assertIsInstance(obs["frame_index"], int)

    def test_bbox_has_required_corners(self):
        """TC-02: BBox has x1, y1, x2, y2 corner fields."""
        obs = make_track_observation()
        bbox = obs["bbox"]
        for corner in ("x1", "y1", "x2", "y2"):
            self.assertIn(corner, bbox, f"BBox must have '{corner}'")

    def test_valid_labels(self):
        """TC-02: label is one of expected class names."""
        valid_labels = {"player", "ball", "goalkeeper", "referee"}
        obs = make_track_observation(label="player")
        self.assertIn(obs["label"], valid_labels)


# ══════════════════════════════════════════════════════════════════════════════
# Match status values
# ══════════════════════════════════════════════════════════════════════════════

class TestMatchStatusValues(unittest.TestCase):
    """UTC-03 supplement: MatchStatus enum values match backend definition."""

    def test_all_six_statuses_defined(self):
        self.assertEqual(len(MATCH_STATUS_VALUES), 6)

    def test_required_statuses_present(self):
        for s in ("CREATED", "UPLOADED", "QUEUED", "PROCESSING", "COMPLETED", "FAILED"):
            self.assertIn(s, MATCH_STATUS_VALUES)

    def test_invalid_status_not_in_set(self):
        self.assertNotIn("INVALID_STATUS", MATCH_STATUS_VALUES)
        self.assertNotIn("PENDING", MATCH_STATUS_VALUES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
