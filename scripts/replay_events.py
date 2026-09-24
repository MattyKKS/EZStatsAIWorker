"""Recompute experimental events in a NEW folder without detection or training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write_event_tables(destination, events, evidence, old, fps):
    columns = ["time_seconds", "event_type", "actor", "target", "status", "reason"]
    timeline = [dict(time_seconds=e.time_seconds, event_type=e.event_type,
                     actor=e.actor_track_id, target=e.target_track_id,
                     status="candidate", reason=e.details.get("method", "")) for e in events]
    review = [dict(time_seconds=round(t["reception_frame"] / fps, 3),
                   event_type="unresolved_transfer", actor=t["actor"], target=t["target"],
                   status="not_counted", reason=t["reason"])
              for t in evidence["transfers"] if t["reason"]]
    review += [dict(time_seconds=e.get("time_seconds", e["frame_index"] / fps),
                    event_type=e["event_type"], actor=e.get("actor_track_id"),
                    target=e.get("target_track_id"), status="legacy_unverified",
                    reason="Old label retained for review, not counted by contact engine")
               for e in old if e["event_type"] in ("shot", "shot_attempt", "goal", "clearance", "long_ball")]
    for name, rows in (("event_timeline.csv", timeline), ("event_review.csv", review)):
        with (destination / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: r["time_seconds"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, help="Override original source path for this machine")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--event-engine", choices=("contacts_v1", "contacts_v2"), default="contacts_v1")
    parser.add_argument("--export-pitch", action="store_true", help="Export quality-gated field positions from saved landmarks; needs OpenCV and sports")
    parser.add_argument("--copy-to", type=Path, help="Copy completed result into this Drive directory")
    args = parser.parse_args(argv)
    print("Event replay starting; no GPU inference or model loading.", flush=True)
    from ez_worker.analytics.contacts import ContactConfig, detect_contact_events
    from ez_worker.analytics.stats import build_track_stats
    from ez_worker.outputs.writer import write_artifacts
    from ez_worker.schemas import AnalysisArtifacts, TrackObservation, VideoMeta

    source = args.run_dir.resolve()
    payload = (source / "tracks.json").read_bytes()
    tracks = [TrackObservation(**o) for o in json.loads(payload)]
    video = VideoMeta(**json.loads((source / "video_meta.json").read_text(encoding="utf-8")))
    if args.video:
        video.path = args.video.resolve()
    if not args.skip_video and not video.path.is_file():
        parser.error("Source video unavailable. Supply --video or --skip-video.")
    destination = (args.output_dir or ROOT / "outputs" /
                   (source.name + "_contacts_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    print(f"Read {len(tracks)} observations. Computing contact evidence...", flush=True)
    events, evidence, possession = detect_contact_events(
        tracks, video, config=ContactConfig(allow_dribble_follow=args.event_engine == "contacts_v2"))
    if args.export_pitch:
        from sports.configs.soccer import SoccerPitchConfiguration
        from ez_worker.spatial.field_positions import export_field_positions
        print("Projecting saved pitch landmarks; no pitch-model inference...", flush=True)
        pitch = json.loads((source / "pitch_keypoints_per_frame.json").read_text(encoding="utf-8"))
        positions = export_field_positions(tracks, video, pitch, SoccerPitchConfiguration().vertices,
                                           cut_frames=evidence["cut_frames"])
        (destination / "field_positions.json").write_text(json.dumps(positions, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Field projection coverage (players): {positions['player_projection_coverage']}", flush=True)
    write_artifacts(AnalysisArtifacts(video=video, tracks=tracks, events=events,
                    stats=build_track_stats(tracks, events, video)), destination)
    held = Counter(t for t in possession.values() if t is not None)
    report_path = destination / "match_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["possession"] = {str(t): round(100 * n / sum(held.values()), 1) for t, n in held.items()}
    report["possession_known_frames"] = sum(held.values())
    report["possession_coverage_percent"] = round(100 * sum(held.values()) / max(1, video.frame_count), 2)
    report["possession_basis"] = "Observed confirmed ground-contact frames only, not full-match possession"
    report["event_detection"] = {"experimental": True, "unsupported": evidence["unsupported_events"]}
    # Unsupported is not zero. Keep missing capabilities out of statistical claims.
    for name in ("total_shots", "total_goals", "total_interceptions", "total_touches"):
        report["summary"][name] = None
    for player in report["players"]:
        player["shots"] = player["touches"] = None
    report["metric_notes"] = {
        "total_passes": "Ground-pass candidates; not measured accuracy or all pass attempts",
        "distance_px": "Image displacement includes camera motion; not physical distance",
        "pass_completion_percent": "Unavailable: unsuccessful pass attempts are not classified",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in ("pitch_keypoints.json", "pitch_keypoints_per_frame.json", "referee_track_ids.json", "identity_decisions.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    if (source / "run_manifest.json").is_file():
        shutil.copy2(source / "run_manifest.json", destination / "source_run_manifest.json")
    shutil.copy2(destination / "tracks.json", destination / "tracks_with_teams.json")
    shutil.copy2(report_path, destination / "match_report_merged.json")
    counts = dict(Counter(e.event_type for e in events))
    old = json.loads((source / "events.json").read_text(encoding="utf-8")) if (source / "events.json").exists() else []
    write_event_tables(destination, events, evidence, old, video.fps)
    quality = dict(event_engine=evidence["method"], experimental=True,
                   event_code_sha256=hashlib.sha256((ROOT / "src/ez_worker/analytics/contacts.py").read_bytes()).hexdigest(),
                   source_run=str(source), source_tracks_sha256=hashlib.sha256(payload).hexdigest(),
                   original_events=dict(Counter(e["event_type"] for e in old)), events=counts,
                   detection_changed=False, identities_changed=False, accuracy=None,
                   possession_coverage_percent=report["possession_coverage_percent"],
                   unsupported_events=evidence["unsupported_events"],
                   accuracy_note="Counts are not accuracy. Review event_evidence.json against the video.")
    for name, data in (("event_evidence.json", evidence), ("quality_summary.json", quality),
                       ("possession_by_frame.json", possession)):
        (destination / name).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(quality, indent=2), flush=True)
    if not args.skip_video:
        print("Rendering saved tracks and revised events...", flush=True)
        from ez_worker.io.render_from_tracks import render_from_tracks
        render_from_tracks(destination)
    if args.copy_to:
        copied = args.copy_to.resolve() / destination.name
        if copied == destination:
            parser.error("--copy-to must differ from the local output directory")
        print(f"Copying completed replay to {copied}...", flush=True)
        shutil.copytree(destination, copied)
    print(f"Event replay complete: {destination}", flush=True)
    return destination


if __name__ == "__main__":
    main()
