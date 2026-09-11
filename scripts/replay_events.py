"""Recompute experimental events in a NEW folder without detection or training."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, help="Override original source path for this machine")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--copy-to", type=Path, help="Copy completed result into this Drive directory")
    args = parser.parse_args(argv)
    print("Event replay starting; no GPU inference or model loading.", flush=True)
    from ez_worker.analytics.contacts import detect_contact_events
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
    events, evidence, possession = detect_contact_events(tracks, video)
    write_artifacts(AnalysisArtifacts(video=video, tracks=tracks, events=events,
                    stats=build_track_stats(tracks, events, video)), destination)
    held = Counter(t for t in possession.values() if t is not None)
    report_path = destination / "match_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["possession"] = {str(t): round(100 * n / sum(held.values()), 1) for t, n in held.items()}
    report["possession_known_frames"] = sum(held.values())
    report["event_detection"] = {"experimental": True, "unsupported": evidence["unsupported_events"]}
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
    quality = dict(event_engine="ground_contacts_v1", experimental=True,
                   event_code_sha256=hashlib.sha256((ROOT / "src/ez_worker/analytics/contacts.py").read_bytes()).hexdigest(),
                   source_run=str(source), source_tracks_sha256=hashlib.sha256(payload).hexdigest(),
                   original_events=dict(Counter(e["event_type"] for e in old)), events=counts,
                   detection_changed=False, identities_changed=False, accuracy=None,
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
