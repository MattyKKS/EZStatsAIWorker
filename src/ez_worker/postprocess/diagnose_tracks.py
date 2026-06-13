"""Diagnostic-only: classify surviving player tracks in a run dir.

Read-only. Prints a per-track table with frame span, foot-position spread,
homography-projected pitch position (best-frame H), referee flag, and match
report stats. Helps decide which of the 33 survivors are real vs spurious.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

PITCH_L = 12000.0  # cm
PITCH_W = 7000.0


def load(run_dir: Path):
    rd = Path(run_dir)
    tracks = json.loads((rd / "tracks_with_teams.json").read_text(encoding="utf-8"))
    vm = json.loads((rd / "video_meta.json").read_text(encoding="utf-8"))
    report = json.loads((rd / "match_report.json").read_text(encoding="utf-8"))
    kp = json.loads((rd / "pitch_keypoints.json").read_text(encoding="utf-8"))
    refs = set(json.loads((rd / "referee_track_ids.json").read_text(encoding="utf-8")))
    return tracks, vm, report, kp, refs


def build_homography(kp):
    """pixel(full-res) -> pitch cm homography from the best-frame keypoints."""
    px = kp["keypoints"]
    cm = kp["keypoint_pitch_xy_cm"]
    src, dst = [], []
    for k in px:
        src.append(px[k])
        dst.append(cm[k])
    src = np.array(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H, src


def project(H, x, y):
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def main(run_dir: str):
    rd = Path(run_dir)
    tracks, vm, report, kp, refs = load(rd)
    W, H = vm["width"], vm["height"]
    Hmat, src = build_homography(kp)

    # convex hull of keypoints in pixels = region where H is trustworthy
    hull = cv2.convexHull(src.astype(np.float32))

    by_id = defaultdict(list)
    for t in tracks:
        if t["label"] == "player":
            by_id[t["track_id"]].append(t)

    rep_players = {p["track_id"]: p for p in report["players"]}

    rows = []
    for tid, obs in by_id.items():
        if tid not in rep_players:
            continue  # not a survivor in the report
        obs.sort(key=lambda o: o["frame_index"])
        feet = [((o["bbox"]["x1"] + o["bbox"]["x2"]) / 2 * W, o["bbox"]["y2"] * H) for o in obs]
        fx = np.array([f[0] for f in feet]); fy = np.array([f[1] for f in feet])
        mfx, mfy = float(np.median(fx)), float(np.median(fy))
        spread = float(math.hypot(fx.std(), fy.std()))
        pcx, pcy = project(Hmat, mfx, mfy)
        # inside trusted hull?
        in_hull = cv2.pointPolygonTest(hull, (mfx, mfy), True)  # >0 inside, dist px
        # off pitch test (pitch is 0..L, 0..W cm) with 300cm margin
        off = not (-300 <= pcx <= PITCH_L + 300 and -300 <= pcy <= PITCH_W + 300)
        teams = Counter(o.get("team_id") for o in obs)
        p = rep_players[tid]
        rows.append({
            "tid": tid, "team": p["team_id"], "n": len(obs),
            "start": obs[0]["frame_index"], "end": obs[-1]["frame_index"],
            "mfx": mfx, "mfy": mfy, "spread": spread,
            "pcx": pcx, "pcy": pcy, "off": off, "hull": in_hull,
            "ref": tid in refs,
            "touches": p.get("touches", 0), "passes": p.get("passes", 0),
            "shots": p.get("shots", 0), "dist": p.get("distance_px", 0),
            "teamvote": dict(teams),
        })

    rows.sort(key=lambda r: (r["team"], -r["n"]))
    print(f"{'tid':>4} {'tm':>2} {'n':>4} {'span':>9} {'foot(px)':>13} "
          f"{'spread':>6} {'pitch(cm)':>15} {'off':>4} {'hull':>6} {'ref':>4} "
          f"{'t/p/s':>8} {'dist':>7}")
    for r in rows:
        span = f"{r['start']}-{r['end']}"
        foot = f"{r['mfx']:.0f},{r['mfy']:.0f}"
        pitch = f"{r['pcx']:.0f},{r['pcy']:.0f}"
        tps = f"{r['touches']}/{r['passes']}/{r['shots']}"
        print(f"{r['tid']:>4} {r['team']:>2} {r['n']:>4} {span:>9} {foot:>13} "
              f"{r['spread']:>6.0f} {pitch:>15} {str(r['off']):>4} "
              f"{r['hull']:>6.0f} {str(r['ref']):>4} {tps:>8} {r['dist']:>7.0f}")

    n_off = sum(r["off"] for r in rows)
    print(f"\ntotal survivors: {len(rows)}   teams: {Counter(r['team'] for r in rows)}")
    print(f"off-pitch flagged: {n_off}   refs flagged: {sum(r['ref'] for r in rows)}")
    print(f"outside trusted hull (neg dist): {sum(r['hull']<0 for r in rows)}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/20260608_224414")
