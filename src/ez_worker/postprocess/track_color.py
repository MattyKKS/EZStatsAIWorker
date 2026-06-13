"""Diagnostic-only: objective per-track torso colour from crops.

Samples the torso band (central x, upper-mid y) of every crop for a track and
reports median HSV + brightness. Used to separate referees (dark/low-sat),
goalkeepers (distinct saturated hue) and off-pitch blur from real players —
independent of the noisy SigLIP team_id.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import cv2


def torso_hsv(crop_dir: Path):
    vals = []
    files = sorted(os.listdir(crop_dir))
    for f in files:
        img = cv2.imread(str(crop_dir / f))
        if img is None:
            continue
        h, w = img.shape[:2]
        # torso band: central 50% width, 20-55% height (shirt, above shorts, away from grass)
        y0, y1 = int(0.20 * h), int(0.55 * h)
        x0, x1 = int(0.25 * w), int(0.75 * w)
        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        # drop green grass pixels (H ~35-85 in OpenCV 0-180, decent sat)
        hh, ss, vv = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        grass = (hh >= 30) & (hh <= 90) & (ss > 60)
        keep = hsv[~grass]
        if len(keep) < 10:
            keep = hsv
        vals.append(np.median(keep, axis=0))
    if not vals:
        return None
    return np.median(np.array(vals), axis=0)


def main(run_dir: str):
    rd = Path(run_dir)
    crops = rd / "player_crops"
    rep = json.loads((rd / "match_report.json").read_text(encoding="utf-8"))
    team = {p["track_id"]: p["team_id"] for p in rep["players"]}
    rows = []
    for d in sorted(os.listdir(crops)):
        tid = int(d.split("_")[1])
        hsv = torso_hsv(crops / d)
        if hsv is None:
            continue
        H, S, V = hsv
        rows.append((tid, team.get(tid, "?"), H, S, V))
    print(f"{'tid':>4} {'tm':>3} {'H':>5} {'S':>5} {'V':>5}  kit-guess")
    for tid, tm, H, S, V in rows:
        if V < 75 and S < 110:
            guess = "DARK(ref?)"
        elif S > 110 and (H < 12 or H > 168):
            guess = "RED(gk?)"
        elif V > 150 and S < 70:
            guess = "WHITE"
        elif 35 <= H <= 90 and S > 60:
            guess = "GREEN"
        else:
            guess = ""
        print(f"{tid:>4} {str(tm):>3} {H:>5.0f} {S:>5.0f} {V:>5.0f}  {guess}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/20260608_224414")
