"""Detect goals by calibrating frames and testing ball-inside-goal-mouth in the image.

Runs PnLCalib over a frame range, projects each goal mouth into the image, and
asks whether an OBSERVED ball position falls inside it.

    python scripts/detect_goals.py --run-dir "G:/My Drive/ezstats/outputs/20260910_170402" \
        --video data/raw/leo_messi_30pass.mp4 --start 92 --end 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor" / "PnLCalib"))

from check_pnlcalib_accuracy import projection_matrix          # noqa: E402
from test_pnlcalib_coverage import calibrate, load_models      # noqa: E402
from utils.utils_calib import FramebyFrameCalib                # noqa: E402

from ez_worker.analytics.goal_mouth import (                   # noqa: E402
    find_goal_mouth_candidates,
    goal_mouth_quad,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, required=True, help="seconds")
    ap.add_argument("--end", type=float, required=True, help="seconds")
    ap.add_argument("--stride", type=int, default=2, help="calibrate every Nth frame")
    ap.add_argument("--save-hits", default=None, help="write an overlay for each hit")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    run = Path(a.run_dir)
    meta = json.loads((run / "video_meta.json").read_text(encoding="utf-8"))
    W, H, fps = meta["width"], meta["height"], meta["fps"]
    lo, hi = int(a.start * fps), int(a.end * fps)

    tracks = json.loads((run / "tracks.json").read_text(encoding="utf-8"))
    ball_xy: dict[int, tuple[float, float]] = {}
    for o in tracks:
        if o.get("label") != "ball":
            continue
        if o.get("interpolated") or o.get("synthetic"):
            continue                      # never let an invented ball score
        f = o["frame_index"]
        if lo <= f <= hi:
            b = o["bbox"]
            ball_xy[f] = ((b["x1"] + b["x2"]) / 2 * W, (b["y1"] + b["y2"]) / 2 * H)
    print(f"observed ball positions in {a.start}-{a.end}s: {len(ball_xy)}")
    if not ball_xy:
        sys.exit("no observed ball in that range")

    model, model_l = load_models(a.device)
    cap = cv2.VideoCapture(a.video)
    projections: dict[int, np.ndarray] = {}
    wanted = [f for f in sorted(ball_xy) if (f - lo) % a.stride == 0]
    print(f"calibrating {len(wanted)} frames ...", flush=True)
    for f in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        cam = FramebyFrameCalib(iwidth=W, iheight=H, denormalize=True)
        try:
            params, _, _ = calibrate(frame, cam, model, model_l, a.device,
                                     0.3434, 0.7867, False)
        except Exception:
            continue
        if params is not None:
            projections[f] = projection_matrix(params)
    print(f"calibrated {len(projections)}/{len(wanted)} = "
          f"{100*len(projections)/max(len(wanted),1):.1f}%")

    cands = find_goal_mouth_candidates(ball_xy, projections, fps)
    print(f"\ngoal candidates: {len(cands)}")
    for c in cands:
        print(f"  t={c.time_seconds:.2f}s side={c.side} frames_inside={c.frames_inside} "
              f"({c.first_frame}-{c.last_frame}) entered_from_outside={c.entered_from_outside}")

    # Per-frame trace, so a near miss is visible rather than silent.
    print("\nframe-by-frame (ball vs goal mouth):")
    for f in sorted(projections):
        if f not in ball_xy:
            continue
        bx, by = ball_xy[f]
        line = f"  f{f:6d} t={f/fps:6.2f}s ball=({bx:7.1f},{by:7.1f})"
        for side in ("left", "right"):
            quad = goal_mouth_quad(projections[f], side)
            if quad is None:
                continue
            d = cv2.pointPolygonTest(quad.astype(np.float32), (float(bx), float(by)), True)
            line += f"  {side}: {'INSIDE' if d >= 0 else f'{-d:6.1f}px out'}"
        print(line)

    if a.save_hits and cands:
        out = Path(a.save_hits)
        out.mkdir(parents=True, exist_ok=True)
        for c in cands:
            for f in range(c.first_frame, c.last_frame + 1):
                if f not in projections or f not in ball_xy:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, frame = cap.read()
                if not ok:
                    continue
                quad = goal_mouth_quad(projections[f], c.side)
                if quad is not None:
                    cv2.polylines(frame, [quad.astype(int)], True, (255, 0, 255), 4)
                bx, by = ball_xy[f]
                cv2.circle(frame, (int(bx), int(by)), 14, (0, 255, 255), 3)
                cv2.putText(frame, f"GOAL CANDIDATE f{f} t={f/fps:.2f}s", (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imwrite(str(out / f"goal_{f:06d}.jpg"), frame)
        print(f"\noverlays -> {out}")
    cap.release()


if __name__ == "__main__":
    main()
