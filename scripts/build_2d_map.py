"""Top-down 2D pitch map from real camera calibration.

The football-manager view: every player as a dot at their true position on a
105x68 m pitch, from a single moving camera.

This was impossible until calibration worked. The old 32-keypoint pose model
yielded a usable homography on 2.6% of Messi frames, so there were no coordinates
to draw. With PnLCalib that is 73.7% overall and 100% through the goal window.

Method, which matches the published single-camera approach (Huang et al., Appl.
Sci. 2022, 12, 7473 - NPSPT): take the BOTTOM-CENTRE of each player box as the
ground contact point, push it through the camera to the pitch plane (z=0), and
plot it. Their Section 3.2 does the same thing and calls the result world
coordinates.

Off-field rejection also follows that paper. They note a hand-drawn mask removes
crowd and substitutes but "erases the information of the players standing on the
edge of the field" too. Working in world coordinates avoids that trade-off
entirely: a point is on the pitch if its metres are inside the rectangle, so a
player on the touchline is kept and a spectator behind it is not.

    python scripts/build_2d_map.py --run-dir <run> --video <mp4> \
        --start 92 --end 99 --out outputs/map2d
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

from check_pnlcalib_accuracy import projection_matrix        # noqa: E402
from test_pnlcalib_coverage import calibrate, load_models    # noqa: E402
from utils.utils_calib import FramebyFrameCalib              # noqa: E402

PITCH_L, PITCH_W = 105.0, 68.0
MARGIN_M = 2.0          # tolerance for a player standing just off the line
TEAM_BGR = {0: (147, 20, 255), 1: (255, 191, 0)}   # pink / cyan, as in the video


def draw_pitch(scale: int = 10, pad: int = 30) -> np.ndarray:
    w, h = int(PITCH_L * scale) + 2 * pad, int(PITCH_W * scale) + 2 * pad
    img = np.full((h, w, 3), (40, 120, 40), np.uint8)
    white = (255, 255, 255)

    def pt(x, y):
        return int(pad + x * scale), int(pad + y * scale)

    cv2.rectangle(img, pt(0, 0), pt(PITCH_L, PITCH_W), white, 2)
    cv2.line(img, pt(PITCH_L / 2, 0), pt(PITCH_L / 2, PITCH_W), white, 2)
    cv2.circle(img, pt(PITCH_L / 2, PITCH_W / 2), int(9.15 * scale), white, 2)
    for x0, x1 in ((0, 16.5), (PITCH_L - 16.5, PITCH_L)):
        cv2.rectangle(img, pt(x0, 13.84), pt(x1, 54.16), white, 2)
    for x0, x1 in ((0, 5.5), (PITCH_L - 5.5, PITCH_L)):
        cv2.rectangle(img, pt(x0, 24.84), pt(x1, 43.16), white, 2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default="outputs/map2d")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    run = Path(a.run_dir)
    meta = json.loads((run / "video_meta.json").read_text(encoding="utf-8"))
    W, H, fps = meta["width"], meta["height"], meta["fps"]
    lo, hi = int(a.start * fps), int(a.end * fps)

    tp = run / "tracks_with_teams.json"
    if not tp.exists():
        tp = run / "tracks.json"
    tracks = json.loads(tp.read_text(encoding="utf-8"))

    by_frame: dict[int, list] = {}
    for o in tracks:
        f = o.get("frame_index")
        if f is None or not (lo <= f <= hi):
            continue
        by_frame.setdefault(f, []).append(o)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, model_l = load_models(a.device)
    cap = cv2.VideoCapture(a.video)

    scale = 10
    writer = None
    stats = {"frames": 0, "calibrated": 0, "on_pitch": 0, "off_pitch": 0}

    for f in sorted(by_frame):
        if (f - lo) % a.stride:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        stats["frames"] += 1
        cam = FramebyFrameCalib(iwidth=W, iheight=H, denormalize=True)
        try:
            params, _, _ = calibrate(frame, cam, model, model_l, a.device,
                                     0.3434, 0.7867, False)
        except Exception:
            params = None
        if params is None:
            continue
        stats["calibrated"] += 1
        p = projection_matrix(params)
        ground = p[:, [0, 1, 3]]                    # z=0 plane -> image
        try:
            inv = np.linalg.inv(ground)             # image -> pitch plane
        except np.linalg.LinAlgError:
            continue

        pitch = draw_pitch(scale)
        pad = 30
        for o in by_frame[f]:
            label = o.get("label")
            if label == "ball":
                bx = (o["bbox"]["x1"] + o["bbox"]["x2"]) / 2 * W
                by = (o["bbox"]["y1"] + o["bbox"]["y2"]) / 2 * H
                colour, radius = (255, 255, 255), 5
            elif label in ("player", "goalkeeper"):
                bx = (o["bbox"]["x1"] + o["bbox"]["x2"]) / 2 * W
                by = o["bbox"]["y2"] * H            # feet = ground contact
                colour = TEAM_BGR.get(o.get("team_id"), (200, 200, 200))
                radius = 7
            else:
                continue

            v = inv @ np.array([bx, by, 1.0])
            if abs(v[-1]) < 1e-9:
                continue
            # PnLCalib's origin is the pitch centre; shift to a corner origin.
            mx = float(v[0] / v[-1]) + PITCH_L / 2
            my = float(v[1] / v[-1]) + PITCH_W / 2

            if not (-MARGIN_M <= mx <= PITCH_L + MARGIN_M
                    and -MARGIN_M <= my <= PITCH_W + MARGIN_M):
                stats["off_pitch"] += 1             # crowd, bench, advertising
                continue
            stats["on_pitch"] += 1

            cx, cy = int(pad + mx * scale), int(pad + my * scale)
            cv2.circle(pitch, (cx, cy), radius, colour, -1)
            cv2.circle(pitch, (cx, cy), radius, (0, 0, 0), 1)
            if label != "ball" and o.get("track_id") is not None:
                cv2.putText(pitch, str(o["track_id"]), (cx + 8, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

        cv2.putText(pitch, f"t={f/fps:.2f}s  frame {f}", (pad, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if writer is None:
            h2, w2 = pitch.shape[:2]
            writer = cv2.VideoWriter(str(out / "map2d.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     max(1.0, fps / a.stride), (w2, h2))
        writer.write(pitch)
        cv2.imwrite(str(out / f"map_{f:06d}.jpg"), pitch)

    cap.release()
    if writer:
        writer.release()
    print(f"frames processed : {stats['frames']}")
    print(f"calibrated       : {stats['calibrated']} "
          f"({100*stats['calibrated']/max(stats['frames'],1):.1f}%)")
    print(f"points on pitch  : {stats['on_pitch']}")
    print(f"rejected off-pitch: {stats['off_pitch']} (crowd / bench / advertising)")
    print(f"output -> {out}")


if __name__ == "__main__":
    main()
