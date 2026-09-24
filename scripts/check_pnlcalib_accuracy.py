"""Accuracy check: draw the model's pitch back onto the frame and look at it.

Coverage said a camera solution was produced. It did not say the solution was
right. This projects the known pitch geometry through the estimated camera and
overlays it on the source frame: if the drawn lines sit on the real painted lines,
the calibration is correct; if they float, it is confidently wrong.

This is the check that cannot be faked by a plausible-looking number, and it is
the one the recovery plan demands before calibration feeds events.

    python scripts/check_pnlcalib_accuracy.py --video data/raw/08fd33_4.mp4 --frames 360
    python scripts/check_pnlcalib_accuracy.py --video data/raw/leo_messi_30pass.mp4 \
        --frames 5691 5711 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "PnLCalib"))
sys.path.insert(0, str(ROOT / "scripts"))

from test_pnlcalib_coverage import calibrate, load_models   # noqa: E402
from utils.utils_calib import FramebyFrameCalib             # noqa: E402

# Pitch line segments in metres, origin at a corner (PnLCalib's convention).
LINES_M = [
    [[0., 54.16], [16.5, 54.16]], [[16.5, 13.84], [16.5, 54.16]],
    [[16.5, 13.84], [0., 13.84]], [[88.5, 54.16], [105., 54.16]],
    [[88.5, 13.84], [88.5, 54.16]], [[88.5, 13.84], [105., 13.84]],
    [[52.5, 0.], [52.5, 68.]], [[0., 68.], [105., 68.]],
    [[0., 0.], [0., 68.]], [[105., 0.], [105., 68.]],
    [[0., 0.], [105., 0.]], [[0., 43.16], [5.5, 43.16]],
    [[5.5, 43.16], [5.5, 24.84]], [[5.5, 24.84], [0., 24.84]],
    [[99.5, 43.16], [105., 43.16]], [[99.5, 43.16], [99.5, 24.84]],
    [[99.5, 24.84], [105., 24.84]],
]


def projection_matrix(params: dict) -> np.ndarray:
    cam = params["cam_params"]
    q = np.array([[cam["x_focal_length"], 0, cam["principal_point"][0]],
                  [0, cam["y_focal_length"], cam["principal_point"][1]],
                  [0, 0, 1]])
    it = np.eye(4)[:-1]
    it[:, -1] = -np.array(cam["position_meters"])
    return q @ (np.array(cam["rotation_matrix"]) @ it)


def ground_homography(p: np.ndarray) -> np.ndarray:
    """Field ground plane (z=0) -> image. Columns 0,1,3 of the 3x4 projection."""
    return p[:, [0, 1, 3]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--kp-threshold", type=float, default=0.3434)
    ap.add_argument("--line-threshold", type=float, default=0.7867)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    out_dir = Path(a.out_dir) if a.out_dir else ROOT / "outputs" / "calib_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    model, model_l = load_models(a.device)
    stem = Path(a.video).stem

    for idx in a.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"frame {idx}: unreadable")
            continue
        cam = FramebyFrameCalib(iwidth=W, iheight=H, denormalize=True)
        params, n_kp, n_ln = calibrate(frame, cam, model, model_l, a.device,
                                       a.kp_threshold, a.line_threshold, False)
        if params is None:
            print(f"frame {idx}: no calibration")
            continue

        p = projection_matrix(params)
        canvas = frame.copy()
        drawn = 0
        for (x1, y1), (x2, y2) in LINES_M:
            # PnLCalib's world origin is the pitch centre.
            a1 = p @ np.array([x1 - 105 / 2, y1 - 68 / 2, 0., 1.])
            a2 = p @ np.array([x2 - 105 / 2, y2 - 68 / 2, 0., 1.])
            if abs(a1[-1]) < 1e-6 or abs(a2[-1]) < 1e-6:
                continue
            a1, a2 = a1 / a1[-1], a2 / a2[-1]
            pt1, pt2 = (int(a1[0]), int(a1[1])), (int(a2[0]), int(a2[1]))
            if max(abs(pt1[0]), abs(pt2[0])) > 5 * W or max(abs(pt1[1]), abs(pt2[1])) > 5 * H:
                continue
            cv2.line(canvas, pt1, pt2, (0, 0, 255), 3)
            drawn += 1

        # Centre circle, to test curvature as well as straight lines.
        circle = [(52.5 + 9.15 * np.cos(t) - 105 / 2, 34 + 9.15 * np.sin(t) - 68 / 2)
                  for t in np.linspace(0, 2 * np.pi, 60)]
        pts = []
        for cx, cy in circle:
            v = p @ np.array([cx, cy, 0., 1.])
            if abs(v[-1]) > 1e-6:
                v = v / v[-1]
                if abs(v[0]) < 5 * W and abs(v[1]) < 5 * H:
                    pts.append([int(v[0]), int(v[1])])
        if len(pts) > 2:
            cv2.polylines(canvas, [np.array(pts)], False, (0, 255, 255), 3)

        # Project the GOAL FRAME in 3D: posts from the ground to 2.44 m, plus the
        # crossbar. This is the decisive check for goal detection. If the drawn goal
        # lands on the real goal, then we can test "is the ball inside the goal mouth"
        # directly IN THE IMAGE, instead of projecting the ball onto the ground plane -
        # which is what made an airborne ball appear past the line without crossing it.
        for gx in (0.0, 105.0):
            mouth = [(gx, 30.34, 0.0), (gx, 30.34, 2.44), (gx, 37.66, 2.44), (gx, 37.66, 0.0)]
            proj = []
            for wx, wy, wz in mouth:
                v = p @ np.array([wx - 105 / 2, wy - 68 / 2, -wz, 1.])
                if abs(v[-1]) < 1e-6:
                    proj = []
                    break
                v = v / v[-1]
                if abs(v[0]) > 5 * W or abs(v[1]) > 5 * H:
                    proj = []
                    break
                proj.append([int(v[0]), int(v[1])])
            if len(proj) == 4:
                cv2.polylines(canvas, [np.array(proj)], True, (255, 0, 255), 4)
                print(f"   goal mouth x={gx:.0f}m drawn at {proj}")

        hg = ground_homography(p)
        cv2.putText(canvas, f"frame {idx}  t={idx/fps:.2f}s  kp={n_kp} lines={n_ln} segs={drawn}",
                    (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        path = out_dir / f"{stem}_{idx:06d}_overlay.jpg"
        cv2.imwrite(str(path), canvas)
        print(f"frame {idx}: kp={n_kp} lines={n_ln} segments drawn={drawn} -> {path}")
        print(f"   ground homography det={np.linalg.det(hg):.4g}")
    cap.release()


if __name__ == "__main__":
    main()
