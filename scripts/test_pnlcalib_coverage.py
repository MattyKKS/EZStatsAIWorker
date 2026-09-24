"""Go/no-go: does PnLCalib calibrate frames our 32-keypoint pose model cannot?

Our current pitch model is a YOLOv8-pose net that must regress all 32 pitch
landmarks as one object. Measured, it finds a median of 8-9 of them, and
FieldProjector accepts a homography on 2.6% of sampled Messi frames and 38% of
08fd33_4 frames - and 0 of ~180 frames in the annotated Messi goal window, because
that is exactly when the camera tightens on the box and landmarks leave frame.

PnLCalib detects keypoints AND field-line extremities, so long lines can carry a
calibration when corners are gone. This script measures whether that is true on
OUR clips, before any integration work is done.

It only measures coverage - whether a camera solution is produced at all. Coverage
is necessary, not sufficient; positional accuracy is a separate check.

    python scripts/test_pnlcalib_coverage.py --video data/raw/08fd33_4.mp4 --samples 30
    python scripts/test_pnlcalib_coverage.py --video data/raw/leo_messi_30pass.mp4 \
        --samples 40 --window 94 97
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as f
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "PnLCalib"
sys.path.insert(0, str(VENDOR))

from model.cls_hrnet import get_cls_net                      # noqa: E402
from model.cls_hrnet_l import get_cls_net as get_cls_net_l   # noqa: E402
from utils.utils_calib import FramebyFrameCalib              # noqa: E402
from utils.utils_heatmap import (                            # noqa: E402
    complete_keypoints,
    coords_to_dict,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
)


def load_models(device: str):
    cfg = yaml.safe_load((VENDOR / "config/hrnetv2_w48.yaml").read_text())
    cfg_l = yaml.safe_load((VENDOR / "config/hrnetv2_w48_l.yaml").read_text())
    m = get_cls_net(cfg)
    m.load_state_dict(torch.load(ROOT / "artifacts/calibration/SV_kp", map_location=device))
    m.to(device).eval()
    ml = get_cls_net_l(cfg_l)
    ml.load_state_dict(torch.load(ROOT / "artifacts/calibration/SV_lines", map_location=device))
    ml.to(device).eval()
    return m, ml


def calibrate(frame_bgr, cam, model, model_l, device, kp_thr, line_thr, pnl_refine):
    """One frame -> (camera params or None, n_keypoints, n_lines)."""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    t = f.to_tensor(img).float().unsqueeze(0)
    if t.size()[-1] != 960:
        t = T.Resize((540, 960))(t)
    t = t.to(device)
    _, _, h, w = t.size()
    with torch.no_grad():
        hm = model(t)
        hm_l = model_l(t)
    kp_c = get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1, :, :])
    ln_c = get_keypoints_from_heatmap_batch_maxpool_l(hm_l[:, :-1, :, :])
    kp_d = coords_to_dict(kp_c, threshold=kp_thr)
    ln_d = coords_to_dict(ln_c, threshold=line_thr)
    kp_d, ln_d = complete_keypoints(kp_d[0], ln_d[0], w=w, h=h, normalize=True)
    cam.update(kp_d, ln_d)
    return cam.heuristic_voting(refine_lines=pnl_refine), len(kp_d), len(ln_d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--samples", type=int, default=30, help="frames spread across the clip")
    ap.add_argument("--window", type=float, nargs=2, default=None,
                    help="also sample this [start end] second range densely")
    ap.add_argument("--window-samples", type=int, default=10)
    ap.add_argument("--kp-threshold", type=float, default=0.3434)
    ap.add_argument("--line-threshold", type=float, default=0.7867)
    ap.add_argument("--pnl-refine", action="store_true")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {a.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{Path(a.video).name}: {total} frames, {fps:.2f} fps, {W}x{H}, device={a.device}")

    targets = sorted(set(np.linspace(0, total - 1, a.samples).astype(int).tolist()))
    tagged = [(int(t), "spread") for t in targets]
    if a.window:
        lo, hi = int(a.window[0] * fps), int(a.window[1] * fps)
        wf = np.linspace(lo, min(hi, total - 1), a.window_samples).astype(int)
        tagged += [(int(t), "window") for t in wf]
    tagged.sort()

    print(f"loading PnLCalib models ...", flush=True)
    model, model_l = load_models(a.device)

    stats = {"spread": [0, 0], "window": [0, 0]}
    kp_counts, times = [], []
    for idx, tag in tagged:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        cam = FramebyFrameCalib(iwidth=W, iheight=H, denormalize=True)
        t0 = time.time()
        try:
            params, n_kp, n_ln = calibrate(frame, cam, model, model_l, a.device,
                                           a.kp_threshold, a.line_threshold, a.pnl_refine)
        except Exception as exc:
            print(f"  frame {idx:6d} [{tag}] ERROR {type(exc).__name__}: {exc}")
            stats[tag][1] += 1
            continue
        times.append(time.time() - t0)
        stats[tag][1] += 1
        ok_cal = params is not None
        if ok_cal:
            stats[tag][0] += 1
        kp_counts.append(n_kp)
        print(f"  frame {idx:6d} t={idx/fps:6.2f}s [{tag:6}] "
              f"kp={n_kp:2d} lines={n_ln:2d} -> {'CALIBRATED' if ok_cal else 'none'}",
              flush=True)
    cap.release()

    print("\n================ RESULT ================")
    for tag, (good, n) in stats.items():
        if n:
            print(f"  {tag:7}: {good}/{n} calibrated = {100*good/n:.1f}%")
    if kp_counts:
        print(f"  keypoints/frame: median {int(np.median(kp_counts))} "
              f"(our pose model: 8-9 of 32)")
    if times:
        print(f"  {np.mean(times):.2f}s per frame on {a.device}")
    print("  baseline to beat - FieldProjector valid frames:")
    print("     08fd33_4 38.0%   |   leo_messi 2.6%   |   messi goal window 0%")


if __name__ == "__main__":
    main()
