"""Per-player distance, speed and heatmaps in real metres.

Everything here is an accumulation over the world coordinates the calibration now
produces, so none of it was possible while the old pitch model calibrated 2.6% of
frames. Pixel "distance" was never a physical quantity: it mixes player motion
with camera motion and with perspective, which is why the old reports carried a
note telling the reader not to treat it as running distance.

Honesty built into the output, because these numbers are the ones a client will
believe too readily:

  * Distance is PER TRACK, not per player. A player whose identity fragments into
    three tracks appears as three short runners. `coverage` and `track_seconds`
    are reported so a short track is visible as short.
  * Only calibrated frames contribute, so distance UNDERCOUNTS whenever coverage
    is below 100%. The shortfall is reported rather than silently scaled up.
  * Displacements implying more than `max_speed_ms` are dropped as tracking or
    calibration jitter rather than counted as sprinting.

    python scripts/build_player_metrics.py --run-dir <run> --video <mp4> \
        --start 0 --end 108 --stride 10 --out outputs/metrics
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor" / "PnLCalib"))

from build_2d_map import PITCH_L, PITCH_W, MARGIN_M, draw_pitch   # noqa: E402
from check_pnlcalib_accuracy import projection_matrix             # noqa: E402
from test_pnlcalib_coverage import calibrate, load_models         # noqa: E402
from utils.utils_calib import FramebyFrameCalib                   # noqa: E402

MAX_SPEED_MS = 10.0          # 36 km/h - at the human record; above this is jitter
SPEED_WINDOW_S = 1.0         # top speed is measured over a window, not one sample


def heatmap_image(points, scale: int = 8, pad: int = 20, blur: int = 41) -> np.ndarray:
    """Occupancy heat over the pitch, drawn on top of the pitch diagram."""
    base = draw_pitch(scale, pad)
    h, w = base.shape[:2]
    acc = np.zeros((h, w), np.float32)
    for mx, my in points:
        cx, cy = int(pad + mx * scale), int(pad + my * scale)
        if 0 <= cx < w and 0 <= cy < h:
            acc[cy, cx] += 1.0
    if acc.max() <= 0:
        return base
    acc = cv2.GaussianBlur(acc, (blur, blur), 0)
    acc = (acc / acc.max() * 255).astype(np.uint8)
    heat = cv2.applyColorMap(acc, cv2.COLORMAP_JET)
    mask = (acc > 12)[:, :, None]
    return np.where(mask, cv2.addWeighted(base, 0.35, heat, 0.65, 0), base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--min-track-seconds", type=float, default=3.0)
    ap.add_argument("--out", default="outputs/metrics")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    run = Path(a.run_dir)
    meta = json.loads((run / "video_meta.json").read_text(encoding="utf-8"))
    W, H, fps = meta["width"], meta["height"], meta["fps"]
    lo, hi = int(a.start * fps), int(min(a.end * fps, meta.get("frame_count", 1e9)))

    tp = run / "tracks_with_teams.json"
    if not tp.exists():
        tp = run / "tracks.json"
    by_frame: dict[int, list] = defaultdict(list)
    team_of: dict[int, int] = {}
    for o in json.loads(tp.read_text(encoding="utf-8")):
        f = o.get("frame_index")
        if f is None or not (lo <= f <= hi):
            continue
        by_frame[f].append(o)
        if o.get("label") != "ball" and o.get("team_id") is not None:
            team_of.setdefault(o["track_id"], o["team_id"])

    model, model_l = load_models(a.device)
    cap = cv2.VideoCapture(a.video)
    wanted = [f for f in sorted(by_frame) if (f - lo) % a.stride == 0]
    print(f"calibrating {len(wanted)} frames ...", flush=True)

    positions: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    team_points: dict[int, list[tuple[float, float]]] = defaultdict(list)
    n_cal = 0
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
            params = None
        if params is None:
            continue
        n_cal += 1
        try:
            inv = np.linalg.inv(projection_matrix(params)[:, [0, 1, 3]])
        except np.linalg.LinAlgError:
            continue
        for o in by_frame[f]:
            if o.get("label") not in ("player", "goalkeeper"):
                continue
            bx = (o["bbox"]["x1"] + o["bbox"]["x2"]) / 2 * W
            by = o["bbox"]["y2"] * H
            v = inv @ np.array([bx, by, 1.0])
            if abs(v[-1]) < 1e-9:
                continue
            mx = float(v[0] / v[-1]) + PITCH_L / 2
            my = float(v[1] / v[-1]) + PITCH_W / 2
            if not (-MARGIN_M <= mx <= PITCH_L + MARGIN_M
                    and -MARGIN_M <= my <= PITCH_W + MARGIN_M):
                continue
            tid = o.get("track_id")
            if tid is None:
                continue
            positions[tid].append((f, mx, my))
            if team_of.get(tid) is not None:
                team_points[team_of[tid]].append((mx, my))
    cap.release()

    coverage = 100.0 * n_cal / max(len(wanted), 1)
    print(f"calibrated {n_cal}/{len(wanted)} = {coverage:.1f}%")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, dropped = [], 0
    max_gap = a.stride * 3
    for tid, pts in sorted(positions.items()):
        pts.sort()
        if len(pts) < 2:
            continue
        seconds = (pts[-1][0] - pts[0][0]) / fps
        if seconds < a.min_track_seconds:
            continue
        dist = 0.0
        for (f0, x0, y0), (f1, x1, y1) in zip(pts, pts[1:]):
            df = f1 - f0
            if df <= 0 or df > max_gap:
                continue
            d = float(np.hypot(x1 - x0, y1 - y0))
            dt = df / fps
            if d / dt > MAX_SPEED_MS:
                dropped += 1
                continue
            dist += d

        # Top speed over a ~1s window, never a single sample. At stride 12 and
        # 60fps the gap between samples is 0.2s, so two metres of positional
        # noise alone reads as 36 km/h - which is how a first pass produced
        # 42 km/h "sprints" for players who were not sprinting. Averaging over a
        # second makes noise cancel instead of accumulate.
        speeds = []
        win = SPEED_WINDOW_S * fps
        j = 0
        for i in range(len(pts)):
            while pts[i][0] - pts[j][0] > win:
                j += 1
            if j < i and pts[i][0] - pts[j][0] >= win * 0.5:
                dt = (pts[i][0] - pts[j][0]) / fps
                d = float(np.hypot(pts[i][1] - pts[j][1], pts[i][2] - pts[j][2]))
                v = d / dt
                if v <= MAX_SPEED_MS:
                    speeds.append(v)
        rows.append({
            "track_id": tid,
            "team_id": team_of.get(tid),
            "distance_m": round(dist, 1),
            "top_speed_ms": round(max(speeds), 2) if speeds else None,
            "top_speed_kmh": round(max(speeds) * 3.6, 1) if speeds else None,
            "track_seconds": round(seconds, 1),
            "samples": len(pts),
        })
        cv2.imwrite(str(out / f"heatmap_track_{tid:04d}.jpg"),
                    heatmap_image([(x, y) for _, x, y in pts]))

    for team, pts in team_points.items():
        cv2.imwrite(str(out / f"heatmap_team_{team}.jpg"), heatmap_image(pts))

    report = {
        "source_run": str(run),
        "window_seconds": [a.start, min(a.end, meta.get("frame_count", 0) / fps)],
        "frames_sampled": len(wanted),
        "frames_calibrated": n_cal,
        "calibration_coverage_percent": round(coverage, 1),
        "units": "metres; speed in m/s and km/h",
        "players": sorted(rows, key=lambda r: -r["distance_m"]),
        "caveats": [
            "Distance is PER TRACK, not per player: a fragmented identity appears "
            "as several short runners. Check track_seconds before comparing players.",
            f"Only calibrated frames contribute, so distance UNDERCOUNTS whenever "
            f"coverage is below 100% (here {coverage:.1f}%). Values are not scaled up.",
            f"{dropped} displacement(s) implying more than {MAX_SPEED_MS} m/s were "
            "dropped as tracking or calibration jitter.",
            f"Top speed is a straight-line average over a {SPEED_WINDOW_S}s window, "
            "not a single sample. Single-sample speed is meaningless at this "
            "sampling rate: two metres of positional noise over 0.2s alone reads "
            "as 36 km/h. Averaging also UNDER-reports a genuine curved sprint.",
            f"Positions are sampled every {a.stride} frames, so distance along a "
            "curved run is slightly under-measured.",
        ],
    }
    (out / "player_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'track':>6} {'team':>5} {'dist_m':>8} {'top_kmh':>8} {'secs':>6}")
    for r in report["players"][:12]:
        print(f"{r['track_id']:>6} {str(r['team_id']):>5} {r['distance_m']:>8.1f} "
              f"{r['top_speed_kmh'] or 0:>8.1f} {r['track_seconds']:>6.1f}")
    print(f"\n{len(rows)} tracks, heatmaps + player_metrics.json -> {out}")


if __name__ == "__main__":
    main()
