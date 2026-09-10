#!/usr/bin/env python
"""Pipeline v2 - portable runner (Windows local AND Google Colab).

Same steps and flags as run_pipeline_v2.ps1, but cross-platform, so the Colab
notebook and the local machine run an identical pipeline instead of two
definitions that drift apart.

    python run_pipeline_v2.py --video data/raw/08fd33_4.mp4
    python run_pipeline_v2.py --video data/raw/BrightonGoal.mp4 --skip-video
    python run_pipeline_v2.py --video data/raw/leo_messi_30pass.mp4 --source-fps 60

Fixes wired in (see docs/product_plan.ipynb "Root Cause Analysis 2026-09-10"):
  RC-1 per-frame homography, RC-2 tracker config v3 (detector conf stays 0.20),
  RC-5 referee cap, RC-6 learned kit colours, real possession,
  and report_cleanup as a step rather than a manual afterthought.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
PLAYER_MODEL = "artifacts/training/player_detector_v2/weights/best.pt"
BALL_MODEL = "artifacts/ball/football-ball-detection.pt"
PITCH_MODEL = "artifacts/pitch/football-pitch-detectionV2.pt"
TRACKER = "configs/bytetrack_football_v3.yaml"


def _ez() -> list[str]:
    """`ez-worker` if it is on PATH, else the module form (Colab-safe)."""
    exe = shutil.which("ez-worker")
    return [exe] if exe else [sys.executable, "-m", "ez_worker.cli"]


def step(name: str, cmd: list[str], *, allow_failure: bool = False, capture: bool = False) -> str:
    """Run one pipeline step, streaming its output live.

    Output is echoed line by line as it arrives (not buffered to the end), because
    `analyze` and `render-stats-video` run for many minutes and a silent cell in
    Colab is indistinguishable from a hung one. `capture` additionally returns the
    text so the caller can parse the run directory out of it.
    """
    print(f"\n>>> [{name}] starting...", flush=True)
    t0 = time.time()
    lines: list[str] = []
    proc = subprocess.Popen(cmd, cwd=ROOT, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line, end="", flush=True)
        if capture:
            lines.append(line)
    proc.wait()
    out = "".join(lines)
    dt = time.time() - t0
    ok = proc.returncode == 0
    print(f"<<< [{name}] {'done' if ok else 'FAILED'} in {int(dt // 60):02d}:{dt % 60:05.2f}",
          flush=True)
    if not ok and not allow_failure:
        sys.exit(f"[{name}] failed with exit code {proc.returncode}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/raw/08fd33_4.mp4")
    ap.add_argument("--source-fps", type=int, default=25,
                    help="60 for the Messi clip - scales the frame-based tracking knobs")
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--kp-stride", type=int, default=10,
                    help="per-frame homography sampling interval (0.4 s at 25 fps)")
    ap.add_argument("--goal-frame", type=int, default=-1)
    ap.add_argument("--skip-video", action="store_true",
                    help="skip render-stats-video (~80%% of runtime) while tuning events")
    a = ap.parse_args()

    s = a.source_fps / 25.0
    gap, interp = int(20 * s), int(10 * s)
    reset, hold = int(20 * s), int(8 * s)
    mtl, mtf = max(1, int(2 * s)), max(1, int(5 * s))
    kp_stride = max(1, int(a.kp_stride * s))

    print(f"PIPELINE v2  video={a.video}  fps={a.source_fps} (scale {s:.2f})  "
          f"kp-stride={kp_stride}  skip_video={a.skip_video}")
    t_all = time.time()
    ez = _ez()

    out = step("analyze", ez + [
        "analyze", "--video", a.video, "--provider", "ultralytics",
        "--model-name", PLAYER_MODEL, "--ball-model-name", BALL_MODEL,
        "--tracker-config", TRACKER, "--render-video", "--frame-step", str(a.frame_step),
        "--ball-detection-imgsz", "1280", "--detection-confidence", "0.20",
        "--detection-iou", "0.45", "--min-player-confidence", "0.18",
        "--min-ball-confidence", "0.15", "--min-track-length", str(mtl),
        "--max-players-per-frame", "28", "--max-unique-players", "60",
        "--min-player-track-frames", str(mtf), "--max-track-gap-frames", str(gap),
        "--max-track-merge-distance-px", "85", "--max-interpolation-gap-frames", str(interp),
        "--max-ball-jump-px", "90", "--ball-reset-gap-frames", str(reset),
        "--ball-reset-confidence", "0.55", "--ball-hold-max-gap-frames", str(hold),
        "--ball-smoothing-alpha", "0.35", "--possession-distance-threshold-px", "120",
        "--possession-min-seconds", "0.15", "--auto-calibrate", "--export-player-crops",
    ], capture=True)

    m = re.search(r"Outputs written to:\s*(\S+)", out)
    if not m:
        sys.exit("Could not parse the run directory from analyze output.")
    run_dir = m.group(1).strip()
    print(f"Run dir: {run_dir}")

    step("prepare-appearance", ez + ["prepare-appearance", "--run-dir", run_dir])
    step("cluster-teams", ez + ["cluster-teams", "--run-dir", run_dir,
                                "--method", "siglip", "--cluster-count", "2"])
    step("apply-team-clusters", ez + ["apply-team-clusters", "--run-dir", run_dir])
    step("detect-pitch-keypoints", ez + [
        "detect-pitch-keypoints", "--run-dir", run_dir, "--model-path", PITCH_MODEL,
        "--per-frame-stride", str(kp_stride)], allow_failure=True)

    step("rerun-events", [sys.executable, "rerun_events.py", run_dir])
    step("apply-team-clusters (sync)", ez + ["apply-team-clusters", "--run-dir", run_dir])

    # Teams from jersey colour: ONE decision per track, overriding SigLIP. Must run
    # AFTER the sync above, which rewrites team_id from the SigLIP clusters and would
    # otherwise undo this.
    step("assign-teams-color", ez + ["assign-teams-color", "--run-dir", run_dir],
         allow_failure=True)

    cleanup = [sys.executable, "-m", "src.ez_worker.postprocess.report_cleanup", run_dir, "--apply"]
    if a.goal_frame >= 0:
        cleanup += ["--goal-frame", str(a.goal_frame)]
    step("report-cleanup", cleanup, allow_failure=True)

    # Render LAST: --from-tracks draws the pipeline's own tracks, so the final events
    # and team assignment must already be on disk.
    if a.skip_video:
        print("
--- render-stats-video SKIPPED (--skip-video) ---")
    else:
        step("render-stats-video", ez + ["render-stats-video", "--run-dir", run_dir,
                                         "--from-tracks"])

    dt = time.time() - t_all
    print("\n" + "=" * 48)
    print(f"PIPELINE v2 complete in {int(dt // 60):02d}:{dt % 60:05.2f}")
    print(f"  Video:       {a.video}")
    print(f"  Run dir:     {run_dir}")
    print(f"  Report:      {run_dir}/match_report_merged.json   <- serve this")
    print("=" * 48)


if __name__ == "__main__":
    main()
