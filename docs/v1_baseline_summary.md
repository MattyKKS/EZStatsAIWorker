# V1 Baseline Summary

This document explains the current working baseline in simple words.

Use this file when:

- a new person joins the project
- someone wants to understand the worker quickly
- a future change breaks the pipeline and the team needs to fall back to the last stable version

## What V1 Baseline Means

V1 baseline is the first version that is good enough to keep as a reference.

It does these things reasonably well:

- reads a football video
- detects players and the ball
- tracks players
- keeps the ball more stable than before
- creates simple events and player stats
- exports JSON files
- can render a processed video

It is not perfect.

Known weak points:

- very fast ball movement can still cause short misses
- goalkeeper and referee can sometimes be confused
- some player tracks can disappear for short periods

Even with those weaknesses, this version is stable enough to keep as the fallback baseline.

## Main Idea Of The Pipeline

The pipeline works like this:

1. Read one input football video.
2. Run YOLO detection and ByteTrack-style tracking.
3. Detect the ball in a separate high-resolution pass.
4. Clean noisy tracks after inference.
5. Build simple events and player stats.
6. Save JSON outputs and, if requested, a processed video.

## Main Files And What They Do

### Core control files

- `src/ez_worker/cli.py`
  - This is the command line entrypoint.
  - It reads the user command and builds the config for a run.

- `src/ez_worker/config.py`
  - This stores the pipeline settings.
  - Examples: frame step, confidence thresholds, ball cleanup settings.

- `src/ez_worker/pipeline.py`
  - This is the main orchestrator.
  - It calls the provider, runs cleanup, creates events and stats, and writes outputs.

- `src/ez_worker/schemas.py`
  - This defines the main data shapes.
  - Examples: video metadata, boxes, track observations, events, stats.

### Inference files

- `src/ez_worker/providers/ultralytics_provider.py`
  - This is the real detection/tracking provider.
  - It runs the trained YOLO model.
  - It tracks players.
  - It runs a separate ball-only pass.
  - It merges ball results in a conservative way so the current good behavior is preserved.

- `src/ez_worker/providers/mock.py`
  - This is only for dry runs.
  - It lets the pipeline work without real computer vision.

### Track cleanup and analytics

- `src/ez_worker/postprocess/tracks.py`
  - This cleans noisy detections after inference.
  - It limits players per frame.
  - It keeps one ball per frame.
  - It cleans bad ball jumps.
  - It reconnects short gaps.
  - It merges short player track fragments.

- `src/ez_worker/analytics/events.py`
  - This creates simple football events from the cleaned tracks.
  - Current examples: touches, passes, shot attempts.

- `src/ez_worker/analytics/stats.py`
  - This turns cleaned tracks and events into player-level summary stats.

### Input/output helper files

- `src/ez_worker/io/video.py`
  - Reads basic video information like FPS, frame count, width, and height.

- `src/ez_worker/io/render.py`
  - Draws the final cleaned tracks and events onto the processed video.
  - This is important because the rendered video now matches the cleaned outputs better.

- `src/ez_worker/io/crops.py`
  - Exports player crops for later appearance-based work.

- `src/ez_worker/outputs/writer.py`
  - Saves the final JSON files to the output folder.

### Training and dataset helper files

- `src/ez_worker/train.py`
  - Trains the football detector from Roboflow data.

- `src/ez_worker/dataset_utils.py`
  - Prepares the local Roboflow dataset YAML.

- `src/ez_worker/data_init.py`
  - Creates the expected data folder layout.

## How The Files Work Together

This is the normal flow:

1. `cli.py` reads the command.
2. `config.py` stores the settings for that run.
3. `pipeline.py` loads the video and picks the provider.
4. `ultralytics_provider.py` creates raw tracks.
5. `tracks.py` cleans and stabilizes those tracks.
6. `events.py` creates simple football events.
7. `stats.py` creates player stats.
8. `render.py` draws the cleaned result video if rendering is enabled.
9. `writer.py` saves the JSON outputs.

## Key Decisions Made In V1

These are the important design choices for this baseline:

### 1. Keep the proposal-aligned stack

We stayed with:

- Python
- YOLO
- OpenCV
- FFmpeg-compatible video flow
- ByteTrack-style tracking

This keeps the worker aligned with the proposal draft.

### 2. Keep player tracking simple and stable

We did not try to solve every player identity problem yet.

Instead, we:

- cap the number of players kept
- merge short player fragments
- interpolate short missing gaps

This gives a stable enough baseline without overcomplicating the worker.

### 3. Handle the ball separately

The ball is much harder than players because it is small and moves fast.

So the worker uses:

- a separate high-resolution ball pass
- one-ball-per-frame logic
- jump cleanup
- conservative ball fallback when the dedicated pass misses

This was one of the most important V1 decisions.

### 4. Render the cleaned result, not the raw detector view

Earlier, the output video could show raw detections that did not match the cleaned JSON outputs.

In V1 baseline, the rendered video is based on the cleaned tracks instead.

That makes debugging and demos easier.

## Current Output Files

Each run creates a folder inside `outputs/`.

Typical files:

- `video_meta.json`
- `tracks.json`
- `events.json`
- `player_stats.json`
- `summary.json`
- `processed_video.mp4` if rendering is enabled

## Recommended Baseline Command

Use the current working command shape as the V1 baseline reference:

```powershell
ez-worker analyze --video data/raw/08fd33_4.mp4 --provider ultralytics --model-name artifacts/training/roboflow_detector_v1_light/weights/best.pt --tracker-config configs/bytetrack_football.yaml --render-video --frame-step 2 --ball-detection-imgsz 1280 --detection-confidence 0.22 --detection-iou 0.45 --min-player-confidence 0.24 --min-ball-confidence 0.06 --min-track-length 3 --max-players-per-frame 22 --max-unique-players 22 --min-player-track-frames 18 --max-track-gap-frames 18 --max-track-merge-distance-px 75 --max-interpolation-gap-frames 10 --max-ball-jump-px 90 --ball-reset-gap-frames 20 --ball-reset-confidence 0.45 --possession-distance-threshold-px 65 --possession-min-consecutive-frames 2
```

## What To Do If A Future Change Breaks Things

If a later experiment makes the pipeline worse:

1. Read this file first.
2. Compare the changed files against this baseline.
3. Re-run the baseline command.
4. Use the outputs from V1 as the reference.
5. Only keep new changes if they clearly improve the result.

## Simple Summary

V1 baseline is the first stable football worker version.

It is not perfect, but it is understandable, proposal-aligned, and good enough to act as the team fallback point.
