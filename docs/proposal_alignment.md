# Proposal Alignment

This repository should stay aligned with the technology choices in the current proposal draft.

## AI Worker Stack From Proposal

- `Python`
- `YOLO`
- `OpenCV`
- `FFmpeg`
- `ByteTrack`
- `SigLIP + UMAP + KMeans` for team classification
- `OpenCV homography` for pitch mapping

## What This Means For The Next Stage

The next stage is not more generic COCO-only inference.

It is:

1. Use a football-specific detector workflow, starting with Roboflow-friendly data and tutorial assets.
2. Keep `ByteTrack` as the main tracker.
3. Keep the current worker output contracts:
   - `tracks.json`
   - `events.json`
   - `player_stats.json`
   - `processed_video.mp4`
4. Add later modules in this order:
   - football-specific player/ball detection
   - stronger tracking stability
   - pitch keypoint detection
   - homography
   - team classification with `SigLIP + UMAP + KMeans`
   - richer event detection

## Recommended Immediate Focus

### Stage A

Roboflow-first dataset setup for:

- player detection
- ball detection
- optional referee class

### Stage B

Replace the generic baseline model with football-specific weights or a Roboflow-trained detector.

### Stage C

Only after detection/tracking are stable, move to:

- team classification
- spatial analytics
- event refinement

## Important Constraint

Do not let the worker drift into a stack that conflicts with the proposal.

For the AI worker, the proposal-aligned path is:

`football-specific YOLO detector -> ByteTrack -> outputs -> later team clustering + homography`
