# EZStats AI Worker — Progress Report
*Kaung | May 2026*

**Goal:** Offline AI pipeline for football match analysis — affordable alternative to tools like Hudl for small clubs and university teams.

**Stack:** YOLOv8 · ByteTrack · SigLIP · UMAP/KMeans · OpenCV · Python

---

## What's Done

- Full end-to-end pipeline in a single command (`run_full_pipeline.ps1`)
- Player, GK, referee, and ball detection + multi-object tracking
- Automatic team classification (no manual labelling required)
- Pitch homography for radar minimap and top-down spatial view
- Per-player stats: possession %, passes, touches, shots, distance
- Two output videos: annotated match video (`stats_video.mp4`) and top-down Voronoi + ball trail (`spatial_video.mp4`)
- Event detection based on ball velocity (Vidal-Codina et al. 2022)
- SoccerNet LSTM event spotter for high-level events (goals, corners, fouls)

## Current Limitations

- Airborne ball maps to wrong pitch position (ground-plane homography limitation)
- Player IDs occasionally switch on occlusion — fragments per-player stats
- Event detection (passes/shots) is heuristic-based, not ground-truth validated
- ~20 min to process 30s of footage on CPU — needs GPU for full matches
- Single fixed camera only — pan/zoom breaks homography

## Before June 22

- Validate event detection against annotated clips
- Test on additional match footage
- Connect pipeline output to dashboard frontend
- Reduce processing time (GPU or higher frame_step)
