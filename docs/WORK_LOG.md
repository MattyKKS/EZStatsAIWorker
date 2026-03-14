# Work Log

## 2026-03-10

### Added

- Initial AI worker scaffold:
  - CLI
  - pipeline orchestration
  - schemas
  - output writers
  - mock provider
- Real Ultralytics inference provider.
- Rendered video output saved into each inference run folder.
- Roboflow dataset preparation and detector training commands.
- Proposal alignment documentation:
  - `docs/proposal_alignment.md`
  - `docs/roboflow_next_stage.md`
- Track post-processing:
  - per-frame player limiting
  - short-track filtering
  - fragmented player tracklet merging
  - global player-track pruning
  - one-ball-per-frame selection
  - short-gap interpolation for players and ball
  - ball outlier cleanup based on jump distance
- Ultralytics inference now follows the tutorial structure more closely:
  - players/referees/goalkeepers tracked
  - ball detected separately per frame
- Revised the previous ball-handling change to avoid dual streaming after it appeared to cause unstable runtime behavior.
- Added a dedicated sequential ball-only pass to better match the tutorial's ball-handling approach.
- Player crop export for the next appearance-model stage.
- Final video rendering now uses the cleaned post-processed tracks instead of raw YOLO overlays.
- Ball cleanup now supports trajectory-based re-acquisition and frame dropping for ambiguous multi-ball detections.
- Ball source merging now keeps the dedicated ball pass as primary but falls back to tracked-stream ball detections on frames where the dedicated pass misses.

### Changed

- Training now supports explicit GPU selection with `--device`.
- Training output path now resolves inside the repo for future runs.
- Inference now maps labels from the actual trained model instead of assuming COCO class IDs.
- Roboflow `data.yaml` was corrected for the current repo layout.
- Event detection now selects the ball by label/confidence instead of assuming a fixed ball track ID.
- Added a football-tuned ByteTrack config under `configs/bytetrack_football.yaml`.
- Ball cleanup now follows the tutorial direction more closely by dropping implausible jump detections.
- Inference provider no longer relies on raw joint tracking of the ball.
- Current implementation uses one tracked stream and then normalizes ball handling per frame.
- Event generation now requires stable short possession before emitting touch/pass events.
- Ball rendering and cleanup are now closer to the tutorial flow:
  - detect ball separately at high resolution
  - drop multi-ball frames
  - keep the candidate that best matches the recent trajectory
  - allow controlled high-confidence re-acquisition after longer gaps
- Ball continuity now uses a conservative fallback path instead of fully discarding tracked-stream ball hints.

### Removed

- Nothing removed yet.

### What Each Main Part Does

- `src/ez_worker/cli.py`
  - Entry point for analyze, dataset prep, and training commands.
- `src/ez_worker/pipeline.py`
  - Runs the full inference pipeline from video to outputs.
- `src/ez_worker/providers/mock.py`
  - Dry-run provider for pipeline testing without real CV.
- `src/ez_worker/providers/ultralytics_provider.py`
  - Real detector/tracker inference using YOLO + ByteTrack-compatible tracking flow.
- `src/ez_worker/postprocess/tracks.py`
  - Cleans and merges noisy player tracklets after inference.
- `src/ez_worker/analytics/events.py`
  - Builds the first rule-based events from tracks.
- `src/ez_worker/analytics/stats.py`
  - Builds player-level stats from cleaned tracks and events.
- `src/ez_worker/train.py`
  - Trains the football-specific detector from Roboflow data.
- `src/ez_worker/dataset_utils.py`
  - Prepares Roboflow dataset YAML for local training.
- `src/ez_worker/io/crops.py`
  - Exports player crops from tracked detections for later appearance embedding and clustering.
