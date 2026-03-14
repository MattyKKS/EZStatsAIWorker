# EZ Stats AI Worker

AI-worker-first football analytics project. The goal is to run the full offline pipeline on a laptop or cloud notebook before any frontend/backend integration:

- input a match video
- detect and track players/ball
- infer simple events
- produce statistics
- export processed assets and JSON outputs

## Current status

This repository now contains the first scaffold:

- CLI entrypoint
- project structure
- config and typed schemas
- pipeline orchestration
- mock provider for dry runs
- event/stat generation from track data
- output writers
- dataset/setup documentation
- work log in `docs/WORK_LOG.md`

The repository now includes:

- a mock provider for dry runs
- a first `ultralytics` provider for real baseline detection/tracking
- a proposal-aligned Roboflow-first next-stage scaffold

## Recommended Python version

Use Python `3.11` or `3.12` for CV/ML compatibility.

On this machine:

- `python` points to `3.13.5`
- `py -3.12` is available and should be used for this project

## Beginner-friendly roadmap

### Phase 0

Set up tooling and sample data.

### Phase 1

Run the pipeline end to end with mock outputs so the folder structure, schemas, and export format are stable.

### Phase 2

Add real detection and tracking:

- player detector
- ball detector
- tracker
- rendered overlay video

### Phase 3

Add first useful football events and stats:

- touches
- carries
- passes
- shots
- possession estimates
- distance and speed summaries

### Phase 4

Add stronger football analytics:

- team assignment
- pitch projection
- heatmaps
- average positions
- x/y trajectory features

## Quick start

1. Create a Python 3.12 virtual environment.
2. Install dependencies from `pyproject.toml`.
3. Put a sample match video in `data/raw/`.
4. Run:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e .[dev,cv]
ez-worker analyze --video data/raw/sample.mp4 --provider mock
```

Outputs are written under `outputs/<run_id>/`.

For real baseline inference:

```bash
pip install -e .[ml]
ez-worker analyze --video data/raw/sample.mp4 --provider ultralytics --render-video
```

If the baseline is noisy, start by filtering short or low-confidence tracks before moving to football-specific training:

```bash
ez-worker analyze --video data/raw/sample.mp4 --provider ultralytics --render-video --min-player-confidence 0.45 --min-track-length 6
```

For the proposal-aligned next stage:

```bash
ez-worker init-data
```

Then put Roboflow detector data under `data/datasets/roboflow/`.

Prepare the Roboflow dataset YAML for local training:

```bash
ez-worker prepare-roboflow --dataset-dir data/datasets/roboflow/detector
```

Train the proposal-aligned football detector:

```bash
ez-worker train-detector --dataset-dir data/datasets/roboflow/detector --model-name yolov8m.pt --epochs 80 --image-size 1280 --batch-size 8
```

Run inference with a trained football-specific detector:

```bash
ez-worker analyze --video data/raw/sample.mp4 --provider ultralytics --model-name artifacts/training/roboflow_detector_v1_light/weights/best.pt --render-video --frame-step 3
```

Export player crops for the next `SigLIP + UMAP + KMeans` stage:

```bash
ez-worker analyze --video data/raw/sample.mp4 --provider ultralytics --model-name artifacts/training/roboflow_detector_v1_light/weights/best.pt --render-video --export-player-crops
```

## Repo Notes

- Change tracking and component responsibilities are documented in `docs/WORK_LOG.md`.
- The implementation should remain aligned with the proposal draft stack.

Force NVIDIA GPU 0 explicitly:

```bash
ez-worker train-detector --dataset-dir data/datasets/roboflow/detector --model-name yolov8m.pt --epochs 60 --image-size 1280 --batch-size 6 --device 0
```

## Repository layout

```text
src/ez_worker/
  analytics/
  io/
  providers/
  outputs/
  cli.py
  config.py
  pipeline.py
  schemas.py
docs/
  datasets.md
```

## What you need to download manually

Read `docs/datasets.md`. Short version:

- yes, you need to manually obtain at least one sample football video
- SoccerNet video data requires their access process
- Kaggle downloads can be scripted after account/API setup
- Roboflow exports can be scripted after you choose the dataset/project
