# Roboflow Next Stage

## Goal

Move from the current generic baseline to a football-specific training/inference path.

## What You Need To Do Manually Next

1. Choose or export a football detection dataset from Roboflow.
2. Download the dataset in a YOLO-compatible format.
3. Place it under:

```text
data/datasets/roboflow/
```

4. Keep your current sample test video in:

```text
data/raw/
```

## What To Prioritize First

First dataset priority:

- players
- ball

Optional for later:

- referee
- goalkeeper
- pitch keypoints

## Why Roboflow First

Because the current baseline already proved:

- the worker runs
- the output contract works
- the render path works

The weak point is model/domain quality, so the next leverage point is football-specific data.

## Suggested Folder Layout

```text
data/
  raw/
  interim/
  processed/
  datasets/
    roboflow/
      detector/
      keypoints/
    kaggle/
    soccernet/
```

## What Kaggle Is For Later

Use Kaggle later for:

- comparison datasets
- research experiments
- notebook baselines
- supplementary validation data

Roboflow is the first priority for detection.

## Repo Commands For This Stage

Prepare the dataset YAML:

```bash
ez-worker prepare-roboflow --dataset-dir data/datasets/roboflow/detector
```

Start training:

```bash
ez-worker train-detector --dataset-dir data/datasets/roboflow/detector --model-name yolov8m.pt --epochs 80 --image-size 1280 --batch-size 8
```
