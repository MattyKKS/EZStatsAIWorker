# Dataset and Video Setup

## What you should download manually

You should manually obtain:

- at least 1 full or partial football match video for local testing
- at least 1 labelled detection dataset for player/ball fine-tuning or validation
- optional event labels if you want supervised event detection later

## Recommended beginner path

### 1. Start with one local test video

Put a video here:

```text
data/raw/
```

Use one clear broadcast-style football video first. Do not start with many matches.

### 2. Use Roboflow for object detection datasets

Best use:

- player
- referee
- ball
- goalkeeper

Practical note:

- you usually export the dataset from the Roboflow project after logging in
- some datasets are public, but export still depends on project access and format choice

Official source:

- https://docs.roboflow.com/datasets/dataset-versions/exporting-data

### 3. Use Kaggle for additional football/computer-vision datasets

Best use:

- extra football images
- tabular event/stat datasets for validation
- notebooks and baseline experiments

Practical note:

- Kaggle downloads can be scripted with the Kaggle API after you configure credentials
- first-time setup usually requires logging in and creating an API token

Official source:

- https://github.com/Kaggle/kaggle-api

### 4. Use SoccerNet for benchmark football data

Best use:

- event labels
- benchmark tasks
- research-aligned evaluation

Practical note:

- SoccerNet is valuable, but video access is not instant
- some assets require their access workflow, including NDA/password-gated download steps for video data

Official source:

- https://www.soccer-net.org/downloads

## Recommendation for today

Today you should manually prepare these:

1. One sample football video in `data/raw/`
2. One detection dataset source you trust

The fastest realistic combination is:

1. your own sample match video or a permitted sample clip
2. one Roboflow football detection dataset
3. optionally SoccerNet later for evaluation/event work

## What can be scripted later

After setup, we can add scripts for:

- validating directory layout
- loading local videos
- ingesting exported annotation files
- training/evaluation dataset manifests
- Kaggle API pulls
- Roboflow API pulls where supported

## Recommended local folder layout

```text
data/
  raw/
  interim/
  processed/
  datasets/
    roboflow/
    kaggle/
    soccernet/
```
