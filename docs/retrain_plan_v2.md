# V2 Retrain Plan

Last updated: `2026-03-15 00:10` Bangkok time

This is the retrain plan for the current AI-worker quality phase.

Use this when:

- goalkeepers are not being detected as `goalkeeper`
- referees are not being detected as `referee`
- sideline players still disappear too often
- the worker code is already reasonably stable, but the detector is still making the wrong class decisions

## Goal

Improve the detector so the worker becomes more reliable on:

1. `goalkeeper`
2. `referee`
3. `player` near sidelines
4. partially visible players
5. wide broadcast football views

This plan keeps the project aligned with the proposal stack:

- `YOLO`
- `ByteTrack`
- `OpenCV`
- `FFmpeg`
- `SigLIP + UMAP + KMeans`

## What We Already Know

From the current run:

- ball handling is usable enough for now
- player tracking is usable but still drops some edge players
- the detector is not outputting meaningful `goalkeeper` or `referee` labels on the real clip

That means:

- more cleanup tuning alone will not solve goalkeeper/referee separation
- the detector now needs better training data

## Main Retrain Strategy

Do not retrain with random extra football images.

Retrain with **hard examples that match the failure cases**:

1. goalkeeper visible but predicted as plain player
2. referee visible but predicted as plain player
3. goalkeeper and referee with similar shirt color
4. players near the touchline
5. partially cropped players near screen edges
6. wide midfield broadcast frames

## Data Priority

### Highest priority

Add more labelled frames for:

- `goalkeeper`
- `referee`
- `player` near left/right sidelines

### Medium priority

Add more labelled frames for:

- partially visible players
- zoomed-out midfield views
- transitions after goal kicks or long clearances

### Low priority for now

- more central easy player frames
- more ball-only examples

The ball is not the main blocker right now.

## Recommended Data Sources

### Source 1: hard frames from your own clip

Best value right now:

- extract frames from your current 30-second benchmark clip
- label the exact failure cases

This is the fastest way to make the detector care about your real deployment case.

### Source 2: extra Roboflow football dataset exports

Good for:

- more goalkeeper examples
- more referee examples
- more broadcast football frames

### Source 3: optional manual hard-example folder

Create a small internal set of:

- goalkeeper vs referee confusion frames
- sideline-player frames
- partially visible player frames

This does not need to be huge.
Even 50 to 150 well-chosen hard frames can help more than hundreds of easy ones.

## Suggested Local Layout

Keep the current Roboflow dataset as the base.
Add hard examples into a second folder so we can track what changed.

```text
data/
  datasets/
    roboflow/
      detector/
    retrain_v2_hard_examples/
      images/
      labels/
```

If you later merge them into one YOLO dataset, keep a note of the source split.

## Label Rules

Use exactly these classes:

1. `ball`
2. `goalkeeper`
3. `player`
4. `referee`

Do not rename classes.
Do not collapse goalkeeper/referee into player.

## Training Plan

### Stage A: quick hard-example retrain

Purpose:

- confirm the model starts emitting `goalkeeper` and `referee` at all

Recommended starting command:

```powershell
ez-worker train-detector --dataset-dir data/datasets/roboflow/detector --model-name yolov8n.pt --epochs 40 --image-size 960 --batch-size 4 --device 0 --run-name retrain_v2_hardpass_a
```

Use this stage first if:

- you are still iterating quickly
- you want faster turnaround on the laptop

### Stage B: stronger retrain

Purpose:

- produce the real next detector candidate after the hard examples are added

Recommended command:

```powershell
ez-worker train-detector --dataset-dir data/datasets/roboflow/detector --model-name yolov8m.pt --epochs 60 --image-size 960 --batch-size 4 --device 0 --run-name retrain_v2_main
```

If VRAM fails, reduce:

- `batch-size` from `4` to `2`

Do not jump back to `1280` immediately.
We care more about stable training than theoretical max resolution.

## Benchmark Rule

After each retrain candidate, run the same benchmark clip:

```text
data/raw/08fd33_4.mp4
```

This clip is your current quality check.

The retrained model is only better if it improves this same clip on:

1. visible goalkeeper boxes
2. visible referee boxes
3. fewer lost sideline players
4. stable or better ball handling
5. equal or better event quality

## Success Criteria

Call the retrain successful if:

- processed video visibly shows goalkeepers with boxes on most of their visible frames
- at least some referee frames are classified as `referee`
- sideline player drops are noticeably reduced
- player count stays realistic
- event count does not collapse

## Failure Criteria

Do not accept the retrain if:

- goalkeeper/referee are still almost always plain `player`
- sideline players get worse
- ball handling becomes worse
- track count explodes
- event quality drops sharply

## Practical Next Step

The next real work should be:

1. collect hard frames from the benchmark clip
2. add more goalkeeper/referee/touchline examples
3. run Stage A
4. benchmark on the same 30-second clip
5. only then move to Stage B

## What Not To Do

Do not:

- retrain on random easy football frames
- change the class names
- change the stack outside the proposal
- assume more epochs alone will fix class confusion

The real fix is better class-balanced hard examples.
