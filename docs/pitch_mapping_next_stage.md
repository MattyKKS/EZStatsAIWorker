# Pitch Mapping Next Stage

This stage prepares the project for later homography work.

It does not run pitch keypoint detection yet.
It creates a safe manual template first.

## Goal

The goal is to start the spatial-analysis path without changing the V1 tracking baseline.

This stage creates:

- a reference video frame
- a pitch mapping template

These will later be used for:

- homography
- pitch projection
- distance and speed in real-world space
- heatmaps and average positions

## Command

```powershell
ez-worker prepare-pitch-mapping --run-dir outputs/20260314_175555
```

## Output

This command creates a folder:

- `pitch_mapping/`

inside the run folder.

Inside it:

- `reference_frame.jpg`
- `pitch_mapping_template.json`

## What To Do With The Template

Open the reference frame and fill in the visible image pixel coordinates for the listed pitch points.

For now this is a manual prep step.
Later it can be replaced or assisted by pitch keypoint detection.

For a midfield broadcast frame, the best first points are usually:

- `top_halfway_touchline`
- `bottom_halfway_touchline`
- `center_spot`
- `center_circle_left`
- `center_circle_right`

You only need at least 4 visible points to continue.

## After Filling The Template

Once at least 4 points have real `image_xy_px` values, run:

```powershell
ez-worker apply-pitch-mapping --run-dir outputs/20260314_175555
```

This creates:

- `tracks_pitch.json`
- `pitch_mapping/homography_result.json`

## Tutorial References Used

This stage uses ideas from both tutorial references:

### Tutorial 1

- supports later pitch keypoint and homography work
- supports the move from image coordinates to pitch coordinates

### Tutorial 2

- explicitly mentions perspective transform and later real-world metrics
- supports building toward distance and speed in football space

## Proposal Alignment

This stage stays inside the proposal stack:

- `OpenCV homography`
- later pitch keypoints
- later spatial analytics

It does not introduce a different stack.
