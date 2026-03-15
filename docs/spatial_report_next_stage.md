# Spatial Report Next Stage

This stage builds simple spatial summaries from `tracks_pitch.json`.

It is a post-run step only.
It does not change the tracker, clustering, or homography result.

## Goal

The goal is to make the first pitch-projected outputs easier to inspect and use.

This stage adds:

- team average positions
- player average positions
- simple team occupancy grids
- ball average position

## Command

```powershell
ez-worker build-spatial-report --run-dir outputs/20260314_175555
```

## Output

This command writes:

- `spatial_report.json`

inside the run folder.

## Why This Is Safe

This stage only reads:

- `tracks_pitch.json`

and writes a new summary file.

If the result looks weak:

- keep V1 tracking
- keep the current homography result
- ignore the spatial report file

## Tutorial References Used

This stage follows both tutorial references:

### Tutorial 1

- supports later average positions and richer football analytics after tracking

### Tutorial 2

- supports perspective-transform use for football-space summaries
- supports later distance, speed, and spatial metrics

## Proposal Alignment

This stays inside the proposal path:

- `OpenCV homography`
- later pitch analytics
- later heatmaps and average positions
