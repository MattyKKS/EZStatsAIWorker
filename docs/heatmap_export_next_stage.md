# Heatmap Export Next Stage

This stage builds heatmap and formation-friendly exports from `tracks_pitch.json`.

It is still a post-run step only.
It does not change tracking, clustering, team assignment, or homography.

## Goal

The goal is to create data that can later feed:

- team heatmaps
- player heatmaps
- average-position views
- formation-like visual layouts

## Command

```powershell
ez-worker build-heatmap-export --run-dir outputs/20260314_175555
```

## Output

This command writes:

- `heatmap_export.json`

inside the run folder.

## What It Contains

- team heatmap grids
- player heatmap grids
- formation-friendly average points per track

## Why This Is Safe

This stage only reads projected pitch tracks and writes a new export file.

If it looks weak:

- keep the current pipeline
- ignore the export
- fall back to the earlier spatial outputs

## Tutorial References Used

This stage follows both tutorial references:

### Tutorial 1

- supports later heatmaps and tactical-style outputs

### Tutorial 2

- supports perspective-transform outputs becoming football-space analytics

## Proposal Alignment

This remains inside the proposal direction:

- homography
- heatmaps
- average positions
