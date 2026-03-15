# Formation Export Next Stage

This stage builds a simple tactical-layout export from `heatmap_export.json`.

It is still a post-run step only.
It does not change tracking, ball handling, team assignment, or homography.

## Goal

The goal is to create an easier tactical summary for later UI work and football analysis.

This stage adds:

- team centroids
- ordered player points
- simple visible line groupings
- a basic visible line shape per team

## Command

```powershell
ez-worker build-formation-export --run-dir outputs/20260314_175555
```

## Output

This command writes:

- `formation_export.json`

inside the run folder.

## What It Contains

- one entry per team
- team centroid positions
- players ordered from one side of the pitch to the other
- simple grouped lines for a formation-like layout

This is not a true tactical model yet.
It is a safe export that later product and analytics work can use.

## Why This Is Safe

This stage only reads:

- `heatmap_export.json`

and writes a new export file.

If it looks weak:

- keep the current pipeline
- ignore the export
- fall back to the earlier spatial outputs

## Tutorial References Used

This stage follows both tutorial references:

### Tutorial 1

- supports later tactical-style outputs after tracking and pitch mapping

### Tutorial 2

- supports football-space layouts after perspective transform

## Proposal Alignment

This remains inside the proposal direction:

- homography
- heatmaps
- average positions
- later football analytics
