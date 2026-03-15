# Analysis Ready Next Stage

This stage builds safer downstream files after role hints are available.

It is a post-run step only.
It does not change tracking, ball handling, team assignment, or role hints.

## Goal

The goal is to remove likely referee and unclear outlier tracks from the files that later analytics will use.

This gives a cleaner base for:

- event improvement
- team summaries
- later UI integration

## Command

```powershell
ez-worker build-analysis-ready --run-dir outputs/20260314_175555
```

## Output

This command writes:

- `analysis_ready_summary.json`
- `tracks_analysis_ready.json`
- `player_stats_analysis_ready.json`

inside the run folder.

## What It Does

- keeps normal players
- keeps goalkeeper candidates
- removes `referee_candidate`
- removes `outlier_candidate`

This is still conservative.
It is meant to reduce noise before later event work.

## Why This Is Safe

This stage only reads current outputs and writes new files.

If it looks weak:

- keep the earlier outputs
- ignore the analysis-ready files
- fall back to V1 or the earlier team outputs

## Proposal Alignment

This remains inside the proposal path:

- `YOLO`
- `ByteTrack`
- `SigLIP + UMAP + KMeans`
- later football analytics
