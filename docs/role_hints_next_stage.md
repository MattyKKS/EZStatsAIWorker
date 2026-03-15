# Role Hints Next Stage

This stage adds conservative role hints after team grouping and spatial mapping.

It is a post-run step only.
It does not change tracking, ball handling, clustering, or team assignment.

## Goal

The goal is to reduce confusion around:

- referee vs player
- goalkeeper vs outfield player

without changing the current baseline pipeline.

## Command

```powershell
ez-worker build-role-hints --run-dir outputs/20260314_175555
```

## Output

This command writes:

- `role_hints.json`
- `player_stats_with_roles.json`

inside the run folder.

## What It Contains

- `goalkeeper_candidate` hints
- `referee_candidate` hints
- `outlier_candidate` hints
- a confidence field
- simple reasons for each hint

This is not a final role classifier.
It is a safe intermediate layer for later improvement.

## Why This Is Safe

This stage only reads the current outputs and writes new files.

If it looks weak:

- keep the existing team outputs
- ignore the role hint files
- fall back to the earlier baseline outputs

## Tutorial References Used

This stage follows both tutorial references:

### Tutorial 1

- supports cleaner downstream football analytics after tracking and grouping

### Tutorial 2

- supports separating confusing groups after appearance-based clustering

## Proposal Alignment

This remains inside the proposal path:

- `YOLO`
- `ByteTrack`
- `SigLIP + UMAP + KMeans`
- later football analytics
