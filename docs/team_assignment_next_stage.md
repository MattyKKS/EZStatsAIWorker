# Team Assignment Next Stage

This stage turns appearance clusters into provisional team IDs.

It stays separate from the V1 tracking baseline.
It does not rerun detection or tracking.

## Goal

After clustering, we need a simple way to use the result.

This step converts cluster output into:

- track-level provisional `team_id`
- player stats with `team_id`
- a simple team summary

## Why This Is Safe

This is a post-run step only.

If the assignments look wrong:

- keep V1 tracking
- keep the raw clustering output
- ignore the team assignment files

## How It Decides Teams

For now, it uses a simple rule:

- the two largest clusters become provisional teams
- smaller leftover clusters are treated as outlier groups

This is practical because the current `siglip-jersey` result already looks like:

- team A
- team B
- mixed or outlier group

## Command

```powershell
ez-worker apply-team-clusters --run-dir outputs/20260314_175555
```

## Output Files

This command creates:

- `team_assignments.json`
- `tracks_with_teams.json`
- `player_stats_with_teams.json`

## Tutorial References Used

This stage follows ideas from both tutorial references:

### Tutorial 1

- advanced football pipeline with detection, tracking, team logic, and later analytics
- supports the direction of using appearance-based grouping as part of the wider worker

### Tutorial 2

- focuses on the top-half jersey region
- uses clustering logic to separate useful shirt information from background
- supports using appearance groups to reason about teams and outliers

## Proposal Alignment

This stage still stays inside the proposal stack:

- `YOLO`
- `ByteTrack`
- `SigLIP + UMAP + KMeans`

No new competing stack is introduced here.
