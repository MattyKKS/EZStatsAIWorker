# Team Report Next Stage

This stage builds a simple team-aware report from the current post-run outputs.

It still does not change tracking.
It uses the existing files from:

- clustering
- team assignment
- events
- player stats

## Goal

The goal is to make the current outputs easier to use.

Instead of only having per-player files, this stage adds:

- team totals
- touch counts by team
- pass relations between teams
- a simple team summary file

## Command

```powershell
ez-worker build-team-report --run-dir outputs/20260314_175555
```

## Output

This command writes:

- `team_report.json`

inside the run folder.

## Why This Is Safe

This is still a post-run step.

If the result looks weak:

- keep V1 tracking
- keep team clustering as-is
- ignore the report file

## Tutorial References Used

This step still follows both tutorial references:

### Tutorial 1

- supports building richer football summaries after detection and tracking
- supports later team-level analytics

### Tutorial 2

- supports using team assignment information built from upper-body appearance cues
- supports later team-level football metrics

## Proposal Alignment

This stage stays inside the proposal stack.

It is only using outputs already produced by:

- `YOLO`
- `ByteTrack`
- `SigLIP + UMAP + KMeans`

No outside competing stack is introduced.
