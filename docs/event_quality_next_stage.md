# Event Quality Next Stage

This stage builds a cleaner event file from the analysis-ready player set.

It is a post-run step only.
It does not change tracking, ball handling, team assignment, or role hints.

## Goal

The goal is to remove obvious noisy events caused by likely referee or outlier tracks.

This stage:

- keeps events involving analysis-ready players
- drops events involving removed referee/outlier tracks
- enriches kept events with team and role context

## Command

```powershell
ez-worker build-analysis-ready-events --run-dir outputs/20260314_175555
```

## Output

This command writes:

- `events_analysis_ready.json`

inside the run folder.

## Why This Is Safe

This stage only reads current outputs and writes a new filtered event file.

If it looks weak:

- keep the original `events.json`
- ignore the analysis-ready event file
- fall back to the earlier outputs

## Proposal Alignment

This remains inside the proposal path:

- tracking first
- team and role cleanup second
- cleaner football events after that
