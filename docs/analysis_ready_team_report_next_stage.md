# Analysis-Ready Team Report Next Stage

This stage rebuilds the team report using the cleaned analysis-ready event file.

It is a post-run step only.
It does not change tracking, ball handling, team assignment, or role hints.

## Goal

The goal is to make the team-level summary match the cleaner event file instead of the older noisy one.

This stage uses:

- `player_stats_analysis_ready.json`
- `events_analysis_ready.json`

## Command

```powershell
ez-worker build-analysis-ready-team-report --run-dir outputs/20260314_175555
```

## Output

This command writes:

- `team_report_analysis_ready.json`

inside the run folder.

## Why This Is Safe

This stage only reads current outputs and writes a new report file.

If it looks weak:

- keep the earlier `team_report.json`
- ignore the analysis-ready team report
- fall back to the previous outputs

## Proposal Alignment

This remains inside the proposal path:

- tracking first
- team and role cleanup second
- cleaner event and report outputs after that
