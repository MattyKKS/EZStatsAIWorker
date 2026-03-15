# Possession Report Next Stage

This stage builds a simple possession and event timeline report from the cleaned event file.

It is a post-run step only.
It does not change tracking, ball handling, or team assignment.

## Goal

The goal is to turn the cleaned event file into something easier to inspect and use later.

This stage adds:

- touches by team
- passes by team
- simple possession segments
- a compact event timeline

## Command

```powershell
ez-worker build-possession-report --run-dir outputs/20260314_175555
```

## Output

This command writes:

- `possession_report.json`

inside the run folder.

## Why This Is Safe

This stage only reads `events_analysis_ready.json` and writes a new report file.

If it looks weak:

- keep the cleaned event file
- ignore the possession report
- fall back to the earlier outputs

## Proposal Alignment

This remains inside the proposal path:

- cleaner event work after team and role cleanup
- later football analytics and reporting
