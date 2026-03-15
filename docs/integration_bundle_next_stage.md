# Integration Bundle Next Stage

This stage builds one integration-ready bundle from the cleaned outputs.

It is a post-run step only.
It does not change tracking, ball handling, or analytics logic.

## Goal

The goal is to give frontend and backend teammates one clear file that points to the important outputs and summarizes the current run.

This stage gathers:

- video summary
- cleaned team summary
- cleaned event summary
- possession summary
- formation summary
- file paths to the detailed artifacts

## Command

```powershell
ez-worker build-integration-bundle --run-dir outputs/20260314_175555
```

## Output

This command writes:

- `integration_bundle.json`

inside the run folder.

## Why This Is Safe

This stage only reads already-built output files and writes one new bundle file.

If it looks weak:

- keep the detailed source files
- ignore the bundle
- fall back to the earlier outputs

## Proposal Alignment

This remains inside the proposal path:

- stable worker outputs first
- integration-friendly contracts after that
