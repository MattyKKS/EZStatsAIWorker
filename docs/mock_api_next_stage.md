# Mock API Next Stage

This stage builds mock endpoint files from the API-friendly response.

It is a post-run step only.
It does not change tracking or analytics.

## Goal

The goal is to give frontend and backend work separate files that resemble real endpoint responses.

This stage creates:

- `match.json`
- `scoreboard.json`
- `formation.json`
- `timeline.json`
- `possession.json`
- `players.json`
- `index.json`

## Command

```powershell
ez-worker build-mock-api --run-dir outputs/20260314_175555
```

## Output

This command writes a `mock_api/` folder inside the run folder.

## Why This Is Safe

This stage only reads `api_response.json` and writes split mock files.

If it looks weak:

- keep the API response file
- ignore the mock API folder

## Proposal Alignment

This remains inside the proposal path:

- stable AI worker outputs first
- integration-ready mock routes after that
