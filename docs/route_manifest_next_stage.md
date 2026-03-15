# Route Manifest Next Stage

This stage builds one route manifest for the mock API files.

It is a post-run step only.
It does not change tracking or analytics.

## Goal

The goal is to make it obvious which mock file should be used for which API path.

## Command

```powershell
ez-worker build-route-manifest --run-dir outputs/20260314_175555
```

## Output

This command writes:

- `mock_api/route_manifest.json`

inside the run folder.

## Why This Is Safe

This stage only reads `mock_api/index.json` and writes one helper file.

If it looks weak:

- keep the mock API files
- ignore the route manifest

## Proposal Alignment

This remains inside the proposal path:

- stable AI worker outputs first
- clearer integration handoff after that
