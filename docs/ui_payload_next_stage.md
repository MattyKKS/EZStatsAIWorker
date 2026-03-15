# UI Payload Next Stage

This stage builds one UI-friendly payload from the cleaned integration outputs.

It is a post-run step only.
It does not change tracking or analytics.

## Goal

The goal is to give frontend work one simple file with:

- cleaned video summary
- scoreboard-style team info
- formation cards
- timeline items
- possession segments
- analysis-ready player list

## Command

```powershell
ez-worker build-ui-payload --run-dir outputs/20260314_175555
```

## Output

This command writes:

- `ui_payload.json`

inside the run folder.

## Why This Is Safe

This stage only reads the cleaned integration outputs and writes one new helper file.

If it looks weak:

- keep `integration_bundle.json`
- keep `output_contract.json`
- ignore the UI payload

## Proposal Alignment

This remains inside the proposal path:

- stable AI worker outputs first
- product-ready payloads after that
