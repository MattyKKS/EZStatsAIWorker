# Output Contract Next Stage

This stage builds a simple contract file for product integration.

It is a post-run step only.
It does not change tracking, analytics, or cleaned outputs.

## Goal

The goal is to tell frontend and backend teammates:

- which file to start from
- which fields are the main ones
- which sections are cleaned and safe to prefer
- where to find the deeper files if needed

## Commands

```powershell
ez-worker build-integration-bundle --run-dir outputs/20260314_175555
ez-worker build-output-contract --run-dir outputs/20260314_175555
```

## Output

This stage writes:

- `integration_bundle.json`
- `output_contract.json`

inside the run folder.

## Why This Is Safe

This stage only reads current output files and writes product-facing helper files.

If it looks weak:

- keep the detailed source files
- ignore the product-facing helpers
- fall back to the earlier outputs

## Proposal Alignment

This remains inside the proposal path:

- stable AI worker outputs first
- clearer integration contracts after that
