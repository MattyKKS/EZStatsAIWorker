# API Response Next Stage

This stage builds one API-friendly response file from the cleaned product payloads.

It is a post-run step only.
It does not change tracking or analytics.

## Goal

The goal is to give backend or frontend integration one response shape that looks like a real endpoint result.

This stage adds:

- `status`
- `api_version`
- one `data` object
- one `meta` object

## Command

```powershell
ez-worker build-api-response --run-dir outputs/20260314_175555
```

## Output

This command writes:

- `api_response.json`

inside the run folder.

## Why This Is Safe

This stage only reads:

- `ui_payload.json`
- `output_contract.json`

and writes one new helper file.

If it looks weak:

- keep the deeper bundle and contract files
- ignore the API response file

## Proposal Alignment

This remains inside the proposal path:

- AI worker outputs first
- product/backend integration shapes after that
