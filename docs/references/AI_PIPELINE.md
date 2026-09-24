# AI pipeline reference

The full stage-by-stage document lives at:

    docs/AI_PIPELINE_CURRENT_AND_FUTURE.md

Kept in one place on purpose. Two copies of a living document drift, and a stale
copy in a folder named "references" is worse than no copy — this project has
already lost months to notes that said a component worked when it did not.

What it covers, for all 10 stages (video load -> detection -> tracking ->
appearance -> teams -> calibration -> transform -> events -> report/render):
technology used, what it is for, what the demo used, what runs now, what was
measured, and what replaces it.

Event ground-truth annotations also live in this folder:
  08_development_window.json
  messi_development_windows.json
Score a run with:
  python scripts/evaluate_events.py --events <run>/events.json \
      --reference docs/references/messi_development_windows.json
