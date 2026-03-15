# Next Steps After V1

This document explains what should happen after the V1 baseline.

Use this file for future work.
Use `docs/v1_baseline_summary.md` as the fallback reference.
Use `docs/project_path_and_status.md` as the main high-level project history and current status file.

## Read These Docs Only

If the docs folder feels too crowded, focus on these files first:

1. `docs/project_path_and_status.md`
2. `docs/v1_baseline_summary.md`
3. `docs/tutorial_references.md`
4. `docs/WORK_LOG.md`

The other `*_next_stage.md` files are small step notes.
They are optional unless you are working on that exact step.

## Current Position

Last updated: `2026-03-14 21:45` Bangkok time

Current stage:

- `Step 3` complete enough for now: role hints were added safely
- `Step 4` is active now: improve downstream event quality using safer role/team outputs

What is already done after V1:

- appearance manifest
- team clustering
- team assignment
- team report
- pitch mapping prep
- homography application
- spatial report
- heatmap export
- formation export
- role hints
- analysis-ready filtering

What is next right now:

- build a route manifest for the mock API files

## Rule For Future Work

Do not change too many things at once.

For each new step:

1. keep V1 as the fallback
2. change one area at a time
3. compare the new result against V1
4. keep the change only if it is clearly better

## What V1 Already Solved

V1 already gives us:

- a working worker pipeline
- trained football detector support
- usable player count stability
- improved ball handling
- simple event and stats outputs
- a processed video path

So the next steps should not restart from zero.

## Recommended Next Order

### Step 1. Freeze and document V1

Done by:

- `docs/v1_baseline_summary.md`
- `docs/WORK_LOG.md`

### Step 2. Move to appearance and team grouping

This is the next best step.

Reason:

- player count is stable enough
- ball handling is acceptable enough for a baseline
- goalkeeper/referee confusion is now more of a role/appearance problem than a raw ball problem

Main future stack for this stage:

- player crops
- `SigLIP`
- `UMAP`
- `KMeans`

Goal:

- separate teams more reliably
- help with role confusion
- prepare the system for mapping track IDs to real players later

### Step 3. Improve role separation

After appearance features are in place, improve:

- player vs referee
- player vs goalkeeper

This can be done by:

- detector label improvement
- appearance-based heuristics
- later manual mapping support in the product

The first safe step for this stage is:

- add conservative post-run role hints without changing the tracker

Current status:

- done enough for now
- keep this as a hint layer, not a final classifier

### Step 4. Improve event quality

After team and role stability improve, refine:

- touches
- passes
- shot attempts
- possession logic

Reason:

better events depend on better player/ball understanding

The next safe step for this stage is:

- use role-aware filtered outputs before changing event logic further

Current status:

- role-aware filtered outputs are now done
- the first event-cleanup step is done
- the cleaned team report is done
- the possession/timeline report is done
- the integration bundle is done
- the output contract is done
- the UI-friendly payload is done
- the API-friendly response shape is done
- the route-level mock files are done
- the next safe step is a route manifest for integration handoff

### Step 5. Add team-aware reporting

Before moving to pitch work, make the current outputs easier to use with:

- provisional team IDs
- team summaries
- simple team event reports

This gives cleaner inputs for later product integration.

### Step 6. Add pitch understanding

After the tracking side is stable enough, add:

- pitch keypoints
- homography
- heatmaps
- average positions

This is later-stage work, not the next immediate step.

The first safe step for this stage is:

- prepare a reference frame and a pitch mapping template
- keep it separate from the main tracking pipeline

The next safe step after that is:

- build simple spatial summaries from projected pitch tracks

After that:

- build heatmap and formation-friendly exports from projected pitch tracks

After that:

- build simple tactical-layout exports from projected pitch tracks

## Things To Avoid Right Now

Do not spend too much time now on:

- chasing perfect ball behavior in every frame
- changing the full model stack
- making many threshold changes at once
- adding too many new features before comparing against V1

## Future Change Checklist

Before accepting a new change, check:

- did player count stay reasonable?
- did ball continuity improve or get worse?
- did event count become more believable?
- did processed video become easier to trust?
- is the change still aligned with the proposal?
- can we explain the change simply?

If the answer is no, fall back to V1.

## Simple Summary

V1 is the stable base.

The next smart move is not to rebuild tracking again.
It is to move into appearance, team grouping, and role separation while keeping V1 as the safety point.
