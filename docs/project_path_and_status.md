# Project Path And Status

This is the main project history and roadmap file.

If a new person joins the project, read this file first.
If the docs folder feels crowded, use this file together with:

1. `docs/v1_baseline_summary.md`
2. `docs/tutorial_references.md`
3. `docs/WORK_LOG.md`

Last updated: `2026-03-15 00:10` Bangkok time

## 1. Project Goal

Build the AI worker first before frontend or backend integration.

The worker should:

- read a football match video
- detect players and the ball
- track them
- create simple football events and stats
- assign provisional teams
- project tracks onto the pitch
- export files that later frontend and backend work can use

This project must stay aligned with the proposal stack:

- `YOLO`
- `ByteTrack`
- `OpenCV`
- `FFmpeg`
- `SigLIP + UMAP + KMeans`
- later homography and pitch analytics

## 2. Current Position

Current main stage:

- `Step 4` in the post-V1 plan

Current sub-step:

- set up V2 retraining for goalkeeper, referee, and touchline-player failures

Current safe baseline:

- `docs/v1_baseline_summary.md`

Current run used for the later steps:

- `outputs/20260314_175555`

## 3. What Is Finished So Far

### Stage 0. Worker foundation

Status:

- done

What was built:

- CLI
- pipeline orchestration
- config and schemas
- output writers
- mock provider

Why this was done:

- to make the project runnable end to end before real AI work

### Stage 1. Real football detection and tracking baseline

Status:

- done enough for baseline use

What was built:

- Ultralytics-based provider
- trained football detector support
- player tracking
- separate ball handling
- cleaned rendering

Why this was done:

- to replace mock outputs with real football CV results

Important result:

- stable enough V1 baseline

### Stage 2. Ball stability improvements

Status:

- done enough for baseline use

What was built:

- separate high-resolution ball pass
- one-ball-per-frame logic
- ball jump cleanup
- controlled ball re-acquisition
- fallback to tracked-stream ball hints when needed

Why this was done:

- the ball was the main blocker for event quality

Important result:

- ball is not perfect, but much more usable than the original baseline

### Stage 3. Appearance and team grouping

Status:

- done enough for current next stages

What was built:

- player crop export
- appearance manifest
- team clustering with `color`, `siglip`, and `siglip-jersey`
- provisional team assignment
- team report

Why this was done:

- player count became stable enough, so the next real problem was team and role confusion

Important result:

- two main team groups plus one outlier group are now available

### Stage 4. Pitch mapping and spatial outputs

Status:

- done enough for current next stages

What was built:

- pitch mapping prep
- manual homography application
- pitch-projected tracks
- spatial report
- heatmap export
- formation export

Why this was done:

- to turn tracked positions into football-space outputs the product can use later

Important result:

- projected pitch coordinates and simple tactical-layout exports are now available

### Stage 5. Role hints

Status:

- done enough as a hint layer

What was built:

- `goalkeeper_candidate`
- `referee_candidate`
- `outlier_candidate`

Why this was done:

- the user observed referee and goalkeeper confusion
- we needed a safe step before changing event logic again

Important result:

- referee/outlier isolation is useful
- goalkeeper hints are usable but not final

### Stage 6. Analysis-ready filtering

Status:

- just completed

What was built:

- filtered downstream files that remove likely referee/outlier tracks

Files created:

- `analysis_ready_summary.json`
- `tracks_analysis_ready.json`
- `player_stats_analysis_ready.json`

Why this was done:

- later event-quality work should use cleaner player-only data

Important result from current run:

- removed 6 likely referee/outlier tracks
- kept 16 analysis-ready players

### Stage 7. Event cleanup on top of analysis-ready outputs

Status:

- completed for the current safe version

What was built:

- a filtered event export that removes events involving removed referee/outlier tracks
- team and role context added onto the kept events

Why this was done:

- the original event file still contained passes and touches involving tracks that the later role step marked as likely referee/outlier

Important result from current run:

- source events reduced from 21 to 16
- 5 noisy touch/pass events were removed

### Stage 8. Analysis-ready team report

Status:

- completed for the current safe version

What was built:

- a cleaner team report that uses `events_analysis_ready.json` and `player_stats_analysis_ready.json`

Why this was done:

- the older team report still reflected noisy events from removed referee/outlier tracks

### Stage 9. Possession and timeline reporting

Status:

- completed for the current safe version

What was built:

- a possession report that summarizes touches, passes, simple possession segments, and a compact timeline from `events_analysis_ready.json`

Why this was done:

- the cleaned event file is now stable enough to feed a more readable report for later product and analytics work

### Stage 10. Integration-ready bundle

Status:

- completed for the current safe version

What was built:

- one bundle file that summarizes the cleaned run and points to the important artifacts for frontend/backend use

Why this was done:

- the output set is now large enough that product-side work needs one simple entry point

Important result:

- `integration-bundle-v2` now includes both raw and cleaned video summary sections
- `video_cleaned` is the preferred product-facing summary

### Stage 11. Output contract for integration

Status:

- completed for the current safe version

What was built:

- a simple contract file that explains which bundle sections and fields frontend/backend should use first

Why this was done:

- new teammates should not need to inspect many JSON files to understand the worker outputs

### Stage 12. UI-friendly payload

Status:

- completed for the current safe version

What was built:

- one UI-oriented payload that reshapes the cleaned bundle into video, scoreboard, formation, timeline, and player sections

Why this was done:

- frontend work needs a simpler shape than the deeper integration bundle

### Stage 13. API-friendly response shape

Status:

- completed for the current safe version

What was built:

- one API-like response file with `status`, `data`, and `meta` sections

Why this was done:

- backend and frontend integration both benefit from a stable response shape that looks like a real endpoint result

### Stage 14. Mock endpoint set

Status:

- completed for the current safe version

What was built:

- a split mock API folder with separate endpoint-style files for match, scoreboard, formation, timeline, possession, and players

Why this was done:

- frontend and backend work often needs route-level mock files instead of one combined response

### Stage 15. Route manifest for mock endpoints

Status:

- completed for the current safe version

What was built:

- one route manifest that maps the mock files to concrete API-style paths

Why this was done:

- teammates need a simple handoff from files to routes when wiring frontend/backend

### Stage 16. Return to AI-worker quality improvement

Status:

- now in progress

What is being improved:

- player continuity across frames
- referee/goalkeeper separation
- tighter referee/outlier handling in downstream files

Why this is being done:

- the AI worker is the core project deliverable
- integration should wait until the worker outputs are cleaner

## 4. What Is Not Finished Yet

These parts are still not final:

- goalkeeper identification
- referee identification
- event quality
- ball reliability during the hardest fast-motion cases
- true tactical understanding
- final backend/frontend integration

## 4A. Current AI Quality Focus

Right now the quality work is focused on:

1. preserving the detector's original class signal all the way through the worker
2. stopping referee-like tracks from competing equally with true players during cleanup
3. using detector class votes plus spatial hints to improve goalkeeper and referee separation
4. tightening analysis-ready outputs before any further integration work

## 5. What We Are Doing Now

Right now the project is here:

1. V1 baseline is frozen as fallback.
2. Team and spatial stages are added as safe post-run modules.
3. Role hints and analysis-ready filtering are added.
4. The next improvement is AI-worker quality again, not integration.
5. The current immediate step is targeted retraining for goalkeeper/referee and touchline-player failures.

## 6. What Comes Next

### Next step

Prepare and run the V2 retraining pass.

Goal:

- make the detector emit real `goalkeeper` and `referee` labels on the benchmark clip
- reduce sideline-player loss at the detector level
- keep the current worker structure and ball pipeline stable

### After that

Rebuild the role, analysis-ready, and event-cleanup layers from the retrained outputs.

Goal:

- measure whether player continuity improves
- measure whether referee/outlier removal becomes cleaner

### After that

Return to event-quality improvements on top of the cleaner retrained outputs.

Goal:

- improve football usefulness without changing the proposal stack

## 7. How To Read The Docs

Use these files as the main set:

- `docs/project_path_and_status.md`
  - complete high-level history and current position
- `docs/v1_baseline_summary.md`
  - safe fallback baseline
- `docs/tutorial_references.md`
  - how the two tutorials influence the project
- `docs/WORK_LOG.md`
  - short running record of recent changes

The smaller `*_next_stage.md` files are step notes.
They are not the main source of truth anymore.

## 8. Simple Summary

The project is moving in the right order.

We are not restarting from zero.
We already have:

- working tracking
- improved ball handling
- team grouping
- pitch mapping
- spatial exports
- role hints
- analysis-ready filtered outputs

The next real work is AI-worker quality improvement on top of the current safe baseline, while keeping V1 as fallback.
