# Appearance Next Stage

This document explains the next stage after the V1 baseline.

The goal is not to replace the current tracking stack.
The goal is to build the next proposal-aligned module on top of it.

## What This Stage Is

This stage prepares the project for:

- `SigLIP`
- `UMAP`
- `KMeans`

These are the tools planned for:

- team grouping
- appearance consistency
- helping separate similar-looking roles later

## What We Already Have

From the V1 baseline, we already have:

- stable enough player counts
- improved ball handling
- exported player crops

That means the next step is to organize those crops in a clean way for the future embedding pipeline.

## New Command

Use this command on a completed run folder:

```powershell
ez-worker prepare-appearance --run-dir outputs/<run_id>
```

Example:

```powershell
ez-worker prepare-appearance --run-dir outputs/20260314_173446
```

## What It Creates

It creates:

- `appearance_manifest.json`

inside the selected run folder.

## What The Manifest Contains

For each tracked player, the manifest stores:

- the track ID
- how many crops were exported
- a few sample crop paths
- a simple upper-body average color summary

The color summary is not the final team-classification method.
It is only lightweight metadata for inspection and preparation.

The planned real next-stage pipeline is still:

`player crops -> SigLIP embeddings -> UMAP -> KMeans`

## Why This Stage Matters

This helps the team:

- inspect crop quality
- see whether the tracks have enough visual data
- prepare the data structure for the next appearance model step
- stay aligned with the proposal without changing the current detection/tracking stack

## After The Manifest

The next command after this stage is:

```powershell
ez-worker cluster-teams --run-dir outputs/<run_id> --method color --cluster-count 3
```

For the proposal-aligned version, later use:

```powershell
pip install -e .[appearance]
ez-worker cluster-teams --run-dir outputs/<run_id> --method siglip --cluster-count 3
```

## What This Stage Does Not Do Yet

It does not yet:

- assign final teams
- run SigLIP
- run UMAP
- run KMeans
- solve goalkeeper/referee confusion fully

Those belong to the next implementation step after this prep stage.
