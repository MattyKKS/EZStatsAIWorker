# Team Clustering Next Stage

This stage adds appearance grouping on top of the V1 baseline.

It does not change the tracker.
It works from exported player crops after a run is complete.

## Goal

The goal is to group tracked players by visual appearance.

This is useful for:

- separating the two teams
- identifying likely outliers like referees
- preparing for later goalkeeper/referee handling

## Supported Modes

### 1. `color`

This is the light preview mode.

It clusters tracks using the upper-body color summaries already stored in `appearance_manifest.json`.

Use this first because:

- it is fast
- it needs no new model downloads
- it lets us verify the pipeline shape

### 2. `siglip`

This is the proposal-aligned mode.

It uses:

- `SigLIP` image embeddings
- optional `UMAP` reduction
- `KMeans` clustering

This is the main target path for the project.

### 3. `siglip-jersey`

This is the safer tutorial-inspired improvement.

It still uses:

- `SigLIP`
- optional `UMAP`
- `KMeans`

But it focuses the embedding on the upper-body jersey area instead of the full crop.
This follows the tutorial idea of concentrating on the shirt region for team separation.

## Command

Preview with color clustering:

```powershell
ez-worker cluster-teams --run-dir outputs/20260314_175555 --method color --cluster-count 3
```

Proposal-aligned clustering:

```powershell
pip install -e .[appearance]
ez-worker cluster-teams --run-dir outputs/20260314_175555 --method siglip --cluster-count 3
```

Tutorial-inspired jersey-focused clustering:

```powershell
ez-worker cluster-teams --run-dir outputs/20260314_175555 --method siglip-jersey --cluster-count 3
```

If `SigLIP` fails with a missing dependency like `SentencePiece` or `protobuf`, install the updated appearance extras again:

```powershell
pip install -e .[appearance]
```

## Output

This command writes:

- `team_clusters.json`

inside the selected run folder.

## Why Cluster Count 3

For now, `3` is a practical starting point:

- team A
- team B
- likely outlier group such as referee or goalkeeper confusion

Later, this can be tuned.

## Safety

This stage is separate from V1 tracking.

If clustering gives bad results:

- keep V1 tracking as-is
- ignore the clustering output
- fall back to `docs/v1_baseline_summary.md`

## Tutorial References Used

This stage uses both tutorial references as guidance:

- Tutorial 1 supports the wider football pipeline and the move toward appearance-based grouping.
- Tutorial 2 directly supports jersey-focused preprocessing by emphasizing the upper-body shirt region.
