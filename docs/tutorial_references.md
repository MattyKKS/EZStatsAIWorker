# Tutorial References

This file records how the current project uses the two tutorial references without drifting away from the proposal stack.

## Reference 1

The first tutorial contributes ideas for:

- football detection and tracking pipeline structure
- separate handling of the ball
- later advanced football analytics
- team and spatial-analysis direction

We used it mainly for:

- detector/tracker workflow shape
- ball-focused cleanup direction
- the idea of later appearance and analytics stages
- later spatial-analysis direction
- later tactical-style outputs after pitch mapping
- cleaner downstream football-role handling after grouping

## Reference 2

The second tutorial contributes ideas for:

- top-half crop focus
- jersey-centered appearance logic
- KMeans-style appearance grouping
- ball interpolation and later football metrics

We used it mainly for:

- focusing team clustering on the upper-body jersey region
- keeping an outlier group for confusing tracks
- supporting later team-aware reporting
- supporting later perspective-transform and football-space metrics work
- supporting later formation-style football-space layouts
- supporting post-cluster handling of confusing non-team tracks

## Important Constraint

These tutorials are references only.

The project must still stay aligned with the proposal stack:

- `YOLO`
- `ByteTrack`
- `OpenCV`
- `FFmpeg`
- `SigLIP + UMAP + KMeans`
- later homography

We do not switch to a different stack just because a tutorial uses a different convenience tool.
