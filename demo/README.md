# Demo outputs — for review

Frozen sample outputs of the EZ Stats AI worker, committed so they can be reviewed
without running the pipeline.

Two runs are kept. They answer different questions, so keep both.

---

## `20260910_170402/` — Messi clip — **the accuracy result**

`leo_messi_30pass.mp4`, 108 s. This is the run to show when someone asks *how
accurate is it*, because it is the only output measured against a complete,
independently written ground truth.

### Measured, not claimed

Scored against `analytics/ground_truth.json` — every pass in 0–97 s annotated by
hand, plus the shot and the goal — at 1.5 s tolerance:

| Class | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **Pass** | 23 | 2 | 4 | **0.92** | **0.85** | **0.885** |
| **Goal** | 1 | 0 | 0 | **1.00** | **1.00** | **1.00** |
| Shot | 0 | 0 | 1 | — | 0.00 | — |
| **Overall** | 24 | 2 | 5 | **0.923** | **0.828** | **0.873** |

The goal is detected at **96.00 s** against an annotated 96.0 s.

Reproduce it:

```
python scripts/evaluate_events.py \
    --events <run>/events_with_goals.json \
    --reference docs/references/messi_full_clip.json \
    --tolerance-seconds 1.5
```

### Files

- **`stats_video.mp4`** — the match video with per-player ellipses, team colours,
  IDs and the stats panel. Re-encoded to 1280×720/30 fps (19 MB); the full-size
  original stays in Drive at `ezstats/outputs/20260910_170402/`.
- **`match_report_merged.json`** — the frontend contract (see
  `docs/BACKEND_INTEGRATION.md`), plus a `measured_accuracy` block carrying the
  table above so the numbers travel with the data.
- `match_report.json` — before post-processing.
- `player_crops/` — one thumbnail per track, referenced by `crop_path`.
- **`analytics/`** — what pitch calibration unlocked on 2026-09-24:
  - `map2d.mp4` — top-down 2D pitch map, players at true positions
  - `heatmap_team_0.jpg`, `heatmap_team_1.jpg` — occupancy heatmaps
  - `player_metrics.json` — distance in **metres** and top speed per track
  - `goal_detected_96.00s.jpg` — the ball inside the projected goal mouth
  - `calibration_overlay.jpg` — the 3D goal frame projected onto the real goal
  - `ground_truth.json`, `measured_score.json` — the reference and the score

### Read these limits before quoting the numbers

- **One clip, one annotator**, timestamps to the nearest second, hence the 1.5 s
  tolerance. A real measurement, not a benchmark.
- **109 tracks for ~22 players.** Identity fragments, so a player can appear as
  several short tracks. Same-team ID swapping is an open research problem — the
  published NPSPT benchmark's best method still logs 130 ID switches in a
  two-minute clip.
- **`null` in the summary means the class is not produced, not that it is zero.**
  There is no shot classifier.
- **The one missed aerial pass** (t=65 s, the long ball across the field) is a
  switch that is deliberately off: aerial contacts are disabled because a ball
  flying over a player was not distinguishable from one at his feet.
- `distance_px` includes camera motion and is not running distance. Use
  `analytics/player_metrics.json` for metres.

---

## `20260608_224414/` — benchmark clip — **the identity result**

`08fd33_4.mp4`, 30 s, Bundesliga. The long-standing reference run: 26 players
(13/13), 8 events, no phantom IDs. Best for showing clean tracking and team
colours.

Measured against `docs/references/08fd33_4_full_clip.json`: **precision 0.625,
recall 0.500**. Much weaker than Messi, and the reason is visible in the data —
**5 of its 10 events put the ball in the air**, against 1 of 29 on Messi. That is
the same disabled aerial capability, quantified.

---

## Viewing

`viewer.html` — open in a browser, click *Load*, choose either
`match_report_merged.json` to browse players, events and stats.

`outputs/`, model weights and raw video are gitignored and stay local.
