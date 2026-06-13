# Demo output — for review

Frozen sample output of the EZ Stats AI worker, included so it can be reviewed
without running the pipeline (fallback while the backend bridge is being built).

## Run: `20260608_224414` (clip: 08fd33_4.mp4, 30 s, Bundesliga)

- **`stats_video.mp4`** — the match video with AI overlays: per-player ellipses +
  team colours, tracker IDs, live stats panel, and the top-down radar. **Open this
  to see the result.**
- **`match_report_merged.json`** — the structured analytics (cleaned): 26 players,
  2 teams (13/13), goalkeepers, 8 events (touches / passes / interceptions /
  clearance), possession, pass network, and a summary.
- **`match_report.json`** — the raw report before post-processing (39 players),
  kept for comparison.
- **`player_crops/`** — thumbnail of each tracked player (referenced by
  `crop_path` in the JSON).
- **`viewer.html`** (in `demo/`) — open in a browser, click *Load*, and select
  `match_report_merged.json` to browse the players/events/stats interactively.

This is sample data only; `outputs/`, model weights, and raw video are not in the
repo (they stay local). See the JSON schema for the frontend data contract.
