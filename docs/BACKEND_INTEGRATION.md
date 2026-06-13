# Backend Integration Spec — EZ Stats AI Worker → Backend → Frontend

**Purpose:** hand-off for building the backend that bridges the AI worker to the
frontend dashboard. Copy this file into the backend repo (or have the backend
Claude session read it) so it has full context without re-deriving it.

> The backend depends on the worker's **output data contract** (the JSON below),
> NOT on the worker's source code. Do **not** vendor/copy worker code into the
> backend — share data (the `outputs/` folder + this contract) instead.

---

## 1. Architecture

```
[AI worker]  ──writes──▶  outputs/<run_id>/      ──read──▶  [Backend API]  ──HTTP/JSON──▶  [Frontend]
 (Python: YOLO+                match_report_merged.json       (FastAPI/Flask)               (dashboard)
  ByteTrack+SigLIP)            stats_video.mp4
                              player_crops/
```

- **Worker** = batch job: process a match video → write a run folder. Heavy ML
  deps (torch, ultralytics, opencv). Its own repo: `MattyKKS/EZStatsAIWorker`.
- **Backend** = long-lived web service: read run folders, serve them as an API.
  Light deps (just a web framework). Separate repo, separate container.
- **Frontend** = your friend's dashboard. Separate repo. Calls the backend API.

Repo decision (already made): **separate repos**, two containers, wired by a
**shared `outputs/` volume** (or object storage in prod). Repo structure and
container structure are independent — you get 2 containers either way.

---

## 2. Where the data lives (important)

`outputs/`, `artifacts/` (model weights), and `data/` (raw video) are **gitignored
in the worker repo** — they are local/runtime artifacts, never on GitHub. So:

- The worker writes run folders to `outputs/<run_id>/` on whatever machine runs it.
- The backend reads from that folder (shared volume for a single-host demo; S3 /
  Supabase / Firebase Storage for production — same API, different source).
- For a no-backend fallback, a frozen sample is committed at
  **`demo/20260608_224414/`** on branch `rework-team-tracking` (annotated video +
  reports + crops + `demo/viewer.html`).

`<run_id>` is the run-folder name, a timestamp like `20260608_224414`.

---

## 3. Per-run files (what the backend serves)

| File | Serve to frontend? | What it is |
|---|---|---|
| `match_report_merged.json` | ✅ **canonical** | cleaned analytics (ship this) |
| `match_report.json` | optional | raw analytics before post-processing |
| `stats_video.mp4` | ✅ | match video with overlays + stats panel + radar |
| `spatial_video.mp4` | ✅ (when present) | top-down radar (Voronoi + ball trail) |
| `player_crops/track_XXXX/*.jpg` | ✅ | player thumbnails (see `crop_path`) |
| `processed_video.mp4` | ❌ | redundant intermediate |
| `tracks*.json`, `*_stats*.json`, `team_*`, `pitch_keypoints.json`, etc. | ❌ | worker intermediates |

**Raw vs merged:** the worker emits `match_report.json` (raw, ~39 players incl.
referees / duplicates / mislabels) **and** `match_report_merged.json` (cleaned:
merges tracking fragments, drops referees + off-pitch detections, collapses GK
fragments, corrects team labels by jersey colour, fixes ID-swaps, re-derives
events). **Always serve the merged one.** It is produced by
`src/ez_worker/postprocess/report_cleanup.py` — currently a **manual post-step**
(`python -m src.ez_worker.postprocess.report_cleanup <run_dir> --apply`). Wiring
it into the pipeline so every run emits it automatically is a pending worker task.

---

## 4. THE CONTRACT — `match_report_merged.json`

Top level:

```jsonc
{
  "match_id": "20260608_224414",
  "video": "data/raw/08fd33_4.mp4",
  "duration_s": 30.0,
  "fps": 25.0,
  "possession": { "0": 50.0, "1": 50.0 },   // % ball possession per team_id (keys are strings)
  "players":  [ ... ],
  "events":   [ ... ],
  "pass_network": { "nodes": [...], "edges": [...] },
  "summary":  { ... }
}
```

**player:**
```jsonc
{
  "track_id": 1,                // stable per-run player id
  "label": "T2#1",              // display label "T<team_id+1>#<id>"
  "team_id": 1,                 // 0 or 1 (the two teams; GK is assigned to a team)
  "touches": 0, "passes": 1, "shots": 0,
  "distance_px": 2440.9,        // total distance covered, pixels
  "crop_path": "player_crops/track_0001/frame_000000.jpg"  // relative to run folder
}
```

**event:**
```jsonc
{
  "type": "pass",               // touch | pass | long_ball | clearance | interception | shot | goal
  "frame": 659,
  "time_s": 26.4,               // = frame / fps; use to seek the video
  "actor": 5,  "actor_label": "T2#5",
  "target": 3, "target_label": "T2#3",   // null for non-directed events (touch/clearance/shot)
  "details": { }                // event-specific extras, e.g. {"distance_cm": 70.4}
}
```
Semantics: same-team actor→target = `pass` / `long_ball`; opponent = `interception`
/ `clearance`.

**pass_network:**
```jsonc
{ "nodes": [ { "id": 1, "label": "T2#1", "team_id": 1 } ],
  "edges": [ { "from": 25, "to": 6, "count": 1 } ] }   // directed pass counts
```

**summary:**
```jsonc
{ "total_touches": 2, "total_passes": 3, "total_long_balls": 0,
  "total_clearances": 1, "total_interceptions": 2, "total_shots": 0, "total_goals": 0 }
```

Frontend tips: `time_s` seeks the video to each event; `team_id` 0/1 → your two
team colours; build a crop URL as `/matches/{run_id}/crops/{crop_path}`.

---

## 5. Recommended backend API

| Endpoint | Returns |
|---|---|
| `GET /matches` | list of run_ids (folder names under `outputs/`) |
| `GET /matches/{id}/report` | contents of `match_report_merged.json` |
| `GET /matches/{id}/video/stats` | stream `stats_video.mp4` |
| `GET /matches/{id}/video/spatial` | stream `spatial_video.mp4` (if present) |
| `GET /matches/{id}/crops/{path}` | serve a player crop image |

Minimal viable bridge for the demo: a ~30-line FastAPI app that serves the
`outputs/` directory as static files + a `/report` endpoint returning the merged
JSON. Stack suggestion: **FastAPI + uvicorn**, CORS enabled for the frontend origin.

---

## 6. Deployment sketch (`docker-compose.yml`, shared volume)

```yaml
services:
  worker:                       # built from the worker repo
    build: ../EZStatsAIWorker
    volumes: [ "shared-outputs:/app/outputs" ]      # worker writes here
  backend:                      # this repo
    build: .
    ports: [ "8000:8000" ]
    volumes: [ "shared-outputs:/app/outputs:ro" ]   # backend reads (read-only)
volumes: { shared-outputs: {} }
```

For prod, replace the shared volume with object storage and have the backend read
run folders from there.

---

## 7. Status / open items (as of 2026-06-13)

- ✅ Worker produces clean reports; `report_cleanup.py` validated on run
  `20260608_224414` (39→26 players, teams 13/13, 8 events, ID-swap fixed).
- ✅ Demo committed at `demo/` (branch `rework-team-tracking`) as a backend-down fallback.
- ⏳ Wire `tracklet_merge` + `report_cleanup` into the worker pipeline (auto-emit merged report).
- ⏳ Build the backend (this spec).
- ⏳ Team classification (SigLIP) still mislabels some players upstream; the colour
  vote in `report_cleanup` corrects it post-hoc, but a better upstream method is
  pending research. Not a backend concern — the merged report is already corrected.
