# EZStats — system architecture and AI pipeline

**For:** redrawing the system architecture diagram and the poster
**Date:** 2026-09-25
**Status:** everything below is what the code actually does today. Where something
was planned but not built, it says so. Every accuracy number is measured against
hand-written ground truth and is reproducible from a command in this document.

---

## 1. What changed since the original proposal — read this first

The original architecture diagram is now wrong in three ways. Please redraw:

| Originally drawn | Reality today | Action for the diagram |
|---|---|---|
| **Redis + BullMQ job queue** | **Not implemented.** `redis` appears in `docker-compose.yml` but is **not a dependency in `package.json`** and no code uses it. There is no queue, no worker pool, no job scheduling. | **Remove Redis and the queue entirely** |
| **Cloud deployment** (RunPod / Lambda / hosted API) | **Not deployed.** Everything runs locally: Postgres in Docker, backend on `localhost:4000`, frontend on `localhost:3000`, AI worker on the developer's machine or Google Colab. | **Remove cloud/hosting boxes**; label it "local / single machine" |
| **Backend triggers the AI worker** | **Manual hand-off.** The worker is run by hand and writes a run folder; the backend later *reads* that folder from disk. Nothing calls the worker over the network. | Draw the link as **"shared folder (manual)"**, a dashed arrow, not an API call |

Everything else in the proposal — NestJS, Prisma, PostgreSQL, Docker, Next.js,
the Python worker — is real and in use.

---

## 2. System architecture (three separate repositories)

```
┌────────────────────────┐   HTTP/JSON    ┌────────────────────────┐
│  FRONTEND              │ ◀────────────▶ │  BACKEND               │
│  Next.js (App Router)  │   cookie auth  │  NestJS + Prisma       │
│  React 19, TypeScript  │                │  TypeScript            │
│  Tailwind CSS          │                └───────────┬────────────┘
│  hls.js (video)        │                            │ Prisma ORM
│  localhost:3000        │                            ▼
└────────────────────────┘                ┌────────────────────────┐
                                          │  PostgreSQL 16         │
                                          │  (Docker container)    │
                                          └────────────────────────┘
                                                      ▲
                          backend READS the run folder│ (read-only)
                                                      │
┌─────────────────────────────────────────────────────┴──────────┐
│  AI WORKER  —  Python, offline batch, run manually             │
│  YOLOv8 · ByteTrack · SigLIP · PnLCalib · OpenCV               │
│  input:  match video (.mp4)                                    │
│  output: outputs/<run_id>/  →  match_report_merged.json,       │
│          stats_video.mp4, player_crops/, analytics/            │
└────────────────────────────────────────────────────────────────┘
```

**Three repos, deliberately separate.** The backend depends on the worker's
**output JSON contract**, never on its source code. The worker's heavy ML
dependencies (torch, ultralytics) never enter the backend.

---

## 3. Tech stack, per component — what is actually installed

### Frontend — `EZ Stats Frontend/ezstats`

| Layer | Technology |
|---|---|
| Framework | **Next.js** (App Router) |
| UI | **React**, **TypeScript** |
| Styling | **Tailwind CSS**, PostCSS, autoprefixer |
| Video playback | **hls.js** |
| Icons | lucide-react |
| Lint | ESLint |

**Pages:** `/` · `/login` · `/register` · `/dashboard` · `/upload` ·
`/team-profile` · `/match-statistics` · `/match-statistics/[matchId]` ·
`/player-statistics` · `/player-statistics/[playerId]`

**Shared lib:** `lib/api.ts` (backend client), `lib/types.ts`,
`lib/TeamContext.tsx` (React context), `middleware.ts` (route guarding)

### Backend — `EZStatsBackend`

| Layer | Technology |
|---|---|
| Framework | **NestJS** (`@nestjs/common`, `core`, `config`, `platform-express`) |
| Language | **TypeScript** |
| ORM | **Prisma** (`@prisma/client`) |
| Database | **PostgreSQL** (Docker) |
| Auth | **bcryptjs** password hashing + **DB-backed sessions** via `cookie-parser` |
| Validation | **class-validator**, **class-transformer** (DTOs) |
| Video | H.264 transcode on upload so the browser can play it |
| Container | **Docker / docker-compose** |

**Modules:** `auth` · `teams` · `players` · `matches` · `health` · `prisma`

**Data model (Prisma):**
`User` → `Session` · `Team` → `Player`, `Match` · `Match` → `PlayerTrackMap` ·
enum `MatchStatus`

`PlayerTrackMap` is the bridge between the worker and the roster: it maps the
worker's `track_id` to a real `Player`, which is how fragmented identities get
combined into one person's stats.

**API surface:**

| Area | Endpoints |
|---|---|
| Auth | `POST /auth/register` · `POST /auth/login` · `POST /auth/logout` · `GET /auth/me` |
| Teams | `GET/POST/PATCH/DELETE /teams/:id` · `POST/DELETE /teams/:id/logo` |
| Players | `GET/POST/PATCH/DELETE /teams/:teamId/players`, `/players/:id` |
| Matches | `GET/POST/PATCH/DELETE /teams/:teamId/matches`, `/matches/:id` |
| **AI output** | `GET /matches/:id/report` · `GET /matches/:id/video/stats` · `GET /matches/:id/video/spatial` · `GET /matches/:id/crops/*` · `GET /matches/:id/player-stats` |
| Identity mapping | `GET/POST /matches/:id/track-maps` |
| Upload | `POST /matches/:id/video` |
| Health | `GET /health` |

### AI worker — `EZ Stats AI Worker`

| Layer | Technology |
|---|---|
| Language | **Python 3.11–3.13** |
| Detection | **Ultralytics YOLOv8** (`yolov8m` players, `yolov8x` ball) |
| Tracking | **ByteTrack** (Ultralytics) |
| Appearance / teams | **SigLIP** (`google/siglip-base-patch16-224`) + **UMAP** + **KMeans** |
| **Pitch calibration** | **PnLCalib** (HRNetv2 keypoints **+ field lines**) ⚠️ GPL-2.0 |
| Geometry / CV | **OpenCV**, **NumPy**, **scikit-learn**, **shapely** |
| Helpers | **supervision** (pinned `0.25.1`), **Roboflow `sports`** |
| Schema | **Pydantic** |
| Compute | local NVIDIA GPU (CUDA) or **Google Colab T4** |

---

## 4. The AI pipeline — 10 stages, with technology and output

Draw this as a **linear pipeline with one branch at the end** (report + video).

| # | Stage | Technology | Input → Output |
|---|---|---|---|
| 1 | **Video load** | OpenCV | `.mp4` → frames, fps, resolution |
| 2 | **Player detection** | YOLOv8m (150 ep, 298 imgs) | frame → boxes: player / goalkeeper / referee |
| 3 | **Ball detection** | YOLOv8x @1280 + tiled inference (`InferenceSlicer`) | frame → ball box |
| 4 | **Tracking** | ByteTrack (Kalman + IoU) | boxes → persistent `track_id` |
| 5 | **Appearance** | SigLIP → 768-d embedding per crop | crop → embedding |
| 6 | **Team assignment** | **Jersey colour**: top-half crop → KMeans → corner vote → per-track clustering (SigLIP as fallback) | track → `team_id`, **one decision per track** |
| 7 | **Pitch calibration** | **PnLCalib** — HRNetv2 heatmaps for keypoints *and* line extremities, RANSAC+DLT, PnL refinement | frame → camera matrix |
| 8 | **Coordinate transform** | Homography, quality-gated | pixel (foot point) → **pitch metres** |
| 9 | **Event detection** | Rule engine (`contacts_v2`): foot distance scaled by player height, ambiguity margin, temporal support. **Goals:** 3D goal mouth projected into the image, ball-inside test | tracks + ball + camera → passes, interceptions, touches, **goals** |
| 10 | **Outputs** | Custom writers + OpenCV rendering | → `match_report_merged.json`, `stats_video.mp4`, `map2d.mp4`, heatmaps, `player_metrics.json` |

### Two design decisions worth putting on the poster

**Stage 7 is the gate.** Stages 8, 9-goal and 10-metrics are all impossible
without it. The previous pitch model produced a usable camera on **2.6%** of
frames on one test clip; PnLCalib raised that to **73.7%**, and to **100%**
through the goal sequence. Every spatial feature — 2D map, heatmaps, distance in
metres, goal line — arrived at once when that stage was fixed.

**Goals are detected in image space, not on the ground.** A ground-plane
homography places an airborne ball past the goal line whether or not it crossed.
Instead the goal mouth (7.32 m × 2.44 m) is projected **into the image** and the
ball is tested for containment, which makes ball height irrelevant.

---

## 5. Measured accuracy — with the limits stated

Scored with `scripts/evaluate_events.py` against hand-written ground truth at
1.5 s tolerance. **Reproducible:**

```bash
python scripts/evaluate_events.py \
    --events <run>/events_with_goals.json \
    --reference docs/references/messi_full_clip.json \
    --tolerance-seconds 1.5
```

### Event detection

| Clip | Class | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| **Messi** (108 s, 29 events) | **Pass** | 23 | 2 | 4 | **0.92** | **0.85** | **0.885** |
| | **Goal** | 1 | 0 | 0 | **1.00** | **1.00** | **1.00** |
| | Shot | 0 | 0 | 1 | — | 0.00 | — |
| | **Overall** | 24 | 2 | 5 | **0.923** | **0.828** | **0.873** |
| **Bundesliga** (30 s, 10 events) | Overall | 5 | 3 | 5 | 0.625 | 0.500 | 0.556 |

The goal was detected at **96.00 s** against an annotated 96.0 s.

**Why the second clip scores lower, and it is not random:** **5 of its 10 events
put the ball in the air**, against 1 of 29 on the Messi clip. Aerial ball contact
is **deliberately disabled** in the event engine, because a ball flying over a
player could not be distinguished from one at his feet. Same system, same switch.

### Component-level

| Component | Metric | Result |
|---|---|---|
| Player detection | mAP@50 | **0.99** |
| Ball detection | frames with ball found | **80–93%** |
| Pitch calibration | frames with valid camera | **100%** / **73.7%** (per clip) |
| Team assignment | tracks correct | **23/25** (colour) vs 22/26 (SigLIP) |
| Identity (good case) | players, team split | **26 players, 13/13**, tracks hold full 30 s |
| Identity (hard case) | tracks vs real players | **109 tracks for ~22 players** |

### Player metrics (only possible after calibration)

| | Bundesliga clip | Messi clip |
|---|---|---|
| Track length | **29.8 s — full clip** | 9–20 s (fragments) |
| Distance per player | **88–111 m** | 29–46 m |
| Top speed | 13.6–29.6 km/h | 18–35 km/h |

---

## 6. Honest limitations — please keep these on the poster

1. **Same-team identity swapping is unsolved.** When two players of the same team
   cross, the tracker cannot tell them apart: it matches on box overlap and
   motion, with no appearance model. This is an **open research problem, not an
   implementation gap** — the published NPSPT benchmark (Huang et al., *Appl.
   Sci.* 2022, 12, 7473) reports its best method reaching **HOTA 50.27 with 130
   identity switches in a two-minute clip**, and concludes it is "far from
   perfect".
2. **Identity fragments.** One player can become several short tracks, which is
   why the Messi clip shows 109 tracks for ~22 players. The backend's
   `PlayerTrackMap` exists to let a human merge them.
3. **Aerial ball contact is disabled**, so headers, volleys and lofted passes are
   missed. This is the largest known accuracy gap and is now measured.
4. **No shot classifier.** Goals are detected; shots that miss are not.
5. **Ground truth is one clip per video, one annotator**, timestamps read to the
   nearest second. These are real measurements, not a benchmark.
6. **Distance undercounts** when calibration coverage is below 100%; values are
   never scaled up to compensate.
7. **`null` in a report means "not produced", not zero.**
8. **PnLCalib is GPL-2.0.** Its obligations trigger on *distributing* software.
   Running it server-side for a SaaS product does not trigger them; shipping an
   installable product would. *(Not legal advice.)*
9. **Camera motion degrades everything.** Quality falls off above roughly
   150 px/s of pan. Fixed wide-angle club cameras are the easiest input.

---

## 7. Where things run

| Component | Runs on |
|---|---|
| Frontend | `localhost:3000` (Next.js dev server) |
| Backend | `localhost:4000` (NestJS) |
| PostgreSQL | Docker container |
| AI worker | Developer machine (NVIDIA GPU, ~0.4 s/frame) **or** Google Colab T4 |
| Model weights | Local `artifacts/`, mirrored to Google Drive for Colab |
| Run outputs | Local `outputs/<run_id>/`, mirrored to Drive |

No cloud hosting. No CI/CD. No queue.

---

## 8. Plan to 30 September

| Priority | Work | Expected effect |
|---|---|---|
| 1 | **Enable aerial ball contact** using the vertical geometry calibration now provides | The largest measured gap; should lift the Bundesliga clip's recall directly |
| 2 | **Assist detection** — link the pass preceding a detected goal | Both halves already detected; it is a join |
| 3 | **Calibration into the main pipeline** as a stage rather than a script | One command produces everything |
| 4 | Shot detection | Currently absent |

**Not attempted before 30 September:** same-team re-identification, cloud
deployment, job queue, real-time processing.

---

## 9. For the poster — the three claims that hold up

1. **Single ordinary camera, no wearables, no multi-camera rig.** Player
   trajectories, team assignment, events and a 2D tactical map from one video.
2. **Pass detection at 92% precision and 85% recall, goal detection exact**,
   measured against hand-written ground truth, reproducible with one command.
3. **Real-world units.** Player positions, heatmaps and distance in **metres** on
   a 105 × 68 m pitch — derived from camera calibration, not pixel heuristics.

And the honest framing that makes those credible: **we state what does not work**
— aerial contact, same-team identity, shots — and we measured it rather than
estimating it.

---

## 10. Files worth showing

| Artefact | Path |
|---|---|
| Annotated match video | `demo/20260910_170402/stats_video.mp4` |
| 2D tactical map | `demo/*/analytics/map2d.mp4` |
| Team heatmaps | `demo/*/analytics/heatmap_team_*.jpg` |
| Goal detected | `demo/20260910_170402/analytics/goal_detected_96.00s.jpg` |
| Calibration proof | `demo/*/analytics/calibration_overlay.jpg` |
| Per-player metres | `demo/*/analytics/player_metrics.json` |
| Ground truth + score | `demo/*/analytics/ground_truth.json`, `measured_score.json` |
| Data contract | `docs/BACKEND_INTEGRATION.md` |
| Full pipeline detail | `docs/AI_PIPELINE_CURRENT_AND_FUTURE.md` |
