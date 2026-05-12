# ⚽ EZStats — Living Product Plan

> ⚠️ **MOVED.** This file is no longer updated.
> The single source of truth is now **`docs/product_plan.ipynb`** — open that file instead.

---

## 📋 Session Rules

| # | Rule |
|---|------|
| 1 | **Read this file first** every Claude session before doing anything |
| 2 | **Finish one step fully** (test it, verify it) before starting the next |
| 3 | **Stuck on a step?** Diagnose and fix it — never skip ahead |
| 4 | **End of session** — update checkboxes and statuses here |

---

## 🎯 Where We Are Right Now

```
Phase 0 ████████░░  75%
  ✅ 0-A  Run scripts fixed (no broken model)
  ✅ 0-C  match_report.json + viewer.html built
  ⏸ 0-B  event_spotter.py rewrite  ← waiting for training overnight
  
NEXT ACTION:  Run .\run_full_pipeline.ps1  →  open outputs/viewer.html
```

---

## 📊 Current Status vs Footovision

| Feature | Footovision | EZStats | Status |
|:--------|:-----------:|:-------:|:------:|
| Player + ball + referee detection | ✅ | works but misses frames, new videos | ⚠️ |
| Homography / top-down pitch | ✅ | not 100% — bad keypoints warp pitch | ⚠️ |
| Team classification | ✅ | breaks on ID re-assign, some frames wrong | ⚠️ |
| Voronoi territorial control | ✅ | wrong — depends on team classification | ⚠️ |
| Basic possession stats | ✅ | wrong — depends on untested events | ⚠️ |
| Three-phase event detection | ✅ | rewritten, **not tested yet** | ⚠️ |
| 9-class Transformer (PASS/SHOT/CROSS…) | ✅ | **training overnight** in Colab | 🔄 |
| Player heatmaps | ✅ | not built | ❌ |
| Per-player speed & distance (meters) | ✅ | pixel hack only — wrong numbers | ❌ |
| Event highlight clips | ✅ | not built | ❌ |
| Pass network | ✅ | not built | ❌ |
| Debug viewer / match_report.json | — | **built tonight** | ✅ |
| Cloud API | ✅ | offline only | ❌ |
| Formation detection | ✅ | not built | ❌ |
| xG model | ✅ | not built | ❌ |
| Real-time processing | ✅ | ~20 min / 30s on CPU | ❌ |
| Mobile dashboard | ✅ | not built | ❌ |
| **Affordable SE Asia pricing** | ❌ enterprise | **our core edge** | ✅ |

---

## 🚀 Phase 0 — Fix Before Any Run

### 0-A · Test Rule-Based Events

- [x] Remove broken `--event-model-name` from `run_full_pipeline.ps1` + `run_new_video.ps1`
- [ ] **Run** `.\run_full_pipeline.ps1` → check `outputs/<timestamp>/events.json`
- [ ] Manually count passes in video, compare to output (target: within 25% of truth)
- [ ] Open `outputs/viewer.html` in browser → load `match_report.json` → verify stats look reasonable

---

### 0-B · Rewrite event_spotter.py for PCBAS-2026 ⏸ WAITING FOR TRAINING

> Training notebook: `docs/train_pcbas2026_player.ipynb`
> Download when done: Colab Drive → `artifacts/training/event_spotter_pcbas2026/model.pt`

**What's wrong with the current code vs what it should be:**

| | ❌ Current | ✅ Should Be |
|---|---|---|
| Architecture | Bi-LSTM | Transformer Encoder (4-layer, 8-head) |
| Classes | 14 BAS-2025 | **9 PCBAS-2026** (different order!) |
| Window | 15 frames | **25 frames** |
| Inference | Global frame features | **Per-player crop features** |
| Output head | Single (action) | **Dual (action + team)** |
| Save path | `event_spotter_bas2025` | **`event_spotter_pcbas2026`** |

**Correct class list (order matters for loading weights):**
```python
BALL_ACTION_CLASSES = [
    'background',  # 0
    'DRIVE',       # 1
    'PASS',        # 2
    'CROSS',       # 3
    'SHOT',        # 4
    'HEADER',      # 5
    'THROW IN',    # 6
    'TACKLE',      # 7
    'BLOCK',       # 8
]
```

**Correct architecture:**
```python
self.proj        = nn.Linear(512, 512)
self.pe          = PositionalEncoding(512, dropout=0.1)
self.encoder     = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(512, nhead=8, dim_feedforward=2048,
                               dropout=0.1, batch_first=True, norm_first=True), 4
)
self.shared      = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1))
self.action_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 9))
self.team_head   = nn.Linear(256, 3)   # 0=background, 1=left-team, 2=right-team
```

**Checklist (do after model.pt is downloaded):**
- [ ] Rewrite `_EventSpotterModel` to Transformer above
- [ ] Update `BALL_ACTION_CLASSES` → 9 PCBAS-2026 classes
- [ ] Update `WINDOW = 25`
- [ ] Update `_CLASS_TO_EVENT` + `_CLASS_THRESHOLDS` for 9 new classes
- [ ] Rewrite inference to be player-centric (per-player crop features, not frame features)
- [ ] Update `fuse_events()` class names
- [ ] Uncomment `--event-model-name` in `run_full_pipeline.ps1` + `run_new_video.ps1`

---

### 0-C · Debug Dashboard

- [x] `outputs/<timestamp>/match_report.json` written after every pipeline run
- [x] `outputs/viewer.html` — open in browser, load JSON, see all stats visually
- [ ] Run pipeline → open viewer → compare against video manually

---

## 🔧 Phase 1 — Make Existing Features Actually Work
*Target: Week 1–2*

### 🔍 Detection Quality Gate
> **Do this first.** Current detector is YOLOv8n trained on only 298 images of Bundesliga footage.
> Everything else depends on detection being reliable.

- [ ] Run on `08fd33_4.mp4` and `19PassesAndMasonGoal.mp4`
- [ ] Players detected consistently — no player vanishing for >10 consecutive frames
- [ ] Ball detected in ≥70% of frames
- [ ] All 22 players on pitch detected at same time

> **If gate fails → retrain detector first:**
> - Annotate frames from failing video in Roboflow format
> - Upgrade `yolov8n` → `yolov8s` for better accuracy
> - Target: 1000+ images, 4 classes (player / goalkeeper / referee / ball)
> - `yolo detect train model=yolov8s.pt data=data/datasets/roboflow/detector/data.yaml epochs=100`

---

### 🎯 Event Detection — Validate

- [ ] Run `.\run_full_pipeline.ps1` (rule-based, no LSTM)
- [ ] Count passes manually in video vs `events.json` — target within 25%
- [ ] No false shots from clearances or ball rolling out
- [ ] Plumb pitch coordinates into `detect_events()`:
  - Accumulate `ball_pitch_pos[frame] = (x_cm, y_cm)` in `stats_video.py`
  - Accumulate `player_pitch_pos[frame][tid] = (x_cm, y_cm)` in `stats_video.py`
  - Pass both into `detect_events()` via `pipeline.py`

---

### 🗺️ Homography — Make It Stable

- [ ] Fix H averaging in `stats_video.py` — compute `np.mean(list(H_deque), axis=0)` (currently just uses last H)
- [ ] Add H quality check — `abs(np.linalg.det(H)) > 0.1` before accepting
- [ ] Clip pitch projections to `[0, 10500] × [0, 6800]` cm after `perspectiveTransform`
- [ ] Visual check: Voronoi on `08fd33_4.mp4` looks geometrically correct

---

### 👕 Team Classification — Reduce Misassignment

- [ ] Fix GK exclusion — filter by pitch position (closest to goal post) not just detection label
- [ ] Add UMAP guard — if `n_samples < 20`, fall back to cosine similarity clustering
- [ ] Add temporal team lock — once a `track_id` is assigned a team, never flip it mid-track
- [ ] Visual check: team colors stable for ≥90% of frames on both test videos

---

### 📈 Possession Stats — Real Numbers

- [ ] Replace pixel distance in `stats.py` with pitch-coordinate distance in meters
- [ ] Validate: possession % on `19PassesAndMasonGoal.mp4` should not be 50-50

---

## ✨ Phase 2 — First Sellable Features
*Target: Week 3–4*

### 🌡️ Player Heatmaps
- [ ] Accumulate `(x_cm, y_cm)` per `tracker_id` in render loop
- [ ] New: `src/ez_worker/analytics/heatmap.py` — Gaussian KDE on 105×68 pitch grid
- [ ] Output PNG per player + team aggregate
- [ ] Add heatmap paths to `match_report.json` + show in `viewer.html`

### 🏃 Per-Player Speed & Distance
- [ ] Distance = sum of Euclidean distance in cm between frames → meters
- [ ] Max speed = `max(Δcm / Δseconds)`, capped at 1200 cm/s (43 km/h)
- [ ] Add to `TrackStats` and `match_report.json`

### 🎬 Event Highlight Clips
- [ ] New: `src/ez_worker/io/clip_exporter.py`
- [ ] On SHOT / GOAL / CORNER / FREE KICK — cut `[event_frame - 2s : event_frame + 3s]`
- [ ] Named: `GOAL_45m23s.mp4`, `SHOT_32m11s.mp4`
- [ ] Add clip paths to events in `match_report.json`

### 🕸️ Pass Network
- [ ] New: `src/ez_worker/analytics/pass_network.py`
- [ ] Directed graph: `player_A → player_B` count from PASS events
- [ ] Output JSON edge list + static pitch diagram image
- [ ] Show in `viewer.html`

---

## 📱 Phase 3 — Friend's Frontend Ready
*Target: Week 5–6*

- [ ] Hand `match_report.json` schema to frontend friend
- [ ] Test on `BrightonGoal.mp4` — first untested video — no crashes, goal detected
- [ ] Heatmaps + clips + pass network all working end-to-end

---

## ☁️ Phase 4 — Cloud API
*Target: Week 7–8*

- [ ] New: `src/ez_worker/api/main.py` — FastAPI
- [ ] `POST /analyze` → upload video → `job_id`
- [ ] `GET /job/{id}` → status + download links
- [ ] Deploy on RunPod / Lambda Labs GPU — 90-min match in < 30 min
- [ ] Simple web UI: drag-and-drop → progress bar → download results

---

## 📐 Phase 5 — Analytics Gap (Month 2)

- [ ] **xG model** — shot `(x_cm, y_cm)` + event class → logistic regression on SoccerNet data
- [ ] **Formation detection** — cluster positions at kickoff/set pieces → 4-4-2 / 4-3-3 etc.
- [ ] **Pressing intensity** — count players within 1000 cm of ball carrier per frame

---

## 🌏 Phase 6 — SE Asia Market Lock-In (Month 3+)

- [ ] Collect 10+ hours Myanmar National League / Thai League footage from clubs
- [ ] Fine-tune YOLO detector on local footage (different jerseys, pitch, lighting)
- [ ] Mobile-responsive dashboard (React Native or PWA)
- [ ] Pricing: **$15 / match** or **$80 / month** per club
- [ ] Onboard first 5 pilot clubs — free trial in exchange for video footage
- [ ] Localization: Burmese + Thai UI

---

## 🗂️ Key Files Reference

| File | Pending Work |
|:-----|:-------------|
| `src/ez_worker/analytics/event_spotter.py` | Full rewrite → PCBAS-2026 Transformer (Phase 0-B) |
| `src/ez_worker/io/stats_video.py` | Accumulate pitch coords; fix H averaging |
| `src/ez_worker/analytics/stats.py` | Distance in real meters |
| `src/ez_worker/spatial/view_transformer.py` | H quality check + projection bounds clamp |
| `src/ez_worker/appearance/team_assignment.py` | GK exclusion fix; UMAP guard |
| `src/ez_worker/outputs/writer.py` | ✅ match_report.json done |
| `run_full_pipeline.ps1` + `run_new_video.ps1` | Uncomment model path after training |
| `outputs/viewer.html` | ✅ Built |
| `src/ez_worker/analytics/heatmap.py` | New — Phase 2 |
| `src/ez_worker/io/clip_exporter.py` | New — Phase 2 |
| `src/ez_worker/analytics/pass_network.py` | New — Phase 2 |
| `src/ez_worker/api/main.py` | New — Phase 4 |
