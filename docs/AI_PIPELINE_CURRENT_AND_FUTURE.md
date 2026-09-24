# EZStats AI pipeline — what every stage uses, and what replaces it

Written 2026-09-24. Every number here was measured on this repo's runs, not estimated.
Where a claim is not measured it says so.

Read with: `PIPELINE_RECOVERY_PLAN.md` (audit + gates), `PIPELINE_V3_HANDOFF.md` (v3 scope).

---

## Quick verdict

| Stage | State | Blocking anything? |
|---|---|---|
| 1 Video load | fine | no |
| 2 Player detection | **good — do not retrain** | no |
| 3 Ball detection | acceptable | no |
| 4 Tracking | **broken: no appearance model** | same-team ID swaps |
| 5 Appearance embedding | computed, then **discarded** | wasted; it is the fix for 4 |
| 6 Team classification | **fixed** (colour per track) | no |
| 7 **Pitch calibration** | **BROKEN — the blocker** | 8, 9-goal, 10 |
| 8 Coordinate transform | correct code, starved of input | metres / 2D map |
| 9 Events — passes | **working** | no |
| 9 Events — shot/goal | **absent** | needs 7 |
| 10 Report + render | fixed (draw-only) | metrics need 7 |

**Stages 7 → 8 → 9-goal → 10-metrics are one chain.** Calibration is not one of ten
problems; it is the gate in front of every spatial feature the product sells:
2D map, heatmaps, distance in metres, player positions, goal line.

---

## 1. Video loading

- **Tech:** OpenCV `VideoCapture`, `sv.get_video_frames_generator`.
- **Purpose:** decode frames; read fps, resolution, frame count.
- **Demo:** same. **Now:** same.
- **Future:** add scene-cut detection (frame-histogram Bhattacharyya distance,
  prototyped: it found 1 cut in Brighton, 3 in Mason, 2 in Messi) so calibration
  and possession reset at cuts instead of interpolating across them.

## 2. Player / goalkeeper / referee detection

- **Tech:** YOLOv8**m**, 150 epochs, 298 images -> `artifacts/training/player_detector_v2`.
- **Purpose:** per-frame boxes + class.
- **Demo:** yes, v2 (inferred: weights dated 2026-06-08 19:15, run at 22:44, and the
  demo's max raw track id 510 sits in the post-v2 fragmentation regime where the
  old detector gave 1151. The run records no model path — provenance gap, later
  fixed by `run_manifest.json`).
- **Measured:** mAP50 **0.99**, recall 0.98. 12.0 players/frame on Brighton.
- **Future: keep. Do not retrain.** A detector at 0.99 cannot be the cause of ID
  swaps, wrong teams or missing goals. Retraining it is how months were spent
  without moving any measured number.

## 3. Ball detection

- **Tech:** YOLOv8**x** @ imgsz 1280 + `sv.InferenceSlicer` (tiled inference, because the
  ball is a few pixels) + `BallTracker` buffer.
- **Purpose:** ball position per frame.
- **Demo:** yes — the **old** ball model. `ball_detector_v2` was trained and
  **rejected**: better validation metrics, worse events, because looser boxes made the
  ball jitter frame to frame and events mis-attributed.
- **Measured:** Brighton 846/908 = **93%**; Messi 5401/6455 = 84%; 08fd33_4 597/750 = 80%.
- **Future:** keep. Coverage is adequate; the ball is not why events fail.

## 4. Multi-object tracking

- **Tech:** Ultralytics **ByteTrack** — Kalman motion prediction + IoU association, two
  association stages (high-score boxes, then low-score boxes).
- **Purpose:** a persistent `track_id` per player across frames.
- **Demo:** yes. **Now:** same, plus `bytetrack_football_v3.yaml`
  (`track_high_thresh` 0.50, `new_track_thresh` 0.60, `fuse_score` off).
- **Measured:** Brighton minted **1220 ids for ~12 concurrent players**; demo 510.
  v3 config cut minted ids to 263 (Brighton) and 142 (demo).
- **Why it still fails:** ByteTrack has **no appearance model**. It associates on box
  overlap and predicted motion. When two players cross, both hypotheses fit, and it
  cannot tell them apart. This is structural, not a threshold.
- **Future:** **BoT-SORT with ReID + GMC.** `configs/botsort_football.yaml` already
  exists with `with_reid: False` and `gmc_method: sparseOptFlow`. GMC compensates
  camera motion; ReID supplies the missing appearance evidence (see stage 5).
  A previous attempt used OSNet ReID and failed — OSNet is trained on pedestrians in
  varied clothing, and footballers in identical kit are its worst case.

## 5. Appearance embedding

- **Tech:** `prepare-appearance` exports per-track crops; **SigLIP**
  (`google/siglip-base-patch16-224`) maps each crop to a **768-d embedding**.
  This is *not* pixel colour — it is learned semantic features encoding shape,
  texture, pose and background.
- **Purpose today:** feeds team clustering (stage 6) and nothing else.
- **Demo:** yes. **Now:** yes, still runs.
- **Future — the cheapest big win:** these embeddings are **exactly the appearance
  feature stage 4 is missing**. Two same-team players crossing are indistinguishable
  by colour but separable by embedding. We compute them every run and discard them.
  Route them into BoT-SORT ReID instead of OSNet: no new model, no training, and it
  is the only route to fixing same-team ID swaps.

## 6. Team classification

- **Tech (demo, still running):** SigLIP 768-d -> **UMAP** to 3-d (non-linear manifold
  reduction) -> **KMeans(k=2)**. Applied **per frame** in the old renderer.
- **Tech (added 2026-09-10):** jersey colour, **once per track** — top-half crop,
  KMeans(k=2) to split shirt from background, four-corner vote to identify which
  cluster is background, per-track median shirt colour, then KMeans across tracks in a
  saturation-weighted chroma + value space, two largest clusters = the teams.
- **Measured:** SigLIP **22/26** correct. Jersey colour **23/25**, and its two misses
  are genuinely ambiguous (one is the confirmed #3<->#16 ID swap, whose median colour
  really is mixed), whereas SigLIP failed on tracks with an unambiguous green signal.
- **Two departures from the tutorial method, both forced by measurement:**
  raw-BGR KMeans is dominated by brightness, so the dark-kitted officials capture a
  cluster and both kits merge; and k=2 fails because the **kit-to-kit gap is the
  smallest in the frame** (green-white 0.35 vs 0.43 to the red keeper and 0.67 to the
  officials), so the kits are the last thing KMeans separates.
- **Per frame was the real bug.** A player's team never changes during a match;
  deciding it 908 times invites 908 chances to flicker.
- **Future:** keep both. Colour decides (better and ~30x cheaper); SigLIP breaks ties
  and covers kits where colour is ambiguous. Nothing is deleted.

## 7. Pitch calibration — **THE BLOCKER**

- **Tech:** YOLOv8-**pose**, `kpt_shape [32,3]`, one class `pitch`. It must regress all
  32 pitch landmarks as a single object.
- **Purpose:** find pitch landmarks so a homography can be fitted.
- **Demo:** yes, sampled every 60 frames, **one homography reused for the whole clip**.
- **Now:** sampled every 10 frames, per-frame homography (`--per-frame-stride`).
- **Measured — the numbers that matter:**
  - finds a **median of 8-9 of its 32** keypoints
  - valid homography on **2.6%** of sampled frames (Messi), **38%** (08fd33_4)
  - raising tolerance 4px -> 8px takes 08 to **98.7%** but Messi only to 23.9%
  - in the annotated Messi goal window, **0 of ~180 frames** calibrate at 4px, and
    only 12 even at 20px
- **Why it fails:** zoom in and most of the 32 landmarks leave the frame, but the model
  must still predict all of them; the off-screen ones come back low-confidence and are
  filtered. **Calibration is worst exactly when a goal happens**, because that is when
  the camera tightens on the penalty area.
- **Future: PnLCalib** (https://github.com/mguti97/PnLCalib, CVIU paper
  https://arxiv.org/pdf/2404.08401). HRNetv2 encoder-decoder producing heatmaps for
  **keypoints AND field-line extremities**, keypoint set augmented with line
  intersections from a second model, initial pose by RANSAC+DLT, then **PnL refinement
  jointly optimising points and lines**. Lines stay visible when corners do not, which
  is precisely our failure mode. **Pretrained weights are published** (`SV_kp`,
  `SV_lines`) — no training.
  - ⚠️ **Licence GPL-2.0 (copyleft).** Fine for proving capability; shipping it inside a
    sold product obliges releasing source. Decide the commercial route separately:
    comply, reimplement from the paper, or evaluate
    https://github.com/NikolasEnt/soccernet-calibration-sportlight (2023 winner).
  - Plus **optical-flow propagation** between calibrated frames — the SoccerNet 2025
    Game State Reconstruction winner used exactly this for temporal smoothing.

## 8. Coordinate transform

- **Tech:** `cv2.findHomography` (RANSAC) then `perspectiveTransform`. `FieldProjector`
  gates each frame: >= 6 landmarks, >= 6 RANSAC inliers, p90 reprojection <= 4 px, and
  refuses to interpolate across scene cuts.
- **Purpose:** image pixels -> pitch centimetres. **This is what makes metres, heatmaps,
  the 2D map and any goal-line test possible.**
- **Demo:** one matrix for the entire clip. Measured camera drift from the calibration
  frame: demo 434 px median, Messi **3106 px** — more than a frame width, so "pitch cm"
  was fiction while the log printed `PITCH (cm)`.
- **Now:** per-frame, quality-gated, nearest valid frame within 3x the stride.
- **Future:** the code is correct; it is **starved of input by stage 7**. Fix 7 and this
  works as written.

## 9. Event detection

- **Passes / interceptions / touches**
  - **Tech (demo):** hand-tuned finite state machine, ~20 thresholds
    (possession radius, pass speed, direction change, reception deceleration, arc).
  - **Tech (now):** `contacts_v2` — real observations only, player-height-scaled foot
    distances, runner-up ambiguity margin, relative ball/player motion, temporal
    support, ball-coverage gates. Refuses to label a transfer when the team is unknown
    rather than guessing.
  - **Measured:** on reviewed windows, **6/6 passes, precision 1.00, recall 1.00**
    (Messi 4/4 on the Sept run; 08fd33_4 2/2 on the demo run). Zero false positives.
  - Note: the September 08 run scores 1/2 where the **June demo run scores 2/2** with
    the same engine. That gap is upstream tracking/ball coverage, not events.
- **Shot / goal — absent**
  - Nothing produces these classes. Recall on reviewed windows is 0.50 *entirely*
    because of them; passes alone are 1.00.
  - `goal_detection.py` implements the actual Law (ball crosses the goal line between
    the 7.32 m posts, persistently, having approached from inside the field) and
    returns **zero candidates**, because only 237 of 5401 Messi ball observations have
    a quality-gated field position. **The rule is fine; the input is missing.**
  - Honest limit even once calibrated: a homography maps the **ground plane**, so an
    airborne ball projects past the line without crossing it. Goal output must stay a
    reviewable candidate, not an assertion.
  - **Future:** goal-line geometry once stage 7 lands; a trained shot spotter
    (https://github.com/SoccerNet/sn-spotting) for shots on target that do not score.
- **Air balls** — the demo deliberately rejected aerial contacts, and `contacts_v2`
  disables unvalidated aerial contacts. This is a known, deliberate omission, not a bug.

## 10. Report and render

- **Tech:** `match_report.json` -> `report_cleanup` (tracklet merge, referee cap,
  colour team correction, ID-swap surgery, frame-based possession) -> renderer.
- **Demo:** the video ran a **second** detection + tracking pass and re-classified teams
  with SigLIP per frame, then Hungarian-matched its own detections back to `tracks.json`
  on an 80 px threshold. Two ID spaces glued by nearest-point matching, ambiguous
  exactly when players overlap.
- **Now:** `render-stats-video --from-tracks` draws the pipeline's own tracks. The drawn
  id **is** the report id by construction. Benchmark renders in **19 s** instead of
  ~15 min.
- **Future:** heatmaps, per-player distance in metres, 2D map — **all gated on stage 7.**

---

## Measurement discipline (the actual lesson)

From March to September every change was judged by eye. The first measured score was
taken on 2026-09-23. That is the root cause of the lost months — not difficulty.

- `scripts/evaluate_events.py` scores events against reviewed windows and **refuses to
  run on draft annotations**.
- References live in `docs/references/`.
- Rules: annotate before tuning; hold out windows; include windows where nothing
  happens; never use the clip title or a desired pass total as a label.
