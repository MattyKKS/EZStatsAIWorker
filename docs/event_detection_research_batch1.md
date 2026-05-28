# Event Detection Research — Batch 1

**Date:** 2026-05-28
**Focus:** Pass receiver detection, ball-to-player distance measurement, camera perspective distortion, homography-based pitch projection

---

## Phase 1 — Literature & Open-Source Research

### FOOTPASS & the GCN Paper (arXiv:2301.10052)

FOOTPASS models each frame as a graph: ball + all players are nodes, edges connect any pair within a **25-meter real-world distance threshold**. Node features are `(x, y, one-hot team)` in normalized pitch coordinates. The GCN learns which nodes transition into "possession" from frame to frame — when the possession edge changes, a pass is inferred.

**Critical detail:** The GCN paper assumes tracking data is **already in metric pitch coordinates**. It does not solve perspective distortion — it requires a homography as a preprocessing step upstream. Without accurate pitch projection, the 25m edge pruning is meaningless.

### Transformer + Homography (Honda et al., CVPR 2022 CVSports)

State-of-the-art for broadcast video. Pipeline:
1. Detect players and ball with YOLOv8
2. Run homography → project all feet to pitch cm coordinates
3. Feed all 20 player positions + velocities + ball state into a transformer encoder (15–30 frame window)
4. Output: probability distribution over players for "who receives the next pass"

Uses **soft attention weights** learned by the transformer rather than hard distance thresholds. Achieves **75–80% receiver prediction accuracy** on broadcast video.

### PathCRF — Ball-Free Approach (arXiv:2602.12080)

Eliminates ball tracking entirely. Builds a fully-connected graph over all 22 player trajectories, uses a Set Attention encoder + CRF decoder to infer which pair of players is currently in a possession relationship. When the selected edge changes → pass detected.

- **69.6% possession edge accuracy**, **75.7% event F1** on SoccerNet
- Works in image coordinates — no homography needed
- Most robust when ball tracking is unreliable
- Trade-off: can't handle simultaneous ball contacts or headers cleanly

### Camera Perspective & Homography — Consensus

- Standard possession distance threshold in metric space: **2 meters (200 cm)**
- All serious approaches use homography or full camera calibration before any distance measurement
- **TVCalib (2022):** differentiable homography from partial field keypoints — handles broadcast shots where only part of the field is visible
- **BroadTrack (WACV 2025):** adds radial lens distortion correction on top of homography, halves reprojection error
- **Consensus:** measuring distances in pixel space is fundamentally inaccurate for broadcast football; homography is not optional, it's the baseline

### Comparison Table

| Approach | Perspective Handling | Distance Metric | Receiver Accuracy | Broadcast Robustness |
|---|---|---|---|---|
| GCN (Rana 2301.10052) | Assumes pre-corrected input | Euclidean in normalized coords | ~65–70% (implicit) | Moderate |
| Transformer + Homography | Explicit homography correction | Euclidean in pitch space (cm) | **75–80%** | High |
| PathCRF (Ball-Free) | Not needed; works in image space | Implicit / learned via CRF | 69.6% | Very High |
| Simple Euclidean (no correction) | None | Pixel distance | Poor | Low |

### Thresholds Used in Practice

| System | Possession Threshold | Pass Speed | Notes |
|---|---|---|---|
| GCN (Rana) | 25m graph edge | Velocity component | Requires metric coords upstream |
| Honda Transformer | Learned (soft attention) | Learned | No hard threshold |
| PathCRF | Learned CRF transition | Learned | Ball-free |
| General literature | **2m (200cm)** hard threshold | — | Most common baseline |

### Key Papers

- Rana, A. S., et al. (2023). "Event Detection in Football using Graph Convolutional Networks" — arXiv:2301.10052
- Honda, Y., et al. (2022). "Pass Receiver Prediction in Soccer using Video and Players' Trajectories" — CVPR CVSports
- Kim, H., et al. (2025). "PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories" — arXiv:2602.12080
- Theiner, J., & Ewerth, R. (2022). "TVCalib: Camera Calibration for Sports Field Registration in Soccer" — WACV
- Magera, F., et al. (2025). "BroadTrack: Broadcast Camera Tracking for Soccer" — WACV
- Sorano, D., et al. (2020). "Automatic Pass Annotation from Soccer VideoStreams Based on Object Detection and LSTM" — ECML PKDD
- FootAndBall (arXiv:1912.05445), SoccerNet-Tracking (arXiv:2204.06918)

---

## Phase 2 — Current Implementation Analysis

### Critical Finding: Pitch Mode Is Never Active

`pipeline.py:58` calls `detect_events` without `ball_pitch_pos` or `player_pitch_pos`. Both default to `None`, so `use_pitch = False` on every run. All pitch-mode code paths in `events.py` are dead code. Everything runs in pixel space.

`pitch_keypoints.json` IS computed (step 5 of the pipeline) but this happens after the analyze step (step 1), so pitch coords are unavailable when events are detected.

### Distance Calculation — `_closest_player()` (events.py:495–530)

**Step 1:** Extract pixel positions
- Ball: `(center_x, y2)` × `(width, height)` — bottom edge of bbox
- Player: `(center_x, y2)` × `(width, height)` — feet (bottom edge of bbox)

**Step 2:** Raw Euclidean distance in pixels
```
raw = sqrt((ball_cx - player_cx)² + (ball_y2 - player_y2)²)
```

**Step 3:** Perspective normalization
```
normalized = raw × (70.0 / player_height_px)
```
`70px` is the assumed standard player height at broadcast midfield zoom.

**What this corrects and doesn't correct:**
- ✅ Distant players (small bbox) get multiplied up — roughly compensates for depth
- ❌ Uniform scale applied to Euclidean distance — doesn't separately fix horizontal vs vertical distortion
- ❌ Two players level on the pitch but spread horizontally still measured in perspective space
- ❌ Ball bottom-y is not the ground contact point when ball is airborne

**Possession threshold:** `80px` (normalized). Roughly 2–3m depending on field position — correct order of magnitude but inconsistent across the pitch.

### Receiver Detection — IN_FLIGHT Phase (events.py:164–295)

```
each frame during IN_FLIGHT:
    closest = argmin_distance(ball, all_players)
    if closest == candidate_tid:
        candidate_count += 1
    else:
        candidate_tid = closest
        candidate_count = 1    ← hard reset on any switch

    if ball_decelerating AND NOT ball_airborne:
        if candidate_count >= 6:
            → CONFIRMED RECEPTION
```

**Problems:**
1. No trajectory prediction — ball flies near bystanders for most flight; wrong player accumulates count
2. Hard reset on any player switch — one frame near a bystander wipes all evidence for the real receiver
3. `ball_airborne` guard delays confirmation until ball lands — can be too late or never fires
4. No team-weighting or direction filtering — an opponent standing between passer and receiver gets counted first
5. Deceleration gate `speed <= flight_speed × 0.65` can flip on/off with noisy velocity, resetting candidates

### Summary of Known Weaknesses

| Issue | Root Cause | Severity |
|---|---|---|
| Pixel distance is perspective-distorted | No homography projection | High |
| Pitch mode never activates | pipeline.py doesn't populate pitch coord dicts | High |
| Nearest player ≠ intended receiver | No trajectory prediction | High |
| Hard candidate reset | `candidate_count = 1` on any switch | Medium |
| Bystander false receiver | No directional filtering | Medium |
| Airborne confirmation lag | `ball_airborne` guard blocks counting until grounded | Medium |

### What We Already Have (Can Reuse)

- `pitch_keypoints.json` — 17 pitch landmarks in pixel coordinates after step 5
- `stats_video.py` — full working homography pipeline: YOLO pitch model → `ViewTransformer` → `cv2.perspectiveTransform` projecting feet to cm
- `ball_pitch_pos` / `player_pitch_pos` parameters in `detect_events` — fully implemented, just never populated
- `_closest_player` — already handles pitch mode (Euclidean in cm) correctly
- `ball_airborne` pixel fallback in IN_FLIGHT — already implemented and working

---

## Phase 3 — Improvement Plan

Four concrete improvements, ordered by impact/risk ratio. No ML training required.

### Fix 1 — Build Homography from `pitch_keypoints.json`, Re-Run Event Detection (Highest Impact)

**What:**
1. After `detect-pitch-keypoints`, load `pitch_keypoints.json`
2. Build static H matrix: `cv2.findHomography(pixel_pts, cm_pts)` from the 17 keypoints
3. For every frame, project each player's feet `(cx, y2)` through H → `(x_cm, y_cm)`
4. For each frame, project ball bottom `(cx, y2)` through H → `(x_cm, y_cm)` (used when grounded)
5. Populate `ball_pitch_pos` and `player_pitch_pos` dicts
6. Call `detect_events` again with those dicts → overwrite `events.json`

**Why this fixes most problems:**
- 200 cm possession threshold becomes real-world 2 meters, consistent everywhere on pitch
- Player-to-ball distance is Euclidean in cm — no perspective distortion
- The existing `ball_airborne` pixel fallback in IN_FLIGHT already handles the airborne-ball-wrong-projection case

**Complexity:** Low — no new algorithms. H matrix construction already implemented in stats_video.py.

---

### Fix 2 — Trajectory Extrapolation for Receiver Pre-Selection (High Impact)

**What:** At launch, record ball velocity vector `(vx, vy)`. Each frame during flight, project the ball's expected landing position:

```python
frames_remaining = shot_no_catch_f - (current_frame - flight_frame)
predicted_x = ball_cx + vx * frames_remaining * decel_factor
predicted_y = ball_cy + vy * frames_remaining * decel_factor
```

Score candidate receivers by distance to `predicted_landing` rather than current ball position.

**Why:** The ball is near bystanders for most of its flight. Trajectory projection points to where the ball is going, not where it currently is. In pitch cm space (after Fix 1), velocity is stable and trajectory is predictable.

**Complexity:** Medium — requires storing `(vx, vy)` at launch and computing projections per frame.

---

### Fix 3 — Sticky Candidate with Decay (Low Risk, Quick Win)

**What:** Replace the hard `candidate_count = 1` reset with confidence decay:

```python
if candidate_tid != closest.track_id:
    if candidate_count > 2:
        candidate_count = max(1, candidate_count - 2)  # decay, not hard reset
    candidate_tid = closest.track_id
```

**Why:** One frame of the ball near a bystander currently wipes all accumulated evidence for the real receiver. Decay means a 1–2 frame bystander blip only slightly degrades the count.

**Complexity:** 3-line change. Lowest risk.

---

### Fix 4 — Directional Pass Constraint (Medium Impact, Low Risk)

**What:** At launch, record ball direction vector. When evaluating a receiver candidate, check that the candidate is in the forward hemisphere of the pass:

```python
# dot product of (passer→candidate) with (ball launch direction)
dot = vx_launch * (candidate_x - passer_x) + vy_launch * (candidate_y - passer_y)
if dot > 0:  # candidate is ahead in pass direction
    count this candidate
```

**Why:** Eliminates players behind the passer and defenders running toward the ball from false receiver assignment.

**Complexity:** Low — `flight_speed` already stored; just needs direction vector stored at launch too.

---

### Priority Order

| Priority | Fix | Impact | Risk | Effort |
|---|---|---|---|---|
| 1 | Homography from pitch_keypoints → re-run detect_events with pitch coords | Very High | Low | 2–3h |
| 2 | Sticky candidate decay (no hard reset) | Medium | Very Low | 15 min |
| 3 | Directional pass constraint | Medium | Low | 30 min |
| 4 | Trajectory extrapolation for receiver scoring | High | Medium | 2–3h |

**Recommended start:** Fix 1 + Fix 3 together. Fix 1 alone dramatically improves distance accuracy. Fix 3 is trivial with no downside. Then verify with a pipeline run before adding Fix 4.
