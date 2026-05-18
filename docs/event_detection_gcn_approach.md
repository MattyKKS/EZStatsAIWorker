# Event Detection — GCN-Inspired Spatial Graph Approach

## Paper Reference
Rana, A.S.S. (2021). "Event Detection in Football using Graph Convolutional Networks"
arXiv:2301.10052v1

---

## What the Paper Does

Each video frame is modelled as a **ball-player graph**:
- **Nodes**: ball + all players on pitch
- **Node features**: `(x, y, one_hot_team)` — normalized pitch position + team label
- **Edges**: drawn between any two nodes within 25m real-world distance
- **GCN**: 2 graph convolutional layers (64 → 64 hidden) + linear(32) per-frame embedding
- **Temporal pooling**: NetVLAD++ — separate past (before event) and future (after event) context windows
- **Output**: multi-label event classifier (12 classes: pass, shot, goal, corner, throw-in, offside, foul, etc.)
- Best result: NetVLAD++ with T=10s window → 31.3 mAP

---

## Our Rule-Based Adaptation (No Training Required)

We adapt the key spatial graph ideas into the existing `events.py` state machine.

| GCN Paper Concept | EZStats Rule-Based Adaptation |
|---|---|
| Ball-player edges by distance < 25m | Per-frame: compute N nearest players sorted by real-world distance |
| Node features include team one-hot | Already have `team_id` per TrackObservation |
| Graph neighborhood = spatial context | Use 2 nearest same-team + 1 nearest opponent around ball |
| Temporal window (10s past+future) | Confirmation window: `pass_min_flight_frames` before accepting reception |
| Per-class NMS after predictions | `_dedupe_dense_events` with `min_frame_gap=6` |

---

## Specific Changes Made

### 1. Self-Reception Prevention
**Problem**: Ball bounces back to original owner → state machine emits A→A "pass"  
**Fix**: In `IN_FLIGHT` phase, if `closest.track_id == owner_tid` → re-enter `POSSESSED` without emitting any event

### 2. Minimum Flight Frames (`pass_min_flight_frames = 3`)
**Problem**: Ball leaves player foot for 1-2 frames (dribble, slight toe-poke) → triggers IN_FLIGHT → immediately "received" = false pass  
**Fix**: Require ball to have been in flight for at least `pass_min_flight_frames` (default 3 = 0.12s at 25fps) before accepting any reception

### 3. Spatial Graph Context (nearest players around ball)
**New helper `_nearest_players`**: Returns all players sorted by distance from ball  
**Usage**: 
- In POSSESSED: nearest same-team player = likely pass target for context
- In IN_FLIGHT: second nearest player checked if nearest == owner
- Future: pressing intensity = nearest opponent distance

### 4. Existing Improvements (already in code)
- **High-arc detection** (`_is_high_arc`): clearance/long_ball when ball follows parabolic arc > 4% frame height
- **Direction change** for slow direct-possession passes (> 35°)
- **Pressing disambiguation**: if owner still within `poss_dist`, nearest opponent is a presser not receiver
- **Box-normalized distance**: corrects for camera zoom/pan (70px reference player height)

---

## Parameter Reference

| Parameter | Default | Description |
|---|---|---|
| `possession_distance_threshold_px` | 50 | Ball-player ownership radius (pixels, normalized by player height) |
| `possession_distance_threshold_cm` | 200 | Ball-player ownership radius in cm (when pitch coords available) |
| `possession_min_seconds` | 0.15 | Seconds near ball to confirm ownership |
| `pass_min_speed_px_per_s` | 200 | Min ball speed to enter IN_FLIGHT (pixel fallback) |
| `pass_min_speed_cms` | 250 | Min ball speed to enter IN_FLIGHT in cm/s |
| `pass_min_flight_frames` | 3 | Min frames ball must be in flight before reception accepted |
| `ball_direction_change_min_deg` | 35 | Direction change for slow direct-possession pass detection |
| `shot_min_speed_px_per_s` | 600 | Min speed for shot classification (pixel fallback) |
| `shot_no_catch_seconds` | 1.0 | No reception within this time = shot |
| `clearance_min_flight_seconds` | 1.2 | Flight > this with arc = clearance/long_ball |
| `clearance_min_arc_frac` | 0.04 | Ball y-range > 4% frame height = high arc |

---

## Future: Full GCN Training Path

If rule-based approach doesn't reach acceptable accuracy:

1. **Data gathering**: Annotate events on existing video frames (pass, shot, clearance, interception)
   - Use viewer.html to mark events manually
   - Aim for 500+ annotations across 3+ videos
2. **Synthetic labels**: If not enough real labels, use rule-based detections + manual correction as silver labels
3. **Architecture** (from paper):
   ```python
   # Per-frame graph embedding
   GCN(64) → BatchNorm → ReLU → GCN(64) → BatchNorm → ReLU → Linear(32) → readout (avg)
   # Temporal pooling
   NetVLAD++(K=64, T_past=5s, T_future=5s)
   # Classifier
   Linear(N_events) → Sigmoid  # multi-label
   ```
4. **Input features per node**: `[x_norm, y_norm, team_0, team_1, ball]` (5-dim)
5. **Training**: Adam lr=1e-3, ReduceLROnPlateau, ~100 epochs
6. Deploy to `event_spotter.py`, wire into pipeline

---

## Test Checklist

- [ ] Run `.\run_full_pipeline.ps1` on `08fd33_4.mp4` → events.json has < 20 passes for 30s clip
- [ ] No A→A self-reception events in events.json
- [ ] No two identical pass events within 3 frames
- [ ] Run on `19PassesAndMasonGoal.mp4` → manually count 19 passes in video → events.json within 25% of truth
- [ ] Mason goal → `shot_attempt` fires at correct timestamp
- [ ] GK long ball → `clearance` or `long_ball`, not `shot_attempt`
