# Single-camera pipeline recovery plan

Status: September 24 audit started. Not a claim of corrected match analytics.
Primary regression clips: `08fd33_4.mp4` and `leo_messi_30pass.mp4`.
Preserve the demo, original runs, weights and config hashes. No retraining yet.

## Findings verified in code and artifacts

1. The legacy event FSM can change owners without emitting a pass. The opt-in
   contact engine recovers some passes, but has no shot/goal/assist classifier.
   Neither pipeline is currently an accurate complete event solution.
2. `artifacts/training/event_spotter_v1/model.pt` has an 18-class output head.
   Current `event_spotter.py` builds a 14-class head. Its directory has no ordered
   class metadata. Tensor inspection used `torch.load(weights_only=True)`.
   Matching tensor dimensions alone would NOT establish matching class semantics.
   Git history at `89907c8` contains an 18-class candidate mapping, but also
   explicitly documents substituting ResNet18 features for training-time
   ResNet152+PCA features. Equal 512-dimensional vectors do not make these
   feature spaces interchangeable. This historical code is evidence to investigate,
   not proof of the current checkpoint's training provenance. Restore the actual
   feature contract and ordered labels before evaluating or retraining that model.
3. Event pitch projection abandons the whole clip if any ball frame lacks usable
   calibration. The radar uses a separate projection path and previously retained
   its last rendered map indefinitely. A plausible radar is not calibrated truth.
4. Keeper-team proximity and overlap foot associations remain unvalidated. Wrong
   team/identity upstream can turn a real pass into an apparent interception.
5. Previous 15-to-23 Messi pass counts were candidate counts, not precision/recall.
   There is no fully reviewed timestamped reference in this audit yet.

## Stage 1: establish what is wrong

- Freeze model/config/source-video versions. Start from saved tracks, not another
  expensive detection run. Do not change detectors and events in the same trial.
- Annotate complete short windows in both primary clips: passes, one-touch play,
  overlap, goalkeeper distribution, missing ball, shot and the Messi goal sequence.
  Include negative windows where no shot/goal occurs. Mark ambiguous periods as
  excluded; do not label missing evidence as negative evidence.
- Define timestamp convention before scoring: the current contact engine emits
  reception time for passes, not kick time. Annotate receptions when evaluating
  that output; separately record departures for the later event schema.
- Evaluate event class/timing and actor/receiver identity separately. Use one-to-one
  matching; repeated detections of one pass are false positives. Wrong shot labels
  must count as both a false shot and a missed true event, not a correct detection.
- Split development and held-out windows from BOTH videos before threshold tuning.
  Reserve Brighton/Mason and additional phone footage for transfer validation.

Implemented starting point: `scripts/evaluate_events.py`. It refuses draft or
unbounded annotations and reports precision, recall, F1, mismatches and coverage.
No accuracy percentage is available until reference annotations exist.

## Stage 2: trustworthy 2D field positions

- One shared camera/calibration stream for analytics and radar. Retain valid
  intervals rather than silently discarding calibration for the whole clip.
- Validate landmark IDs, inlier distribution, reprojection error and temporal
  continuity. Reset on cuts; represent unavailable positions as unknown. The first
  patch expires distant radar keypoints and stops indefinite old-map display.
- Use feet/ground-contact points for players. For homogeneous image point p,
  q = H p and field position = (q_x/q_z, q_y/q_z); reject unstable denominators
  and implausible coordinates. Keep explicit metres/centimetres/pixel units.
- Compute ground velocities as (X_t - X_previous) / elapsed_seconds only inside
  valid calibrated intervals. Do not convert panning pixel movement to metres.
- Inspect phone lens distortion and actual pitch dimensions. A generic full-size
  field template is not metric truth for an unknown training pitch.
- Do not project an airborne ball onto the grass and call that its true location.
  Unknown ball height and occlusion limit single-view reconstruction. A 2D pitch
  transform alone cannot establish a ball crossing under the crossbar.

Gate: manually checked landmarks/player foot positions on held-out frames, plus
reported calibration coverage. Numerical error thresholds depend on footage and
reference quality; do not invent an accuracy claim from RANSAC inlier counts.

## Stage 3: possession and identity

- Measure false ball detections, especially shoes; gaps; identity switches; and
  team mistakes separately. Detector confidence is not measured correctness.
- Preserve observed versus inferred ball samples. Inferred samples cannot alone
  confirm a touch. Handle camera cuts before carrying ownership forward.
- Combine close foot contact, duration, ambiguity, relative movement and an
  observable trajectory change. Nearest player alone does not establish control.
- Separate raw track IDs from persistent player identities. Jersey team colours
  identify teams, not individual same-team players. Do not cap IDs at 11 to hide
  fragmentation or force assignments during occlusion.
- Resolve goalkeeper teams with evidence or leave unknown, not nearest-team median.
- Include one-touch passes, return passes, fly-bys and overlapping challenges in
  regression cases. Keep uncertain transfers outside confirmed interception stats.

## Stage 4: football event semantics

- Pass: intentional distribution to a teammate, with distinct departure and
  reception evidence. A tackle, deflection, clearance and shot are not synonyms.
- Shot: goal-directed attempt with suitable scene/trajectory evidence. Speed and
  a receiver timeout are insufficient. Include blocked and off-target examples.
- Goal: separately confirm the goal outcome; do not promote every shot or ball
  disappearance to a goal. Under occlusion, return uncertainty rather than certainty.
- Assist: attribute the qualifying prior teammate action only after a goal and
  possession chain are established. Record the chosen statistical definition.
- Recover the 18-class checkpoint's original training config/ordered labels and
  evaluate it as a separate branch. Never guess the labels, resize its head, or
  assume the BAS-2025 class list applies. Its existing 2 fps temporal sampling also
  needs evaluation for closely spaced actions. Video-based spotting needs source
  frames, but does NOT require rerendering or rerunning player/ball detection.
- Fuse event evidence only after each source is evaluated. Do not merge two lists
  merely to increase the total. Report rejected and unresolved candidates.

## Stage 5: release and retraining decision

Compare old/new predictions on identical reviewed windows, at predeclared timing
tolerances (initially 0.5 and 1.0 seconds). Report per-class precision/recall and
actor correctness, ID switches, ball false positives and calibration error.
No change is promoted merely because total events rise. Both primary clips must
improve on agreed metrics without hiding regressions, then test other footage.

Only consider detector retraining if reviewed boxes demonstrate detector errors
that materially limit the result after association/configuration fixes. Evaluate
existing weights first. A separately trained action spotter is a different task
from retraining YOLO boxes. Any training proposal needs a held-out comparison;
no training run can guarantee an improvement in advance.

## Primary references

- IFAB Law 10 (goal outcome):
  https://www.theifab.com/laws/latest/determining-the-outcome-of-a-match/
- Opta event/assist definitions:
  https://www.statsperform.com/opta-event-definitions/
- SoccerNet game-state reconstruction (single moving camera, calibration,
  roles, identity and minimap evaluation):
  https://github.com/SoccerNet/sn-gamestate
- SoccerNet action/ball-action spotting:
  https://github.com/SoccerNet/sn-spotting

Video games can inform presentation, but their internal simulation state does
not solve inference from a phone recording. Use real-footage research for that.

## Evaluation command

Prepare a reviewed reference following the draft JSON template in
`docs/event_reference_template.json`. Actor IDs must refer to the frozen run's ID
space for `--match-actors`; stable real-player labels require an explicit mapping.
Reference windows are half-open [start, end) seconds on the original clip.

```text
python scripts/evaluate_events.py --events outputs/<run>/events.json --reference <reviewed-reference.json> --tolerance-seconds 0.5 --output outputs/<new-evaluation.json>
```

This command needs neither GPU nor video rendering. It does not automatically
create ground truth. Existing output files are not overwritten.

First-patch validation: 154 tests passed, with two existing Windows sklearn/joblib
warnings. No new detection run, video render, training or accuracy measurement was
performed. Changes are local; this stage has not been published to Colab.

## Saved-run recovery implementation

`contacts_v2` is now selectable in event replay. V1 and the legacy full pipeline
remain available. V2 adds two evidence paths without changing detector weights,
observations, track IDs or teams:

- A short close contact followed by multiple observations of player/ball co-motion
  can establish a running reception without requiring the ball to stop or turn.
- A ball reacquired near its previously established owner after a brief detector
  dropout can bound departure later than the old contact end. Missing time during
  a dribble is no longer automatically treated as all flight time. This requires
  real, unambiguous near-owner observations; predicted/interpolated balls cannot
  establish reacquisition.

These paths remain heuristic and are isolated behind the V2 option. Stationary
fly-bys, cuts and unsupported synthetic receptions are covered by tests. They
do not establish complete football semantics under arbitrary occlusion.

Observed local results on preserved saved runs:
- Messi: 25 pass candidates plus 2 uncertain transfers (previous V1: 23 passes).
  The recovered passes near 49.4 and 92.65 seconds match the source-frame review.
- 08: same event types, times and actors as V1 (2 pass candidates, 2 uncertain
  transfers). This is not improvement over the best demo; the keeper-team error
  still causes a missed backpass.
- Brighton: 4 pass candidates, unchanged count; one reception moved from 8.80 to
  8.64 seconds. No independent annotation validates that timing change yet.

Development references are in `docs/references`. They explicitly disclose
assistant review of sampled source frames and approximate timestamps. They are
not independent human-labelled benchmarks. On just the two declared Messi
windows (four passes, one shot, one goal), legacy matches 3 passes, V1 matches 2,
and V2 matches 4. All three still miss the shot and goal. Do not report 100% pass
accuracy for the whole video, or 30 passes, from this small diagnostic.

`--export-pitch` optionally writes `field_positions.json` from saved keypoints,
with per-sample RANSAC/reprojection checks, bounded interpolation, cut boundaries
and explicit missing positions. Unlike the old all-or-nothing event projection,
it retains valid intervals. It does NOT yet replace the legacy radar/event path.
OpenCV and the existing sports package are needed for this optional export.

Using the existing library's field template and a 4-pixel RANSAC threshold, the
quality-gated player projection coverage was only about 5.3% for Messi and 54.0%
for 08. Coverage is not position accuracy. The library defaults to a 120x70m
template with a 20.15m penalty-box length; do not call those measured dimensions.
IFAB's standard penalty-area depth is 16.5m:
https://www.theifab.com/laws/latest/the-field-of-play/
The export labels template units and unverified dimensions explicitly. Resolving
template/model landmark geometry and testing calibration remains necessary.

Validation: 163 tests pass (two existing Windows sklearn/joblib warnings).
No full detection run or video render was needed. No trained goal model was
enabled, and no goal or assist was inserted into automatic outputs. A separately
labelled review/correction workflow has been proposed to the user for the deadline,
but has not been enabled or assumed approved.

Event-only rerun in an updated Colab checkout:

```text
python -u scripts/replay_events.py --run-dir /content/drive/MyDrive/ezstats/outputs/20260910_170402 --event-engine contacts_v2 --skip-video --copy-to /content/drive/MyDrive/ezstats/outputs
```

The existing notebook's E0-E3 cells select V2 and still skip rendering. Change
`EVENT_ENGINE` to `contacts_v1` for comparison. Full inference cells are unchanged.
