# Pipeline v3 handoff

## Goal and scope

Produce a usable stats video with consistent identities, stable teams and credible
events using the existing player, ball and pitch models. Run Messi first, then
08fd33_4, BrightonGoal and 19PassesAndMasonGoal. No backend work or retraining.
85-90% is the user's acceptance target, not an achieved measurement.

## Entry points

Notebook startup fix: run cells execute the logging helper with runpy inside
the kernel, rather than inheriting notebook OS file descriptors through an
outer subprocess.run. Setup subprocess output is explicitly streamed too.
Cell 5 copies only the three required weights. The Colab helper prints the GPU
name, requires CUDA, and passes --device 0 for player and ball inference.

- `docs/run_on_colab_v3.ipynb`: setup, then four independent video cells.
- `python scripts/run_colab_clip.py leo_messi_30pass.mp4`: run one clip and save
  its output/log to mounted Drive. Copies the video to runtime storage first.
- `python run_pipeline_v3.py --video data/raw/leo_messi_30pass.mp4`: portable
  runner, writes local outputs only. Reads actual video FPS.
- `python run_pipeline_v3.py --resume outputs/<run>`: redo finalization/rendering
  from preserved raw tracks/crops. This overwrites final outputs in that run.

The original v2 runner and demo files remain available. Shared bug fixes affect
v2 too; use commit 720f51a to reproduce the code before this audit.

## Evidence from the September 10 audit

Compared Drive `ezstats/outputs/20260910_154501` with
`demo/20260608_224414`. A local working copy of the Drive output is in
`outputs/audit_20260910_154501`; its derived outputs have been recomputed during
the audit. The Drive original and demo were not modified.

- Demo merged report: 26 IDs (13/13), 8 events. This is the user's visual
  benchmark, not a fully labelled ground-truth dataset. A sampled demo frame
  also labels a green-shirted player REF.
- Original Colab merged report: 31 IDs (16/15), 10 events, including clearances
  at 16.40 and 16.44 seconds. Three report/video team disagreements (4, 9, 59).
- Video ignored report-only ID merges, swaps and event corrections.
- Team colour overrides happened after event detection, leaving stale teams in
  event classifications and stats.
- Provider stores referee as label=player/source_label=referee; new renderer
  ignored source_label, drawing officials as team players.
- Upstream proximity-only merging joined fragments before seeing jersey/team
  evidence. ByteTrack had no appearance or camera-motion compensation.
- Missing tracker IDs were replaced with detection array indexes.
- Ball interpolation imposed a minimum 16-frame gap and then filled up to
  32 frames despite the configured limit. Synthetic balls were unmarked.
- Reception counters transferred evidence between different candidates; a
  one-frame close approach could bypass reception confirmation.
- Frame zero was treated as missing in possession/launch timestamps.
- Airborne heuristic averaged feet from players at different field depths.
- Radar passed a (1,2) ball array into an API expecting (2,), suppressing radar
  through a swallowed exception.
- Video showed whole-clip event counts from frame zero and used a constant
  pixel-to-kilometre conversion despite camera motion.

## Implemented flow

Existing detectors -> BoT-SORT native ReID + sparse optical-flow camera motion
-> bounded ball cleanup -> stable detector-role votes -> duration-weighted
two-kit colour clustering excluding officials/keepers -> conservative short
fragment reconnection -> time-varying pitch projection -> events -> stats/report
-> draw those exact tracks/events. No separate report_cleanup pass.

Fragment reconnection requires same role/team, compatible colour, a short
nonoverlapping gap, predicted position within 0.75 player heights, and a clear
alternative-match margin in both directions. No hard 11-player cap.

Final reports and renderer share IDs and events. Interpolated balls cannot
independently confirm contact. V3 disables unvalidated aerial contacts. Report
possession uses confirmed ownership frames, with coverage recorded. Overlay
counts visible players and accumulates event counts over time; uncalibrated
kilometres are omitted. Models/config/package versions are recorded in a manifest.

## Validation and limitations

Full detection, pitch calibration and rendering completed locally for the
750-frame benchmark: `outputs/20260910_233153`. Finalization was rerun as the
event fixes were completed. This candidate still has 31 player tracks and
incorrect/missing events; it has NOT demonstrated better overall accuracy than
the demo. Do not present code tests or fewer IDs as tracking accuracy.

The next external validation is Messi in Colab. Same-team identity switches,
long absences, airborne ball reconstruction, automatic goals, crosses and
assists remain unresolved. A timeout-derived shot is not proof of a goal.
Unknown teams must not be interpreted as successful same-team passes.

Use manually labelled event times/actors and sampled player identities to
measure event precision/recall, team accuracy, ID switches and fragmentation.
Compare all clips with the same model hashes/config, retaining raw artifacts.

## Research used

- Ultralytics tracking and implementation documentation:
  https://docs.ultralytics.com/modes/track/
  https://docs.ultralytics.com/reference/trackers/byte_tracker/
  https://docs.ultralytics.com/reference/trackers/bot_sort/
  BoT-SORT adds camera compensation and optional appearance matching. The older
  tracker YAML's claim that fuse_score breaks ByteTrack's second stage is not
  supported by the checked implementation: second-stage association uses IoU.
- SoccerNet Game State Reconstruction:
  https://github.com/SoccerNet/sn-gamestate
  https://arxiv.org/abs/2404.11335
  Separates tracking, re-identification, roles, teams, jersey numbers and camera
  calibration. Team colour alone is not same-team person identification.
- Roboflow sports examples: https://github.com/roboflow/sports
  Useful reusable calibration/visualization components. A tutorial overlay is
  not evidence of match-long identity or event accuracy.

Start with association/dataflow fixes. Detection confidence is not tracking
accuracy, and high detector scores do not rule out ball or referee errors.
Retraining should follow measured domain-specific detection errors, rather than
being the default response to ID switches.

## September 11 completed-run audit and experimental event replay

Source runs (Drive originals untouched):
- `20260910_170402`: Messi, 15 passes, 4 interceptions, no goals; 109 fragmented
  player tracks, 4,968 direct ball observations out of 6,455 frames.
- `20260910_173839`: 08, 2 passes, 1 interception, an erroneous early shot
  candidate; 31 player tracks, 597 direct ball observations out of 750 frames.
- `20260910_174331`: Brighton, 1 pass; 104 player tracks, 718 direct ball
  observations out of 1,150 frames. The local 46-second source cuts from match
  clock 1:04 to 11:30 near clip second 41. Do not assume continuous play.

The legacy FSM can silently change possession when direct-possession emission
is disabled. Absolute speed, reception slowdown and airborne heuristics also
miss short exchanges. It has no goal confirmation stage: threshold tuning or
detector retraining cannot supply that missing capability.

Added opt-in `analytics/contacts.py`, `--event-engine contacts`, and
`scripts/replay_events.py`. Contacts use real observations, player-height-scaled
foot distances, runner-up ambiguity, relative ball/player movement, temporal
support, concurrent-player separation and ball-coverage gates. A straight fly-by
must not be treated as reception. Unknown/opposing team transfers are not
automatically interceptions. Discontinuities inferred from tracking reset
ownership; this is NOT a complete video scene-cut detector.

Replays create a NEW folder with unchanged final tracks, updated consistent
reports/video, source-track SHA256 and detailed rejection evidence. They do not
fix wrong upstream teams or identities. The default legacy path stays intact.
See `EVENT_REPLAY_COLAB.md` for independent Messi-first cells.

Initial revised Messi result: 23 pass hypotheses and 2 uncertain ball transfers.
08 remains at 2 pass hypotheses plus 2 uncertain transfers; Brighton has 4 pass
hypotheses. Those figures do not establish an overall improvement on either clip.
Sampled original frames support recovered passes near 3.8, 12.0 and 36.2 seconds.
This is not a full annotation or a precision/recall measurement. Some legacy
passes are now withheld; one-touch and occluded contacts remain missing. Do not
claim 30 correct passes or 85-90% accuracy from these counts.

Visual audit also found the black goalkeeper in 08 assigned the green team,
although the visible backpass is from white. Current keeper-centroid assignment
is not trustworthy. Messi overlap contacts near 75 and 94 seconds can select
the opponent's feet. Fixing those requires better association evidence, not
forcing all transfers to the attacking team. Neither defect was silently
patched with clip-specific team labels.

Automatic goals/shots/crosses/assists remain unsupported by the contact engine.
Messi's ball is visibly in the net near 98.5 seconds, but this manual observation
was NOT inserted as an automatically detected goal. Brighton's title alone is
not goal ground truth. The new mode explicitly records unsupported classes.

Useful next implementation references, separate from detector retraining:
- https://github.com/SoccerNet/sn-spotting (action and ball-action spotting)
- https://github.com/lRomul/ball-action-spotting (2023 competition solution)
- https://github.com/SoccerNet/sn-teamspotting (team-aware spotting baseline)

Next priorities: timestamped event ground truth, visual scene boundaries,
keeper/team evidence and overlap association, then a separately validated
goal/shot spotter. Never use the clip title or desired pass total as labels
inside inference. Mason Mount was still running and is not covered by this audit.

Verification: 141 tests pass, including fly-by rejection, camera translation,
25/60 fps, interpolated-ball rejection, ambiguity, missing flight, scene reset,
ID fragmentation and source-preserving replay. Two existing Windows sklearn/
joblib warnings remain. These are software tests, not benchmark accuracy.
