# =============================================================================
# PIPELINE v2 — 2026-09-10
#
# Pipeline C's models (unchanged — no retraining) plus every fix from the
# Root Cause Analysis in docs/product_plan.ipynb:
#
#   RC-1  per-frame homography      -> --per-frame-stride on detect-pitch-keypoints,
#                                      consumed by rerun_events.py. Was ONE matrix
#                                      calibrated on a single frame and applied to
#                                      the whole clip, which drifts as the camera pans.
#   RC-2  tracker config            -> configs/bytetrack_football_v3.yaml, which
#                                      finally populates ByteTrack's second
#                                      association (detector conf stays at 0.20;
#                                      the tracker's high thresh moves to 0.50).
#   RC-5  referee filter cap        -> report_cleanup no longer deletes a whole
#                                      team for wearing a dark kit.
#   RC-6  learned kit colours       -> team correction learns this match's two kit
#                                      colours instead of assuming green vs white.
#   possession                      -> share of frames on the ball, not a count of
#                                      two touch events.
#
#   report_cleanup is now a PIPELINE STEP, not a manual afterthought, so every run
#   emits match_report_merged.json (the file the backend serves).
#
# Usage:
#   .\run_pipeline_v2.ps1 -Video data/raw/08fd33_4.mp4                  # regression gate
#   .\run_pipeline_v2.ps1 -Video data/raw/BrightonGoal.mp4 -SkipVideo   # fast: ~8 min
#   .\run_pipeline_v2.ps1 -Video data/raw/leo_messi_30pass.mp4 -SourceFps 60
#   .\run_pipeline_v2.ps1 -Video data/raw/BrightonGoal.mp4 -GoalFrame 1050
#
#   -SkipVideo skips render-stats-video (~30 min of the ~38 min runtime). Use it
#   while tuning events; drop it for the final deliverable video.
# =============================================================================
param(
    [string]$Video      = "data/raw/08fd33_4.mp4",
    [int]$SourceFps     = 25,       # 60 for the Messi clip — scales frame-based knobs
    [int]$FrameStep     = 1,
    [int]$KpStride      = 10,       # per-frame homography sampling (0.4 s at 25 fps)
    [int]$GoalFrame     = -1,       # mark the shot nearest this frame as a goal
    [switch]$SkipVideo
)

$PLAYER_MODEL  = "artifacts/training/player_detector_v2/weights/best.pt"
$BALL_MODEL    = "artifacts/ball/football-ball-detection.pt"
$PITCH_MODEL   = "artifacts/pitch/football-pitch-detectionV2.pt"
$trackerConfig = "configs/bytetrack_football_v3.yaml"

# Frame-based tracking knobs scale with fps so a 60 fps clip keeps the same
# real-time windows as the 25 fps benchmark. Event thresholds are NOT here —
# they live in config.py in per-second / cm units and need no scaling.
$s = [math]::Round($SourceFps / 25.0, 2)
$gapFrames    = [int](20 * $s)
$interpGap    = [int](10 * $s)
$ballResetGap = [int](20 * $s)
$ballHoldGap  = [int](8  * $s)
$minTrackLen  = [int](2  * $s)
$minTrackFrm  = [int](5  * $s)
$kpStrideEff  = [int]($KpStride * $s)

function Invoke-Step {
    param([string]$Name, [scriptblock]$Block, [switch]$AllowFailure)
    Write-Host ""; Write-Host ">>> [$Name] starting..."
    $t = [System.Diagnostics.Stopwatch]::StartNew()
    $result = & $Block
    $t.Stop()
    $elapsed = "{0:mm\:ss\.ff}" -f $t.Elapsed
    if (-not $? -and -not $AllowFailure) { Write-Error "[$Name] FAILED after $elapsed"; exit 1 }
    Write-Host "<<< [$Name] done in $elapsed"
    return $result
}

$pipelineStart = [System.Diagnostics.Stopwatch]::StartNew()
Write-Host "PIPELINE v2  video=$Video  fps=$SourceFps (scale $s)  kp-stride=$kpStrideEff  skipVideo=$SkipVideo"

$output = Invoke-Step "analyze" {
    ez-worker analyze `
      --video $Video --provider ultralytics `
      --model-name $PLAYER_MODEL --ball-model-name $BALL_MODEL `
      --tracker-config $trackerConfig --render-video --frame-step $FrameStep `
      --ball-detection-imgsz 1280 --detection-confidence 0.20 --detection-iou 0.45 `
      --min-player-confidence 0.18 --min-ball-confidence 0.15 `
      --min-track-length $minTrackLen --max-players-per-frame 28 `
      --max-unique-players 60 --min-player-track-frames $minTrackFrm `
      --max-track-gap-frames $gapFrames --max-track-merge-distance-px 85 `
      --max-interpolation-gap-frames $interpGap --max-ball-jump-px 90 `
      --ball-reset-gap-frames $ballResetGap --ball-reset-confidence 0.55 `
      --ball-hold-max-gap-frames $ballHoldGap --ball-smoothing-alpha 0.35 `
      --possession-distance-threshold-px 120 --possession-min-seconds 0.15 `
      --auto-calibrate --export-player-crops
}

$runDir = ($output | Select-Object -Last 1).Trim() -replace "Analysis complete\. Outputs written to: ", ""
Write-Host "Run dir: $runDir"

Invoke-Step "prepare-appearance"  { ez-worker prepare-appearance --run-dir $runDir }
Invoke-Step "cluster-teams"       { ez-worker cluster-teams --run-dir $runDir --method siglip --cluster-count 2 }
Invoke-Step "apply-team-clusters" { ez-worker apply-team-clusters --run-dir $runDir }


# RC-1: sample keypoints across the clip, not just the single best frame.
Invoke-Step "detect-pitch-keypoints" -AllowFailure {
    ez-worker detect-pitch-keypoints --run-dir $runDir --model-path $PITCH_MODEL --per-frame-stride $kpStrideEff
}

Invoke-Step "rerun-events"             { python rerun_events.py $runDir }
Invoke-Step "apply-team-clusters (sync)" { ez-worker apply-team-clusters --run-dir $runDir }

# Teams from jersey colour: ONE decision per track, overriding SigLIP. Must run
# AFTER the sync above, which rewrites team_id from the SigLIP clusters and would
# otherwise undo this. On the benchmark it corrects tracks 3 and 14 (both confirmed
# SigLIP errors) and, being per-track, removes team flicker entirely.
Invoke-Step "assign-teams-color" -AllowFailure { ez-worker assign-teams-color --run-dir $runDir }

# report_cleanup is a real step now — every run emits match_report_merged.json.
Invoke-Step "report-cleanup" -AllowFailure {
    if ($GoalFrame -ge 0) {
        python -m src.ez_worker.postprocess.report_cleanup $runDir --apply --goal-frame $GoalFrame
    } else {
        python -m src.ez_worker.postprocess.report_cleanup $runDir --apply
    }
}

# Render LAST: --from-tracks draws the pipeline's own tracks, so it needs the final
# events and the final team assignment to already be on disk.
if (-not $SkipVideo) {
    Invoke-Step "render-stats-video" { ez-worker render-stats-video --run-dir $runDir --from-tracks }
} else {
    Write-Host "`n--- render-stats-video SKIPPED (-SkipVideo) ---"
}

$pipelineStart.Stop()
Write-Host ""
Write-Host "================================================"
Write-Host "PIPELINE v2 complete in $("{0:mm\:ss\.ff}" -f $pipelineStart.Elapsed)"
Write-Host "  Video:        $Video"
Write-Host "  Run dir:      $runDir"
Write-Host "  Report:       $runDir\match_report_merged.json   <- serve this"
if (-not $SkipVideo) { Write-Host "  Stats video:  $runDir\stats_video.mp4" }
Write-Host "================================================"
