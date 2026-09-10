# ============================================================================
# PIPELINE C on the LEO MESSI tiki-taka clip (leo_messi_30pass.mp4)
#   Players : NEW player detector (player_detector_v2)
#   Ball    : OLD ball-only detector (football-ball-detection.pt)
#   Spatial : OFF (uncomment last step to enable)
#
# This clip is 1920x1036 @ ~60 fps, ~108 s (~6455 frames). Event detection is
# fps-independent (cm/s, px/s, seconds — converted via video.fps), so passes /
# the shot detect correctly at 60fps. ONLY the frame-based TRACKING knobs are
# scaled by 60/25 = 2.4x to preserve the same real-time windows as the 25fps
# benchmark (less ID fragmentation). Ball detection conf left low for the slight
# blur. NOTE: pipeline emits "shot" not "goal" — the final shot IS the goal.
#
# Usage:  .\run_pipeline_C_messi.ps1
# Runtime: long (~6455 frames). If too slow for a first look, add  -FrameStep 2
#          for a faster rough pass (~half the frames).
# ============================================================================
param(
    [int]$FrameStep = 1,
    [switch]$DrawSavedBall   # draw the smoothed analyze ball on the video instead
                             # of re-detecting per frame (reduces visible ball jumps
                             # on blurry footage). Try this if the live ball jumps.
)

$VIDEO        = "data/raw/leo_messi_30pass.mp4"
$PLAYER_MODEL = "artifacts/training/player_detector_v2/weights/best.pt"
$BALL_MODEL   = "artifacts/ball/football-ball-detection.pt"
$PITCH_MODEL  = "artifacts/pitch/football-pitch-detectionV2.pt"
$trackerConfig = "configs/bytetrack_football.yaml"

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

# Tracking gap knobs SCALED for ~60fps (2.4x the 25fps values). Event thresholds
# are NOT here — they live in config.py as per-second units and need no scaling.
$output = Invoke-Step "analyze" {
    ez-worker analyze `
      --video $VIDEO --provider ultralytics `
      --model-name $PLAYER_MODEL --ball-model-name $BALL_MODEL `
      --tracker-config $trackerConfig --render-video --frame-step $FrameStep `
      --ball-detection-imgsz 1280 --detection-confidence 0.20 --detection-iou 0.45 `
      --min-player-confidence 0.18 --min-ball-confidence 0.18 `
      --min-track-length 5 --max-players-per-frame 28 `
      --max-unique-players 60 --min-player-track-frames 12 `
      --max-track-gap-frames 48 --max-track-merge-distance-px 85 `
      --max-interpolation-gap-frames 24 --max-ball-jump-px 50 `
      --ball-reset-gap-frames 48 --ball-reset-confidence 0.65 `
      --ball-hold-max-gap-frames 19 --ball-smoothing-alpha 0.25 `
      --possession-distance-threshold-px 120 --possession-min-seconds 0.15 `
      --auto-calibrate --export-player-crops
}

$runDir = ($output | Select-Object -Last 1).Trim() -replace "Analysis complete\. Outputs written to: ", ""
Write-Host "Run dir: $runDir"

Invoke-Step "prepare-appearance" { ez-worker prepare-appearance --run-dir $runDir }
Invoke-Step "cluster-teams" { ez-worker cluster-teams --run-dir $runDir --method siglip --cluster-count 2 }
Invoke-Step "apply-team-clusters" { ez-worker apply-team-clusters --run-dir $runDir }
Invoke-Step "detect-pitch-keypoints" -AllowFailure { ez-worker detect-pitch-keypoints --run-dir $runDir --model-path $PITCH_MODEL }
Invoke-Step "render-stats-video" {
    if ($DrawSavedBall) {
        ez-worker render-stats-video --run-dir $runDir --pitch-model-path $PITCH_MODEL --player-model-path $PLAYER_MODEL --ball-model-path $BALL_MODEL --no-ball-detector
    } else {
        ez-worker render-stats-video --run-dir $runDir --pitch-model-path $PITCH_MODEL --player-model-path $PLAYER_MODEL --ball-model-path $BALL_MODEL
    }
}
Invoke-Step "rerun-events" { python rerun_events.py $runDir }
Invoke-Step "apply-team-clusters (sync)" { ez-worker apply-team-clusters --run-dir $runDir }
# Spatial video OFF for now (uncomment to enable):
# Invoke-Step "render-spatial-video" { ez-worker render-spatial-video --run-dir $runDir --pitch-model-path $PITCH_MODEL --ball-model-path $BALL_MODEL }

$pipelineStart.Stop()
Write-Host ""
Write-Host "================================================"
Write-Host "PIPELINE C (Messi 30-pass) complete in $("{0:mm\:ss\.ff}" -f $pipelineStart.Elapsed)"
Write-Host "  Video:       $VIDEO  (frame-step $FrameStep)"
Write-Host "  Run dir:     $runDir"
Write-Host "  Stats video: $runDir\stats_video.mp4"
Write-Host "  Next: python -m src.ez_worker.postprocess.report_cleanup $runDir --apply"
Write-Host "================================================"
