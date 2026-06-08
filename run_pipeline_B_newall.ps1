# ============================================================================
# PIPELINE B — ALL NEW PLAYER MODEL (no ball-only detector anywhere)
#   Players : NEW player detector (player_detector_v2)
#   Ball    : NEW player detector's ball class for EVENTS + 2D map + visual
#             (--no-ball-detector makes render draw the ball from tracks.json,
#              i.e. the new model's ball — NOT the old ball-only model)
#   Spatial : ON
# Tests whether the new player model alone is good enough for everything.
# ============================================================================

$PLAYER_MODEL = "artifacts/training/player_detector_v2/weights/best.pt"
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

# NOTE: no --ball-model-name -> the main (new player) model handles the ball pass.
$output = Invoke-Step "analyze" {
    ez-worker analyze `
      --video data/raw/08fd33_4.mp4 --provider ultralytics `
      --model-name $PLAYER_MODEL `
      --tracker-config $trackerConfig --render-video --frame-step 1 `
      --ball-detection-imgsz 1280 --detection-confidence 0.20 --detection-iou 0.45 `
      --min-player-confidence 0.18 --min-ball-confidence 0.15 `
      --min-track-length 2 --max-players-per-frame 28 `
      --max-unique-players 60 --min-player-track-frames 5 `
      --max-track-gap-frames 20 --max-track-merge-distance-px 85 `
      --max-interpolation-gap-frames 10 --max-ball-jump-px 90 `
      --ball-reset-gap-frames 20 --ball-reset-confidence 0.55 `
      --ball-hold-max-gap-frames 8 --ball-smoothing-alpha 0.35 `
      --possession-distance-threshold-px 120 --possession-min-seconds 0.15 `
      --auto-calibrate --export-player-crops
}

$runDir = ($output | Select-Object -Last 1).Trim() -replace "Analysis complete\. Outputs written to: ", ""
Write-Host "Run dir: $runDir"

Invoke-Step "prepare-appearance" { ez-worker prepare-appearance --run-dir $runDir }
Invoke-Step "cluster-teams" { ez-worker cluster-teams --run-dir $runDir --method siglip --cluster-count 2 }
Invoke-Step "apply-team-clusters" { ez-worker apply-team-clusters --run-dir $runDir }
Invoke-Step "detect-pitch-keypoints" -AllowFailure { ez-worker detect-pitch-keypoints --run-dir $runDir --model-path $PITCH_MODEL }
Invoke-Step "render-stats-video" { ez-worker render-stats-video --run-dir $runDir --pitch-model-path $PITCH_MODEL --player-model-path $PLAYER_MODEL --no-ball-detector }
Invoke-Step "rerun-events" { python rerun_events.py $runDir }
Invoke-Step "apply-team-clusters (sync)" { ez-worker apply-team-clusters --run-dir $runDir }
# Spatial video OFF for now (uncomment to enable):
# Invoke-Step "render-spatial-video" { ez-worker render-spatial-video --run-dir $runDir --pitch-model-path $PITCH_MODEL --no-ball-detector }

$pipelineStart.Stop()
Write-Host ""
Write-Host "================================================"
Write-Host "PIPELINE B (all NEW player model, no ball-only) complete in $("{0:mm\:ss\.ff}" -f $pipelineStart.Elapsed)"
Write-Host "  Run dir:       $runDir"
Write-Host "  Stats video:   $runDir\stats_video.mp4"
Write-Host "================================================"
