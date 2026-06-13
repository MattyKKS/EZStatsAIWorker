# ============================================================================
# PIPELINE D — NEW player + NEW (retrained) ball detector
#   Players : NEW player detector (player_detector_v2)
#   Ball    : NEW ball detector (football-ball-detection-v2.pt) — events + 2D map + visual
#   Spatial : OFF (uncomment last step to enable)
# The real test: does the retrained ball model recover the keeper pass that C missed?
#
# BEFORE running: download the trained ball model from Drive
#   MyDrive/ezstats/runs/ball_detector_v2/weights/best.pt
#   -> artifacts/ball/football-ball-detection-v2.pt   (NEW name, keep the old one)
# ============================================================================

$VIDEO        = "data/raw/08fd33_4.mp4"
$PLAYER_MODEL = "artifacts/training/player_detector_v2/weights/best.pt"
$BALL_MODEL   = "artifacts/ball/football-ball-detection-v2.pt"   # <- NEW trained ball model
$PITCH_MODEL  = "artifacts/pitch/football-pitch-detectionV2.pt"
$trackerConfig = "configs/bytetrack_football.yaml"

if (-not (Test-Path $BALL_MODEL)) {
    Write-Error "New ball model not found at $BALL_MODEL — download best.pt from Drive and place it there first."
    exit 1
}

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

$output = Invoke-Step "analyze" {
    ez-worker analyze `
      --video $VIDEO --provider ultralytics `
      --model-name $PLAYER_MODEL --ball-model-name $BALL_MODEL `
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
Invoke-Step "render-stats-video" { ez-worker render-stats-video --run-dir $runDir --pitch-model-path $PITCH_MODEL --player-model-path $PLAYER_MODEL --ball-model-path $BALL_MODEL }
Invoke-Step "rerun-events" { python rerun_events.py $runDir }
Invoke-Step "apply-team-clusters (sync)" { ez-worker apply-team-clusters --run-dir $runDir }
# Spatial video OFF for now (uncomment to enable):
# Invoke-Step "render-spatial-video" { ez-worker render-spatial-video --run-dir $runDir --pitch-model-path $PITCH_MODEL --ball-model-path $BALL_MODEL }

$pipelineStart.Stop()
Write-Host ""
Write-Host "================================================"
Write-Host "PIPELINE D (NEW player + NEW ball) complete in $("{0:mm\:ss\.ff}" -f $pipelineStart.Elapsed)"
Write-Host "  Run dir:     $runDir"
Write-Host "  Stats video: $runDir\stats_video.mp4"
Write-Host "================================================"
