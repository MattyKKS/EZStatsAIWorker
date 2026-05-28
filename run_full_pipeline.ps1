# Full pipeline: analyze -> team clustering -> stats video
# Output folder is auto-detected from the analyze step

# ── Tracker selection ──────────────────────────────────────────────────────
# Step 1: try bytetrack_v2 (lower match_thresh, no ReID — fast)
# Step 2: if IDs still swap in viewer, switch to botsort (has ReID, ~20% slower)
# $trackerConfig = "configs/bytetrack_football_v2.yaml"
$trackerConfig = "configs/bytetrack_football.yaml"
# ──────────────────────────────────────────────────────────────────────────


function Invoke-Step {
    param([string]$Name, [scriptblock]$Block, [switch]$AllowFailure)
    Write-Host ""
    Write-Host ">>> [$Name] starting..."
    $t = [System.Diagnostics.Stopwatch]::StartNew()
    $result = & $Block
    $t.Stop()
    $elapsed = "{0:mm\:ss\.ff}" -f $t.Elapsed
    if (-not $? -and -not $AllowFailure) {
        Write-Error "[$Name] FAILED after $elapsed"
        exit 1
    }
    Write-Host "<<< [$Name] done in $elapsed"
    return $result
}

$pipelineStart = [System.Diagnostics.Stopwatch]::StartNew()

$output = Invoke-Step "analyze" {
    ez-worker analyze `
      --video data/raw/08fd33_4.mp4 `
      --provider ultralytics `
      --model-name artifacts/training/roboflow_detector_v1_light/weights/best.pt `
      --tracker-config $trackerConfig `
      --render-video --frame-step 1 `
      --ball-detection-imgsz 1280 --detection-confidence 0.20 --detection-iou 0.45 `
      --min-player-confidence 0.18 --min-ball-confidence 0.10 `
      --min-track-length 2 --max-players-per-frame 28 `
      --max-unique-players 60 --min-player-track-frames 5 `
      --max-track-gap-frames 20 --max-track-merge-distance-px 85 `
      --max-interpolation-gap-frames 10 --max-ball-jump-px 90 `
      --ball-reset-gap-frames 20 --ball-reset-confidence 0.55 `
      --ball-hold-max-gap-frames 8 --ball-smoothing-alpha 0.35 `
      --possession-distance-threshold-px 120 --possession-min-seconds 0.15 `
      --auto-calibrate --export-player-crops
      # --event-model-name artifacts/training/event_spotter_pcbas2026/model.pt  ← uncomment after training finishes
}

# Extract run dir from the last line of output
$runDir = ($output | Select-Object -Last 1).Trim() -replace "Analysis complete\. Outputs written to: ", ""
Write-Host "Run dir: $runDir"

Invoke-Step "prepare-appearance" { ez-worker prepare-appearance --run-dir $runDir }

Invoke-Step "cluster-teams" { ez-worker cluster-teams --run-dir $runDir --method siglip --cluster-count 2 }

Invoke-Step "apply-team-clusters" { ez-worker apply-team-clusters --run-dir $runDir }

Invoke-Step "detect-pitch-keypoints" -AllowFailure { ez-worker detect-pitch-keypoints --run-dir $runDir --model-path artifacts/pitch/football-pitch-detectionV2.pt }

Invoke-Step "render-stats-video" { ez-worker render-stats-video --run-dir $runDir --pitch-model-path artifacts/pitch/football-pitch-detectionV2.pt --player-model-path artifacts/training/roboflow_detector_v1_light/weights/best.pt }

# Re-run event detection with homography-projected pitch coordinates (build H from
# pitch_keypoints.json written above, project all player/ball positions to cm, then
# call detect_events with those pitch-space coords). Overwrites events.json.
Invoke-Step "rerun-events" { python rerun_events.py $runDir }

# Re-apply team clusters using the video-consistent team assignments written by render-stats-video.
# This ensures match_report/player_labels use the same T1/T2 labels the user sees in the video.
Invoke-Step "apply-team-clusters (sync)" { ez-worker apply-team-clusters --run-dir $runDir }

# Invoke-Step "render-spatial-video" { ez-worker render-spatial-video --run-dir $runDir --pitch-model-path artifacts/pitch/football-pitch-detectionV2.pt }

$pipelineStart.Stop()
$totalElapsed = "{0:mm\:ss\.ff}" -f $pipelineStart.Elapsed

Write-Host ""
Write-Host "================================================"
Write-Host "Pipeline complete in $totalElapsed"
Write-Host "  Stats video:   $runDir\stats_video.mp4"
Write-Host "  Spatial video: $runDir\spatial_video.mp4"
Write-Host "================================================"
