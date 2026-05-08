# Full pipeline: analyze -> team clustering -> stats video
# Output folder is auto-detected from the analyze step

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
      --tracker-config configs/bytetrack_football.yaml `
      --render-video --frame-step 1 `
      --ball-detection-imgsz 1280 --detection-confidence 0.20 --detection-iou 0.45 `
      --min-player-confidence 0.18 --min-ball-confidence 0.10 `
      --min-track-length 2 --max-players-per-frame 28 `
      --max-unique-players 60 --min-player-track-frames 5 `
      --max-track-gap-frames 20 --max-track-merge-distance-px 85 `
      --max-interpolation-gap-frames 10 --max-ball-jump-px 90 `
      --ball-reset-gap-frames 20 --ball-reset-confidence 0.55 `
      --ball-hold-max-gap-frames 8 --ball-smoothing-alpha 0.35 `
      --possession-distance-threshold-px 65 --possession-min-consecutive-frames 2 `
      --auto-calibrate --export-player-crops `
      --event-model-name artifacts/training/event_spotter_v1/model.pt
}

# Extract run dir from the last line of output
$runDir = ($output | Select-Object -Last 1).Trim() -replace "Analysis complete\. Outputs written to: ", ""
Write-Host "Run dir: $runDir"

Invoke-Step "prepare-appearance" { ez-worker prepare-appearance --run-dir $runDir }

Invoke-Step "cluster-teams" { ez-worker cluster-teams --run-dir $runDir --method siglip --cluster-count 2 }

Invoke-Step "apply-team-clusters" { ez-worker apply-team-clusters --run-dir $runDir }

Invoke-Step "detect-pitch-keypoints" -AllowFailure { ez-worker detect-pitch-keypoints --run-dir $runDir --model-path artifacts/pitch/football-pitch-detectionV2.pt }

Invoke-Step "render-stats-video" { ez-worker render-stats-video --run-dir $runDir --pitch-model-path artifacts/pitch/football-pitch-detectionV2.pt --player-model-path artifacts/training/roboflow_detector_v1_light/weights/best.pt }

Invoke-Step "render-spatial-video" { ez-worker render-spatial-video --run-dir $runDir --pitch-model-path artifacts/pitch/football-pitch-detectionV2.pt }

$pipelineStart.Stop()
$totalElapsed = "{0:mm\:ss\.ff}" -f $pipelineStart.Elapsed

Write-Host ""
Write-Host "================================================"
Write-Host "Pipeline complete in $totalElapsed"
Write-Host "  Stats video:   $runDir\stats_video.mp4"
Write-Host "  Spatial video: $runDir\spatial_video.mp4"
Write-Host "================================================"
