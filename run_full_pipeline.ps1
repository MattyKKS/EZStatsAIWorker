# Full pipeline: analyze -> team clustering -> stats video
# Output folder is auto-detected from the analyze step

$output = ez-worker analyze `
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
  --auto-calibrate --export-player-crops

if (-not $?) { Write-Error "analyze failed"; exit 1 }

# Extract run dir from the last line of output
$runDir = ($output | Select-Object -Last 1).Trim() -replace "Analysis complete\. Outputs written to: ", ""
Write-Host "Run dir: $runDir"

ez-worker prepare-appearance --run-dir $runDir
if (-not $?) { Write-Error "prepare-appearance failed"; exit 1 }

ez-worker cluster-teams --run-dir $runDir --method siglip --cluster-count 2
if (-not $?) { Write-Error "cluster-teams failed"; exit 1 }

ez-worker apply-team-clusters --run-dir $runDir
if (-not $?) { Write-Error "apply-team-clusters failed"; exit 1 }

ez-worker detect-pitch-keypoints --run-dir $runDir --model-path artifacts/pitch/football-pitch-detectionV2.pt
if (-not $?) { Write-Warning "detect-pitch-keypoints failed — minimap will be skipped" }

ez-worker render-stats-video --run-dir $runDir --pitch-model-path artifacts/pitch/football-pitch-detectionV2.pt --player-model-path artifacts/training/roboflow_detector_v1_light/weights/best.pt
if (-not $?) { Write-Error "render-stats-video failed"; exit 1 }

ez-worker render-spatial-video --run-dir $runDir --pitch-model-path artifacts/pitch/football-pitch-detectionV2.pt
if (-not $?) { Write-Error "render-spatial-video failed"; exit 1 }

Write-Host ""
Write-Host "Done. Stats video:   $runDir\stats_video.mp4"
Write-Host "      Spatial video: $runDir\spatial_video.mp4"
