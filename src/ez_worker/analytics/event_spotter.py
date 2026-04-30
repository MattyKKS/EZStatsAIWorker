from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ez_worker.schemas import Event, VideoMeta

SOCCERNET_CLASSES = [
    "background", "kick-off", "goal", "substitution", "offside", "foul",
    "indirect free-kick", "clearance", "ball out of play", "throw-in",
    "penalty", "corner", "challenge", "direct free-kick", "shot",
    "yellow card", "red card", "yellow->red card",
]

WINDOW = 15       # frames of features per prediction (matches training)
STRIDE = 3        # prediction stride in feature-frame units
FEAT_FPS = 2.0    # features sampled at 2fps (matches SoccerNet training)
CONF_THRESHOLD = 0.40  # minimum softmax confidence to emit an event


def run_event_spotter(
    video_path: Path,
    video: VideoMeta,
    model_name: str,
) -> list[Event]:
    """
    Run the trained LSTM event spotter on the video.

    Extracts ResNet18 features at 2fps, runs sliding-window inference,
    and returns a list of predicted events mapped back to video frame indices.

    NOTE: The model was trained on SoccerNet's ResNet152+PCA-512 features.
    We use ResNet18 (also 512-dim) as a local approximation — predictions may
    be imprecise until the model is retrained with ResNet18 features.
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for the event spotter. "
            "Run: pip install torchvision"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature extractor: ResNet18 avgpool → 512-dim
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load trained LSTM
    model = _EventSpotterModel().to(device)
    state = torch.load(str(model_name), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Sample frames at FEAT_FPS intervals
    frame_interval = max(1, int(round(video.fps / FEAT_FPS)))
    sampled_frame_indices: list[int] = list(range(0, video.frame_count, frame_interval))

    features = _extract_features(
        video_path, sampled_frame_indices, backbone, preprocess, device
    )
    if len(features) < WINDOW:
        return []

    feat_tensor = torch.tensor(np.array(features), dtype=torch.float32)

    events: list[Event] = []
    with torch.no_grad():
        for start in range(0, len(features) - WINDOW + 1, STRIDE):
            window = feat_tensor[start : start + WINDOW].unsqueeze(0).to(device)
            logits = model(window)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_class = int(probs.argmax().item())
            confidence = float(probs[pred_class].item())

            if pred_class == 0 or confidence < CONF_THRESHOLD:
                continue

            # Map feature-frame center back to video frame index
            center_feat_idx = start + WINDOW // 2
            if center_feat_idx >= len(sampled_frame_indices):
                continue
            video_frame_idx = sampled_frame_indices[center_feat_idx]

            events.append(Event(
                frame_index=video_frame_idx,
                time_seconds=round(video_frame_idx / video.fps, 2),
                event_type=SOCCERNET_CLASSES[pred_class],
                details={"spotter_confidence": round(confidence, 3)},
            ))

    return _dedupe_spotter_events(events, min_gap_frames=int(video.fps * 2))


def _extract_features(
    video_path: Path,
    frame_indices: list[int],
    backbone,
    preprocess,
    device,
) -> list[np.ndarray]:
    import torch

    cap = cv2.VideoCapture(str(video_path))
    features: list[np.ndarray] = []
    current_frame = 0
    idx_set = set(frame_indices)
    idx_pos = 0

    try:
        while idx_pos < len(frame_indices):
            target = frame_indices[idx_pos]
            if current_frame != target:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                current_frame = target

            ok, frame = cap.read()
            if not ok:
                break
            current_frame += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = preprocess(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = backbone(tensor).squeeze(0).cpu().numpy()
            # L2-normalize to match SoccerNet feature scale
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            features.append(feat)
            idx_pos += 1
    finally:
        cap.release()

    return features


def _dedupe_spotter_events(events: list[Event], min_gap_frames: int) -> list[Event]:
    seen: dict[str, int] = {}
    out: list[Event] = []
    for e in sorted(events, key=lambda x: x.frame_index):
        last = seen.get(e.event_type, -999999)
        if e.frame_index - last >= min_gap_frames:
            out.append(e)
            seen[e.event_type] = e.frame_index
    return out


class _EventSpotterModel:
    """Recreates the architecture trained in Colab Cell C."""
    def __new__(cls):
        import torch.nn as nn
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(512, 256, batch_first=True,
                                    bidirectional=True, num_layers=2, dropout=0.3)
                self.head = nn.Sequential(
                    nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 18)
                )
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, out.shape[1] // 2])
        return _Model()
