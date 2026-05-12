from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ez_worker.schemas import Event, VideoMeta

# ── BAS-2025 class list (matches Colab training Cell C label order) ──────────
BALL_ACTION_CLASSES = [
    "background",           # 0 — suppressed
    "PASS",                 # 1
    "DRIVE",                # 2
    "HEADER",               # 3
    "HIGH PASS",            # 4
    "OUT",                  # 5
    "CORNER",               # 6
    "CROSS",                # 7
    "THROW IN",             # 8
    "SHOT",                 # 9
    "BALL PLAYER BLOCK",    # 10
    "PLAYER SUCCESSFUL TACKLE",  # 11
    "FREE KICK",            # 12
    "GOAL",                 # 13
]

# Per-class confidence thresholds — higher for rare/high-stakes events
_CLASS_THRESHOLDS: dict[str, float] = {
    "GOAL":                    0.60,
    "SHOT":                    0.50,
    "CORNER":                  0.50,
    "FREE KICK":               0.50,
    "THROW IN":                0.45,
    "CROSS":                   0.45,
    "HEADER":                  0.45,
    "PLAYER SUCCESSFUL TACKLE": 0.45,
    "PASS":                    0.38,
    "HIGH PASS":               0.38,
    "DRIVE":                   0.38,
    "BALL PLAYER BLOCK":       0.38,
    "OUT":                     0.38,
}
_DEFAULT_THRESHOLD = 0.40

# Event type mapping: BAS-2025 class name → internal event_type string
_CLASS_TO_EVENT: dict[str, str] = {
    "PASS":                    "pass",
    "DRIVE":                   "drive",
    "HEADER":                  "header",
    "HIGH PASS":               "high_pass",
    "OUT":                     "out",
    "CORNER":                  "corner",
    "CROSS":                   "cross",
    "THROW IN":                "throw_in",
    "SHOT":                    "shot_attempt",
    "BALL PLAYER BLOCK":       "ball_player_block",
    "PLAYER SUCCESSFUL TACKLE": "tackle",
    "FREE KICK":               "free_kick",
    "GOAL":                    "goal",
}

WINDOW    = 15    # feature-frames per prediction (matches BAS-2025 training)
STRIDE    = 3     # prediction stride in feature-frame units
FEAT_FPS  = 2.0   # features extracted at 2fps (matches BAS-2025 training)

NUM_CLASSES = len(BALL_ACTION_CLASSES)  # 14


def run_event_spotter(
    video_path: Path,
    video: VideoMeta,
    model_name: str,
) -> list[Event]:
    """
    Run the BAS-2025-trained LSTM event spotter.

    Extracts ResNet18 features at 2fps, runs sliding-window inference,
    and returns events mapped back to video frame indices.
    Requires model trained with docs/train_event_spotter_bas.ipynb.
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

    # ResNet18 feature extractor (512-dim avgpool output)
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = _EventSpotterModel().to(device)
    state = torch.load(str(model_name), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

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
            window  = feat_tensor[start: start + WINDOW].unsqueeze(0).to(device)
            logits  = model(window)
            probs   = torch.softmax(logits, dim=-1)[0]
            pred_cls = int(probs.argmax().item())
            confidence = float(probs[pred_cls].item())

            # Skip background and low-confidence predictions
            if pred_cls == 0:
                continue
            class_name = BALL_ACTION_CLASSES[pred_cls]
            threshold  = _CLASS_THRESHOLDS.get(class_name, _DEFAULT_THRESHOLD)
            if confidence < threshold:
                continue

            center_feat_idx = start + WINDOW // 2
            if center_feat_idx >= len(sampled_frame_indices):
                continue
            video_frame_idx = sampled_frame_indices[center_feat_idx]

            events.append(Event(
                frame_index=video_frame_idx,
                time_seconds=round(video_frame_idx / video.fps, 2),
                event_type=_CLASS_TO_EVENT.get(class_name, class_name.lower()),
                details={
                    "spotter_class": class_name,
                    "spotter_confidence": round(confidence, 3),
                },
            ))

    return _dedupe_spotter_events(events, min_gap_frames=int(video.fps * 1.5))


def fuse_events(
    rule_events: list[Event],
    spotter_events: list[Event],
    video_fps: float,
    upgrade_window_frames: int | None = None,
) -> list[Event]:
    """
    Fuse rule-based and LSTM spotter events following Footovision's approach.

    Strategy:
    - LSTM upgrades rule-based events within ±upgrade_window_frames:
        rule PASS/TOUCH + LSTM CROSS within window → upgrade to CROSS
        rule shot_attempt + LSTM GOAL within window → upgrade to GOAL
    - LSTM emits directly for classes rules cannot detect:
        HEADER, CORNER, FREE KICK, THROW IN, GOAL, TACKLE, BALL PLAYER BLOCK
    - Rule-based events are kept as-is when no LSTM counterpart found.
    """
    if upgrade_window_frames is None:
        upgrade_window_frames = int(video_fps * 0.5)  # 0.5 second window

    # Classes that LSTM can detect but rules cannot — emit directly
    _lstm_only = {
        "header", "corner", "free_kick", "throw_in",
        "goal", "tackle", "ball_player_block", "cross",
        "high_pass", "drive", "out",
    }
    # Upgrades: (rule_event_type → set of LSTM types that upgrade it)
    _upgrades: dict[str, set[str]] = {
        "pass":        {"cross", "high_pass", "header"},
        "shot_attempt": {"goal"},
        "touch":       {"header", "drive"},
    }

    used_spotter: set[int] = set()
    fused: list[Event] = []

    for rule_evt in rule_events:
        upgraded = False
        for i, sp_evt in enumerate(spotter_events):
            if i in used_spotter:
                continue
            if abs(rule_evt.frame_index - sp_evt.frame_index) > upgrade_window_frames:
                continue
            upgrades_for_type = _upgrades.get(rule_evt.event_type, set())
            if sp_evt.event_type in upgrades_for_type:
                # Upgrade the rule event with LSTM type and timing
                fused.append(Event(
                    frame_index=sp_evt.frame_index,
                    time_seconds=sp_evt.time_seconds,
                    event_type=sp_evt.event_type,
                    actor_track_id=rule_evt.actor_track_id,
                    target_track_id=rule_evt.target_track_id,
                    details={**rule_evt.details, **sp_evt.details, "upgraded_from": rule_evt.event_type},
                ))
                used_spotter.add(i)
                upgraded = True
                break
        if not upgraded:
            fused.append(rule_evt)

    # Emit LSTM-only events not matched to any rule event
    for i, sp_evt in enumerate(spotter_events):
        if i not in used_spotter and sp_evt.event_type in _lstm_only:
            fused.append(sp_evt)

    return sorted(fused, key=lambda e: e.frame_index)


# ── Internal helpers ──────────────────────────────────────────────────────────

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
    idx_pos = 0

    try:
        while idx_pos < len(frame_indices):
            target = frame_indices[idx_pos]
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if not ok:
                break
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = preprocess(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = backbone(tensor).squeeze(0).cpu().numpy()
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
    out:  list[Event]    = []
    for e in sorted(events, key=lambda x: x.frame_index):
        last = seen.get(e.event_type, -999999)
        if e.frame_index - last >= min_gap_frames:
            out.append(e)
            seen[e.event_type] = e.frame_index
    return out


class _EventSpotterModel:
    """Bi-LSTM architecture — must match BAS-2025 Colab training (Cell C)."""
    def __new__(cls):
        import torch.nn as nn
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    512, 256,
                    batch_first=True, bidirectional=True,
                    num_layers=2, dropout=0.3
                )
                self.head = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, NUM_CLASSES),  # 14 BAS-2025 classes
                )
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, out.shape[1] // 2])
        return _Model()
