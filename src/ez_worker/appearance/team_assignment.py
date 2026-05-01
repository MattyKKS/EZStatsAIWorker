from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


def cluster_teams(
    run_dir: Path,
    *,
    method: str,
    cluster_count: int,
    model_name: str,
) -> Path:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "appearance_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Appearance manifest not found: {manifest_path}. "
            "Run 'ez-worker prepare-appearance --run-dir ...' first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_tracks = manifest.get("tracks", [])

    # Exclude goalkeeper and referee tracks from clustering — they have distinct
    # appearances that distort the team clusters.
    excluded_ids = _load_non_player_track_ids(run_dir)
    cluster_tracks = [t for t in all_tracks if t["track_id"] not in excluded_ids]
    excluded_tracks = [t for t in all_tracks if t["track_id"] in excluded_ids]

    # Fall back to all tracks if exclusion leaves too few
    if len(cluster_tracks) < cluster_count:
        cluster_tracks = all_tracks
        excluded_tracks = []

    if method == "color":
        features = _color_features(cluster_tracks)
        assignments = _cluster_features(features, cluster_count=cluster_count)
    elif method in {"siglip", "siglip-jersey"}:
        focus = method == "siglip-jersey"
        # Tutorial approach: fit on bulk video crops (stride=60), predict per-track
        assignments = _siglip_cluster_by_crop(
            cluster_tracks, model_name=model_name, focus_upper_body=focus,
            cluster_count=cluster_count, run_dir=run_dir,
        )
    else:
        raise ValueError(
            f"Unsupported method '{method}'. Use 'color', 'siglip', or 'siglip-jersey'."
        )

    clustered_entries = [
        {
            "track_id": track["track_id"],
            "cluster_id": int(cluster_id),
            "crop_count": track["crop_count"],
            "upper_body_hsv": track.get("upper_body_hsv"),
            "excluded": False,
        }
        for track, cluster_id in zip(cluster_tracks, assignments, strict=False)
    ]
    excluded_entries = [
        {
            "track_id": track["track_id"],
            "cluster_id": None,
            "crop_count": track["crop_count"],
            "upper_body_hsv": track.get("upper_body_hsv"),
            "excluded": True,
        }
        for track in excluded_tracks
    ]

    output = {
        "run_dir": str(run_dir),
        "method": method,
        "cluster_count": cluster_count,
        "model_name": model_name if method in {"siglip", "siglip-jersey"} else None,
        "preprocessing": (
            "upper-body jersey focus (HSV, green-masked)"
            if method in {"color", "siglip-jersey"}
            else "full crop"
        ),
        "excluded_track_ids": sorted(excluded_ids),
        "tracks": clustered_entries + excluded_entries,
    }
    output_path = run_dir / "team_clusters.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output_path


def _load_non_player_track_ids(run_dir: Path) -> set[int]:
    """Return track_ids whose dominant source_label is goalkeeper or referee."""
    tracks_path = run_dir / "tracks.json"
    if not tracks_path.exists():
        return set()
    raw = json.loads(tracks_path.read_text(encoding="utf-8"))
    by_track: dict[int, list[str]] = {}
    for obs in raw:
        if obs.get("label") != "player":
            continue
        tid = int(obs["track_id"])
        label = (obs.get("source_label") or obs.get("label") or "player").lower()
        by_track.setdefault(tid, []).append(label)

    excluded: set[int] = set()
    for tid, labels in by_track.items():
        counts = Counter(labels)
        dominant = counts.most_common(1)[0][0]
        if dominant in {"goalkeeper", "referee"}:
            excluded.add(tid)
    return excluded


def _color_features(tracks: list[dict]) -> np.ndarray:
    features: list[list[float]] = []
    for track in tracks:
        color = track.get("upper_body_hsv")
        if color is None:
            raise ValueError(
                f"Track {track.get('track_id')} is missing color metadata. "
                "Re-run 'prepare-appearance' on a run with valid player crops."
            )
        features.append([float(value) for value in color])
    return np.asarray(features, dtype=np.float32)


def _siglip_cluster_by_crop(
    tracks: list[dict],
    *,
    model_name: str,
    focus_upper_body: bool,
    cluster_count: int,
    run_dir: Path,
) -> np.ndarray:
    """
    Tutorial-exact approach:
    1. Fit UMAP+KMeans on bulk crops sampled from the video at stride=60
    2. Predict per saved track crop, majority-vote per track
    """
    try:
        import cv2
        import torch
        from PIL import Image
        from transformers import AutoProcessor, SiglipVisionModel
    except ImportError as exc:
        raise RuntimeError(
            "SigLIP dependencies are not installed. Run 'pip install -e .[appearance]'."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  SigLIP using device: {device}")
    processor = AutoProcessor.from_pretrained(model_name)
    siglip = SiglipVisionModel.from_pretrained(model_name).to(device)
    siglip.eval()

    valid_track_ids = {t["track_id"] for t in tracks}

    def _embed_pil(img: "Image.Image") -> np.ndarray:
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = siglip(**inputs)
        return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

    # --- Phase 1: collect bulk crops from video at STRIDE=60 for UMAP fitting ---
    summary_path = run_dir / "summary.json"
    tracks_path = run_dir / "tracks.json"
    fit_embeddings: list[np.ndarray] = []

    if summary_path.exists() and tracks_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        video_path = summary.get("video_path", "")
        raw_tracks = json.loads(tracks_path.read_text(encoding="utf-8"))

        # Build frame→player-bbox lookup (only player-class tracks we care about)
        by_frame: dict[int, list[dict]] = {}
        for obs in raw_tracks:
            if obs.get("label") != "player":
                continue
            if int(obs["track_id"]) not in valid_track_ids:
                continue
            fi = int(obs["frame_index"])
            by_frame.setdefault(fi, []).append(obs)

        STRIDE = 60
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % STRIDE == 0 and frame_idx in by_frame:
                    h, w = frame.shape[:2]
                    for obs in by_frame[frame_idx]:
                        bbox = obs.get("bbox", {})
                        x1 = max(0, int(float(bbox.get("x1", 0)) * w))
                        y1 = max(0, int(float(bbox.get("y1", 0)) * h))
                        x2 = min(w, int(float(bbox.get("x2", 1)) * w))
                        y2 = min(h, int(float(bbox.get("y2", 1)) * h))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        if focus_upper_body:
                            img = _focus_jersey_region(img)
                        fit_embeddings.append(_embed_pil(img))
                frame_idx += 1
            cap.release()

    # Fall back to saved crops if video collection failed
    if not fit_embeddings:
        for track in tracks:
            for crop_path in track.get("sample_crops", []):
                try:
                    img = Image.open(crop_path).convert("RGB")
                    if focus_upper_body:
                        img = _focus_jersey_region(img)
                    fit_embeddings.append(_embed_pil(img))
                except Exception:
                    continue

    if not fit_embeddings:
        raise ValueError("No crops available for SigLIP fitting.")

    fit_features = np.asarray(fit_embeddings, dtype=np.float32)
    print(f"  SigLIP: fitting UMAP+KMeans on {len(fit_features)} crops from video")

    # Fit UMAP + KMeans on bulk video crops
    try:
        import umap as umap_mod
        reducer = umap_mod.UMAP(
            n_components=3,
            n_neighbors=min(15, max(2, len(fit_features) - 1)),
            min_dist=0.1,
            random_state=42,
        )
        fit_proj = reducer.fit_transform(fit_features)
    except ImportError:
        reducer = None
        fit_proj = fit_features

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    km.fit(fit_proj)

    # Save fitted model so render-stats-video can do per-frame prediction
    try:
        import joblib
        joblib.dump(
            {"reducer": reducer, "km": km, "model_name": model_name, "focus_upper_body": focus_upper_body},
            run_dir / "team_classifier.joblib",
        )
        print(f"  Saved team classifier → {run_dir / 'team_classifier.joblib'}")
    except Exception as e:
        print(f"  Warning: could not save team classifier: {e}")

    # --- Phase 2: predict per saved track crop, majority-vote per track ---
    track_id_order = [t["track_id"] for t in tracks]
    track_votes: dict[int, Counter] = {tid: Counter() for tid in track_id_order}

    for track in tracks:
        for crop_path in track.get("sample_crops", []):
            try:
                img = Image.open(crop_path).convert("RGB")
                if focus_upper_body:
                    img = _focus_jersey_region(img)
                emb = _embed_pil(img).reshape(1, -1)
                if reducer is not None:
                    proj = reducer.transform(emb)
                else:
                    proj = emb
                label = int(km.predict(proj)[0])
                track_votes[track["track_id"]][label] += 1
            except Exception:
                continue

    assignments = np.array([
        track_votes[tid].most_common(1)[0][0] if track_votes[tid] else 0
        for tid in track_id_order
    ])
    return assignments


def _siglip_features(
    tracks: list[dict],
    *,
    model_name: str,
    focus_upper_body: bool,
) -> np.ndarray:
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, SiglipVisionModel
    except ImportError as exc:
        raise RuntimeError(
            "SigLIP dependencies are not installed. Run 'pip install -e .[appearance]' "
            "before using '--method siglip'."
        ) from exc

    # Tutorial uses SiglipVisionModel + last_hidden_state mean pooling (NOT get_image_features)
    processor = AutoProcessor.from_pretrained(model_name)
    model = SiglipVisionModel.from_pretrained(model_name)
    model.eval()

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for track in tracks:
            crop_paths = track.get("sample_crops", [])
            if not crop_paths:
                raise ValueError(
                    f"Track {track.get('track_id')} has no sample crops in the manifest."
                )

            crop_embeddings: list[np.ndarray] = []
            for crop_path in crop_paths:
                image = Image.open(crop_path).convert("RGB")
                if focus_upper_body:
                    image = _focus_jersey_region(image)
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                # Mean pool over sequence dimension — matches tutorial exactly
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                crop_embeddings.append(embedding)

            embeddings.append(np.mean(crop_embeddings, axis=0))

    return np.asarray(embeddings, dtype=np.float32)


def _cluster_features(features: np.ndarray, *, cluster_count: int) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise RuntimeError(
            "Clustering dependencies are not installed. Run 'pip install -e .[appearance]' "
            "before building team clusters."
        ) from exc

    if features.shape[0] >= 4 and features.shape[1] > 3:
        try:
            import umap
        except ImportError:
            reduced = features
        else:
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=min(15, max(2, features.shape[0] - 1)),
                min_dist=0.1,
                random_state=42,
            )
            reduced = reducer.fit_transform(features)
    else:
        reduced = features

    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    return model.fit_predict(reduced)


def _focus_jersey_region(image):
    width, height = image.size
    top = 0
    bottom = max(1, int(height * 0.55))
    left = int(width * 0.15)
    right = max(left + 1, int(width * 0.85))
    return image.crop((left, top, right, bottom))
