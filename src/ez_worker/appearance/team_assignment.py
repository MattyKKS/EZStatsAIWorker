from __future__ import annotations

import json
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
    tracks = manifest.get("tracks", [])
    if len(tracks) < cluster_count:
        raise ValueError(
            f"Not enough tracks to form {cluster_count} clusters. "
            f"Found only {len(tracks)} tracks."
        )

    if method == "color":
        features = _color_features(tracks)
    elif method == "siglip":
        features = _siglip_features(tracks, model_name=model_name, focus_upper_body=False)
    elif method == "siglip-jersey":
        features = _siglip_features(tracks, model_name=model_name, focus_upper_body=True)
    else:
        raise ValueError(
            f"Unsupported method '{method}'. Use 'color', 'siglip', or 'siglip-jersey'."
        )

    assignments = _cluster_features(features, cluster_count=cluster_count)
    output = {
        "run_dir": str(run_dir),
        "method": method,
        "cluster_count": cluster_count,
        "model_name": model_name if method in {"siglip", "siglip-jersey"} else None,
        "preprocessing": (
            "upper-body jersey focus" if method == "siglip-jersey" else "full crop"
        ),
        "tracks": [
            {
                "track_id": track["track_id"],
                "cluster_id": int(cluster_id),
                "crop_count": track["crop_count"],
                "upper_body_mean_bgr": track.get("upper_body_mean_bgr"),
            }
            for track, cluster_id in zip(tracks, assignments, strict=False)
        ],
    }
    output_path = run_dir / "team_clusters.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output_path


def _color_features(tracks: list[dict]) -> np.ndarray:
    features: list[list[float]] = []
    for track in tracks:
        color = track.get("upper_body_mean_bgr")
        if color is None:
            raise ValueError(
                f"Track {track.get('track_id')} is missing color metadata. "
                "Re-run 'prepare-appearance' on a run with valid player crops."
            )
        features.append([float(value) for value in color])
    return np.asarray(features, dtype=np.float32)


def _siglip_features(
    tracks: list[dict],
    *,
    model_name: str,
    focus_upper_body: bool,
) -> np.ndarray:
    try:
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "SigLIP dependencies are not installed. Run 'pip install -e .[appearance]' "
            "before using '--method siglip'."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
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
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                crop_embeddings.append(image_features.squeeze(0).cpu().numpy())

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
                n_components=min(8, features.shape[1]),
                n_neighbors=min(8, max(2, features.shape[0] - 1)),
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
