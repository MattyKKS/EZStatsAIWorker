"""Team assignment by jersey colour — one decision per track, cached.

Method
------
Follows the approach used by the working open-source football-analysis projects
(e.g. abdullahtarek/football_analysis), which is simpler and, on our own data,
more accurate than the SigLIP embedding we were using:

  1. Take the **top half** of the player crop — jersey, not shorts or legs.
  2. Run KMeans(k=2) on that crop's pixels. One cluster is the shirt, the other
     is background (grass, crowd, advertising).
  3. Decide which is which by **voting on the four corners** of the crop. Corners
     are background almost by definition, so whichever cluster owns them is the
     background and the *other* cluster is the jersey.
  4. Per track, take the median jersey colour over that track's crops.
  5. KMeans across tracks in a brightness-decoupled colour space; the two
     LARGEST clusters are the teams (see `assign_teams` for why k>2).
  6. Assign **once per track_id** and cache it.

Why not SigLIP
--------------
Measured on the benchmark clip: the two kits separate perfectly by colour
(GREEN saturation 70-118, WHITE saturation 12-36 — no overlap), yet the SigLIP +
UMAP + KMeans path put 4 of 26 players on the wrong team. Deep embeddings encode
pose, background and lighting alongside kit; for "which of two shirts is this",
the shirt's colour is the whole signal.

Step 3 is the part our previous `track_color.torso_hsv` lacked. That function
took a fixed central patch and filtered green by a hard-coded hue range, which is
why grass bleed made it noisy and why it could not generalise to a video whose
kits were not green-vs-white. Learning the background cluster from the corners
needs no assumption about pitch or kit colour.

A player's team never changes during a match, so this is decided once per track
rather than re-predicted every frame.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _kmeans(data: np.ndarray, k: int, *, seed: int = 42):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(data)


def jersey_color(crop_bgr: np.ndarray) -> np.ndarray | None:
    """Dominant shirt colour (BGR) of one player crop, background removed."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 4:
        return None

    top = crop_bgr[: max(1, h // 2), :]
    pixels = top.reshape(-1, 3).astype(np.float32)
    if len(pixels) < 8 or np.allclose(pixels.std(axis=0), 0):
        return None

    km = _kmeans(pixels, 2)
    labels = km.labels_.reshape(top.shape[:2])

    # The four corners are background; whichever cluster owns most of them is
    # the background cluster, so the other one is the shirt.
    corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
    background = max(set(corners), key=corners.count)
    shirt = 1 - background
    return km.cluster_centers_[shirt]


def track_jersey_colors(crops_dir: Path, *, max_crops: int = 12) -> dict[int, np.ndarray]:
    """{track_id: median shirt colour} over each track's exported crops."""
    import cv2

    out: dict[int, np.ndarray] = {}
    if not crops_dir.is_dir():
        return out
    for d in sorted(crops_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("track_"):
            continue
        try:
            tid = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        files = sorted(d.glob("*.jpg"))
        if not files:
            continue
        step = max(1, len(files) // max_crops)
        cols = []
        for f in files[::step][:max_crops]:
            img = cv2.imread(str(f))
            c = jersey_color(img)
            if c is not None:
                cols.append(c)
        if cols:
            out[tid] = np.median(np.asarray(cols), axis=0)
    return out


def _feature(bgr: np.ndarray) -> np.ndarray:
    """Shirt colour -> (chroma_x, chroma_y, value), brightness decoupled from hue.

    Clustering raw BGR does not work here: Euclidean distance in BGR is dominated
    by brightness, so the dark-kitted referees sit far from everyone and capture a
    whole cluster, leaving both team kits merged in the other one. Measured on the
    benchmark clip, that is exactly what happened -- k=2 on BGR returned a single
    cluster containing every outfield player of both teams.

    Hue carries "which kit" and saturation carries "how strongly", so they go in
    as a saturation-weighted chroma vector (also continuous across the 0/180 hue
    wrap). Value goes on its own axis, where the dark officials separate out.
    Ch.6 of the DIP course: HSV exists precisely so intensity stops contaminating
    colour comparisons.
    """
    import cv2

    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0].astype(np.float32)
    h, s, v = hsv[0] * 2.0, hsv[1] / 255.0, hsv[2] / 255.0
    th = np.radians(h)
    return np.array([s * np.cos(th), s * np.sin(th), v], dtype=np.float32)


def assign_teams(
    colors: dict[int, np.ndarray], *, k: int = 5, min_per_side: int = 3
) -> tuple[dict[int, int], np.ndarray]:
    """Cluster per-track shirt colours -> {track_id: team_id} for the two teams.

    The two LARGEST clusters are the teams; everything else (referees, keepers,
    stray crowd detections) is left unassigned for the caller to fall back on.

    Why k=5 rather than 2. A football frame holds more than two colour groups:
    two outfield kits, the officials, and one or two goalkeepers who deliberately
    wear a colour unlike either kit. Measured on the benchmark clip, the gap
    between the two KITS is the SMALLEST of all of them -- green-to-white 0.35,
    versus 0.43 to the red keeper and 0.67 to the dark officials. So with k=2 or
    k=3 the kits are the last thing KMeans separates and both teams stay merged
    in one cluster (k=2 and k=3 both returned a single 26-track blob containing
    every outfield player). Giving KMeans enough clusters to spend on the
    officials and keepers first lets the kits finally split. k=4, 5 and 6 all
    produce the same answer, so 5 is a safe middle.
    """
    if len(colors) < 2 * min_per_side:
        return {}, np.empty((0, 3))
    tids = sorted(colors)
    data = np.asarray([_feature(colors[t]) for t in tids], dtype=np.float32)
    k = max(2, min(k, len(tids)))
    km = _kmeans(data, k)

    from collections import Counter

    counts = Counter(int(l) for l in km.labels_)
    if len(counts) < 2:
        return {}, km.cluster_centers_
    (a, a_n), (b, b_n) = counts.most_common(2)
    if min(a_n, b_n) < min_per_side:
        return {}, km.cluster_centers_
    out = {t: (0 if int(l) == a else 1)
           for t, l in zip(tids, km.labels_) if int(l) in (a, b)}
    return out, km.cluster_centers_[[a, b]]


def apply_to_run(run_dir: Path, *, write: bool = True) -> dict:
    """Assign a team per track from crop colour and rewrite tracks_with_teams.json.

    Returns a summary dict. `write=False` makes it a dry run.
    """
    run_dir = Path(run_dir).resolve()
    colors = track_jersey_colors(run_dir / "player_crops")
    if len(colors) < 4:
        return {"ok": False, "reason": f"only {len(colors)} tracks had usable crops"}

    team_of, centres = assign_teams(colors)

    tracks_path = run_dir / "tracks_with_teams.json"
    if not tracks_path.exists():
        tracks_path = run_dir / "tracks.json"
    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))

    prev = {}
    for o in tracks:
        if o.get("track_id") is not None and o.get("team_id") is not None:
            prev.setdefault(o["track_id"], o["team_id"])

    # Keep the new team ids aligned with the old ones where possible, so a run
    # does not silently swap "team 0" and "team 1" between pipeline versions.
    agree = sum(1 for t, n in team_of.items() if prev.get(t) == n)
    if agree < len(team_of) / 2:
        team_of = {t: 1 - n for t, n in team_of.items()}
        centres = centres[::-1]

    changed = sorted(t for t, n in team_of.items() if t in prev and prev[t] != n)

    if write:
        for o in tracks:
            tid = o.get("track_id")
            if o.get("label") != "ball" and tid in team_of:
                o["team_id"] = team_of[tid]
        (run_dir / "tracks_with_teams.json").write_text(
            json.dumps(tracks), encoding="utf-8")
        (run_dir / "team_assignments_color.json").write_text(
            json.dumps({str(t): int(n) for t, n in team_of.items()}, indent=2),
            encoding="utf-8")

    counts: dict[int, int] = defaultdict(int)
    for n in team_of.values():
        counts[n] += 1
    return {
        "ok": True,
        "tracks": len(team_of),
        "per_team": dict(counts),
        "changed_vs_siglip": changed,
        "centres": centres.tolist(),
    }
