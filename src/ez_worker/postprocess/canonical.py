"""Finalize identities and roles before computing events or drawing the video."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ez_worker.appearance.team_color import _feature, track_jersey_colors
from ez_worker.schemas import TrackObservation


def reconnect_fragments(tracks, colors, video):
    """Join only unambiguous short continuations; never coexisting players."""
    from scipy.optimize import linear_sum_assignment

    by_id = defaultdict(list)
    for obs in tracks:
        if obs.label != "ball":
            by_id[obs.track_id].append(obs)
    ids = sorted(by_id)
    for observations in by_id.values():
        observations.sort(key=lambda o: o.frame_index)
    features = {t: _feature(c) for t, c in colors.items()}
    costs = np.full((len(ids), len(ids)), 1e6)

    def foot(obs):
        return np.array([obs.bbox.cx * video.width, obs.bbox.y2 * video.height])

    for i, tid in enumerate(ids):
        a = by_id[tid]
        end = a[-1]
        tail = a[-min(len(a), 5)]
        velocity = (foot(end) - foot(tail)) / max(1, end.frame_index - tail.frame_index)
        for j, other in enumerate(ids):
            b = by_id[other]
            start = b[0]
            gap = start.frame_index - end.frame_index
            if tid == other or not (0 < gap <= video.fps * 0.8):
                continue
            if end.source_label != start.source_label or end.team_id != start.team_id:
                continue
            if end.label == "player" and end.team_id is None:
                continue
            if tid not in features or other not in features:
                continue
            colour_distance = float(np.linalg.norm(features[tid] - features[other]))
            if colour_distance > 0.18:
                continue
            height = min(end.bbox.y2 - end.bbox.y1, start.bbox.y2 - start.bbox.y1) * video.height
            if height < 10:
                continue
            distance = float(np.linalg.norm(foot(end) + velocity * gap - foot(start))) / height
            if distance <= 0.75:
                costs[i, j] = distance + gap / video.fps * 0.25

    # Add a separate unmatched option for every row. Hungarian assignment alone
    # does not establish identity: require a clear margin in both directions too.
    augmented = np.concatenate([costs, np.full_like(costs, 1.0)], axis=1)
    rows, columns = linear_sum_assignment(augmented)
    remap = {}
    for i, j in zip(rows, columns):
        if j >= len(ids) or costs[i, j] >= 1.0:
            continue
        row_alternatives = np.delete(costs[i], j)
        col_alternatives = np.delete(costs[:, j], i)
        second = min(row_alternatives.min(initial=1e6), col_alternatives.min(initial=1e6))
        if second - costs[i, j] < 0.25:
            continue
        remap[ids[j]] = ids[i]
    for tid in remap:
        root = remap[tid]
        while root in remap:
            root = remap[root]
        remap[tid] = root
    return [o.model_copy(update={"track_id": remap.get(o.track_id, o.track_id)})
            if o.label != "ball" else o for o in tracks], remap


def assign_roles_and_teams(
    tracks: list[TrackObservation], colors: dict[int, np.ndarray]
) -> tuple[list[TrackObservation], dict]:
    from sklearn.cluster import KMeans

    by_id = defaultdict(list)
    for obs in tracks:
        if obs.label != "ball":
            by_id[obs.track_id].append(obs)
    roles = {}
    for tid, observations in by_id.items():
        votes = Counter()
        for obs in observations:
            votes[obs.source_label or obs.label] += obs.confidence
        roles[tid] = votes.most_common(1)[0][0]

    # Officials and keepers wear separate kits. Do not let their colours take
    # one of the two outfield clusters, or weight short fragments as full players.
    tids = sorted(t for t in colors if roles.get(t) == "player")
    teams = {}
    centres = []
    if len(tids) >= 2:
        features = np.asarray([_feature(colors[t]) for t in tids])
        if len(np.unique(features.round(3), axis=0)) >= 2:
            km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(
                features, sample_weight=[len(by_id[t]) for t in tids])
            teams = {t: int(label) for t, label in zip(tids, km.labels_)}
            centres = km.cluster_centers_.tolist()

    # A keeper's kit is deliberately different. Use contemporaneous outfield
    # positions, not centroids averaged across a panning clip. Leave ties unknown.
    by_frame = defaultdict(list)
    for obs in tracks:
        if obs.track_id in teams and obs.label != "ball":
            by_frame[obs.frame_index].append(obs)
    for tid, role in roles.items():
        if role != "goalkeeper":
            continue
        votes = Counter()
        for obs in by_id[tid]:
            groups = defaultdict(list)
            for p in by_frame[obs.frame_index]:
                groups[teams[p.track_id]].append(p.bbox.cx)
            if all(groups[t] for t in (0, 1)):
                distances = [abs(obs.bbox.cx - float(np.median(groups[t]))) for t in (0, 1)]
                if abs(distances[0] - distances[1]) > 0.03:
                    votes[int(distances[1] < distances[0])] += 1
        if votes and max(votes.values()) / sum(votes.values()) >= 0.7:
            teams[tid] = votes.most_common(1)[0][0]

    result = []
    for obs in tracks:
        if obs.label == "ball":
            result.append(obs)
        else:
            role = roles[obs.track_id]
            result.append(obs.model_copy(update={
                "label": "referee" if role == "referee" else "player",
                "source_label": role,
                "team_id": teams.get(obs.track_id),
            }))
    return result, {"roles": roles, "teams": teams, "team_centres": centres}


def prepare_canonical_tracks(run_dir: Path) -> tuple[list[TrackObservation], dict]:
    raw = run_dir / "tracks_raw.json"
    if not raw.exists():
        raw = run_dir / "tracks.json"
    tracks = [TrackObservation(**o) for o in json.loads(raw.read_text(encoding="utf-8"))]
    colors = track_jersey_colors(run_dir / "player_crops")
    tracks, decisions = assign_roles_and_teams(tracks, colors)
    from ez_worker.schemas import VideoMeta
    video = VideoMeta(**json.loads((run_dir / "video_meta.json").read_text(encoding="utf-8")))
    tracks, remap = reconnect_fragments(tracks, colors, video)
    decisions["fragment_remap"] = remap
    decisions["roles"] = {t: role for t, role in decisions["roles"].items() if t not in remap}
    decisions["teams"] = {t: team for t, team in decisions["teams"].items() if t not in remap}
    (run_dir / "identity_decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    return tracks, decisions
