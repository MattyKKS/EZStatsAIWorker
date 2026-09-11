"""Experimental, auditable ground-contact events from final tracking observations.

Counts are hypotheses, not measured accuracy. No goal, shot, aerial-contact or
long-term identity inference is possible from this contact model alone.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import median

from ez_worker.schemas import Event, TrackObservation, VideoMeta


@dataclass(frozen=True)
class ContactConfig:
    radius_heights: float = 0.65
    ambiguity_margin: float = 0.12
    close_heights: float = 0.25
    min_contact_seconds: float = 0.12
    max_contact_gap_seconds: float = 0.15
    controlled_seconds: float = 0.16
    controlled_speed_heights_per_second: float = 1.8
    max_transfer_seconds: float = 3.0
    min_separation_heights: float = 1.0


def _foot(o, video):
    return (o.bbox.cx * video.width, o.bbox.y2 * video.height)


def detect_contact_events(tracks: list[TrackObservation], video: VideoMeta,
                          *, config: ContactConfig | None = None,
                          cut_frames=()):
    """Return events, evidence report and observed-only possession frames.

    Image motion is measured relative to concurrently visible players. A change
    of ID without spatial separation is rejected, not counted as a pass.
    ``cut_frames`` must come from the source video when available.
    """
    cfg = config or ContactConfig()
    if video.fps <= 0:
        raise ValueError("fps must be positive")
    frames = defaultdict(list)
    for o in tracks:
        frames[o.frame_index].append(o)
    cuts = sorted(set(cut_frames))
    episodes = []
    previous_players = {}
    previous_frame = None
    cumulative = [0.0, 0.0]
    ball_path = {}
    samples = {}
    inferred_cuts = []
    for f, observations in sorted(frames.items()):
        players = {o.track_id: o for o in observations
                   if o.label in ("player", "goalkeeper")
                   and o.source_label != "referee" and not o.is_interpolated}
        common = players.keys() & previous_players.keys()
        discontinuity = previous_frame is not None and (
            f - previous_frame > video.fps * 0.5 or
            any(previous_frame < c <= f for c in cuts))
        motion = (0.0, 0.0)
        if len(common) >= 4:
            shifts = [tuple(a - b for a, b in zip(_foot(players[t], video),
                      _foot(previous_players[t], video))) for t in common]
            motion = tuple(median(s[i] for s in shifts) for i in (0, 1))
            if math.hypot(*motion) > video.width * 0.08:
                discontinuity = True
        elif len(players) >= 4 and len(previous_players) >= 4:
            discontinuity = True
        if discontinuity:
            inferred_cuts.append(f)
        cumulative = [cumulative[i] + motion[i] for i in (0, 1)]
        previous_players, previous_frame = players, f
        balls = [o for o in observations if o.label == "ball" and not o.is_interpolated]
        if len(balls) != 1:
            continue
        ball = balls[0]
        bx, by = _foot(ball, video)
        ball_path[f] = (bx - cumulative[0], by - cumulative[1])
        candidates = []
        for p in players.values():
            h = (p.bbox.y2 - p.bbox.y1) * video.height
            if h < 10:
                continue
            px = min(max(bx, p.bbox.x1 * video.width), p.bbox.x2 * video.width)
            d = math.hypot(bx - px, by - p.bbox.y2 * video.height) / h
            candidates.append((d, p.track_id, h))
        candidates.sort()
        if not candidates or candidates[0][0] > cfg.radius_heights:
            continue
        d, tid, height = candidates[0]
        if len(candidates) > 1 and candidates[1][0] - d < cfg.ambiguity_margin:
            continue
        samples[f] = (players, height)
        foot = _foot(players[tid], video)
        relative = (f, (bx - foot[0]) / height, (by - foot[1]) / height)
        if (episodes and episodes[-1]["id"] == tid
                and f - episodes[-1]["end"] <= video.fps * cfg.max_contact_gap_seconds
                and not any(episodes[-1]["end"] < c <= f for c in inferred_cuts)):
            episode = episodes[-1]
            episode["end"] = f
            episode["frames"].append(f)
            episode["closest"] = min(episode["closest"], d)
            episode["relative"].append(relative)
        else:
            episodes.append(dict(start=f, end=f, id=tid, team=players[tid].team_id,
                                 frames=[f], closest=d, relative=[relative]))

    accepted, rejected = [], []
    minimum = max(2, math.ceil(video.fps * cfg.min_contact_seconds))
    for e in episodes:
        reason = None
        controlled = 0.0
        for a, b in zip(e["relative"], e["relative"][1:]):
            dt = (b[0] - a[0]) / video.fps
            if dt <= cfg.max_contact_gap_seconds and math.dist(a[1:], b[1:]) / dt <= cfg.controlled_speed_heights_per_second:
                controlled += dt
        e["controlled_seconds"] = round(controlled, 3)
        del e["relative"]
        if len(e["frames"]) < minimum:
            reason = "insufficient_real_observations"
        elif e["closest"] > cfg.close_heights:
            reason = "no_close_foot_contact"
        elif controlled < cfg.controlled_seconds:
            # Brief contact needs a camera-compensated redirect. Straight-through
            # trajectories near a bystander's feet must not confirm possession.
            first, last = e["start"], e["end"]
            before = [f for f in ball_path if first - video.fps * .2 <= f < first]
            after = [f for f in ball_path if last < f <= last + video.fps * .2]
            redirect = 0.0
            if before and after:
                a, b = ball_path[min(before)], ball_path[first]
                c, d = ball_path[last], ball_path[max(after)]
                u, v = (b[0] - a[0], b[1] - a[1]), (d[0] - c[0], d[1] - c[1])
                scale = samples[first][1]
                if min(math.hypot(*u), math.hypot(*v)) > scale * .12:
                    cosine = sum(x * y for x, y in zip(u, v)) / (math.hypot(*u) * math.hypot(*v))
                    redirect = math.degrees(math.acos(max(-1, min(1, cosine))))
            e["redirect_degrees"] = round(redirect, 1)
            if redirect < 35:
                reason = "brief_contact_without_redirect"
        if reason:
            rejected.append(dict(e, reason=reason))
        else:
            accepted.append(e)

    events, transfers, possession = [], [], {}
    owner = None
    for e in accepted:
        for f in e["frames"]:
            possession[f] = e["team"]
        cut = owner and any(owner["end"] < c <= e["start"] for c in inferred_cuts)
        if owner and owner["id"] == e["id"] and not cut:
            owner = e
            continue
        if owner:
            reason = None
            gap = (e["start"] - owner["end"]) / video.fps
            if cut:
                reason = "scene_or_tracking_discontinuity"
            elif gap > cfg.max_transfer_seconds:
                reason = "unobserved_transfer_gap"
            # Both players must coexist in at least one contact frame. This
            # rejects sequential ID fragments without guessing a re-identification.
            separations = []
            for f in (owner["end"], e["start"]):
                visible, h = samples[f]
                if owner["id"] in visible and e["id"] in visible:
                    p, q = _foot(visible[owner["id"]], video), _foot(visible[e["id"]], video)
                    separations.append(math.dist(p, q) / h)
            if reason is None and (not separations or max(separations) < cfg.min_separation_heights):
                reason = "identity_or_close_challenge_ambiguous"
            observed = sum(f in ball_path for f in range(owner["end"], e["start"] + 1))
            coverage = observed / max(1, e["start"] - owner["end"] + 1)
            if reason is None and coverage < .5:
                reason = "insufficient_ball_coverage"
            evidence = dict(actor=owner["id"], target=e["id"],
                            departure_frame=owner["end"], reception_frame=e["start"],
                            ball_coverage=round(coverage, 3), reason=reason)
            transfers.append(evidence)
            if reason is None:
                # Opposing-team contact can be a tackle, deflection, keeper-team
                # error or interception. Do not manufacture interception stats.
                kind = "pass" if e["team"] is not None and e["team"] == owner["team"] else "ball_transfer"
                events.append(Event(frame_index=e["start"], time_seconds=round(e["start"] / video.fps, 3),
                                    event_type=kind, actor_track_id=owner["id"], target_track_id=e["id"],
                                    details=dict(evidence, method="ground_contacts_v1",
                                                 review_required=kind != "pass")))
        owner = e
    return events, dict(method="ground_contacts_v1", experimental=True,
                        config=asdict(cfg), cut_frames=inferred_cuts,
                        contacts=accepted, rejected_contacts=rejected, transfers=transfers,
                        unsupported_events=["goal", "shot", "cross", "assist", "aerial_contact"],
                        accuracy=None), possession
