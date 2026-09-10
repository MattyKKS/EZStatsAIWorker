"""Non-destructive match_report cleanup: merger + referee + off-pitch + GK collapse.

Pipeline-safe DIAGNOSTIC. Reads an existing run dir and writes
match_report_merged.json. Does NOT touch the pipeline, stats_video, or the
team classifier. Layers four conservative reductions on the raw 39-player report:

  1. tracklet_merge.plan_merges  -> collapse occlusion/duplicate fragments.
  2. referee removal             -> dark low-saturation torsos (officials the
                                    player detector mislabelled). Validated to
                                    match the known referee crops; NEVER removes
                                    an event actor.
  3. off-pitch removal           -> tracks whose best-frame homography foot
                                    projection lands beyond the goal lines, or
                                    short+stationary tracks projected off the
                                    pitch rectangle and outside the trusted
                                    keypoint hull (advertising / crowd / bench).
  4. goalkeeper collapse         -> a cluster of red, near-goal, co-located
                                    tracks folded into the longest one.

Every removal is printed with its reason so the plan can be audited before the
report is trusted. Event actors are protected throughout.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import cv2

from .tracklet_merge import plan_merges
from .track_color import torso_hsv

PITCH_L = 12000.0
PITCH_W = 7000.0

# At most this many tracks may be removed as officials. Above it, the dark
# low-saturation group is a team kit, not the referee crew (see build_plan).
MAX_REFEREES = 4


def _foot(o, W, H):
    b = o["bbox"]
    return ((b["x1"] + b["x2"]) / 2 * W, b["y2"] * H)


def _homography(kp):
    px, cm = kp["keypoints"], kp["keypoint_pitch_xy_cm"]
    src = np.array([px[k] for k in px], dtype=np.float32)
    dst = np.array([cm[k] for k in px], dtype=np.float32)
    Hm, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return Hm, cv2.convexHull(src)


def _project(Hm, x, y):
    p = Hm @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def _color_team(hsv):
    """Confident jersey colour bucket from torso HSV, else None (ambiguous)."""
    if hsv is None:
        return None
    Hc, S, V = hsv
    if 40 <= Hc <= 78 and S > 55:
        return "GREEN"
    if V > 180 and S < 55:
        return "WHITE"
    return None  # dark (ref/GK), red (GK) or otherwise ambiguous -> leave team alone


def _half_hsv(crops_dir: Path, lo: int, hi: int):
    """(median torso HSV, n_crops) over crops with frame index in [lo, hi)."""
    if not crops_dir.is_dir():
        return None, 0
    vals = []
    for f in sorted(os.listdir(crops_dir)):
        try:
            fi = int(f.split("_")[1].split(".")[0])
        except Exception:
            continue
        if not (lo <= fi < hi):
            continue
        img = cv2.imread(str(crops_dir / f))
        if img is None:
            continue
        h, w = img.shape[:2]
        patch = img[int(0.20 * h):int(0.55 * h), int(0.25 * w):int(0.75 * w)]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        hh, ss = hsv[:, 0], hsv[:, 1]
        keep = hsv[~((hh >= 30) & (hh <= 90) & (ss > 60))]
        vals.append(np.median(keep if len(keep) >= 10 else hsv, axis=0))
    if not vals:
        return None, 0
    return np.median(np.array(vals), axis=0), len(vals)


def detect_id_swaps(survivors, footframes, crops, *, cross_px=20.0,
                    min_side_frames=40, min_side_crops=2):
    """Find ByteTrack identity swaps: two tracks that cross tightly AND whose
    jersey colour flips to the OTHER's colour at the crossing frame.

    Conservative — fires only when ALL hold:
      - tracks come within `cross_px` at some shared frame f,
      - f is at least `min_side_frames` inside BOTH tracks' spans,
      - each track has >= `min_side_crops` crops on each side of f,
      - both tracks show a confident, OPPOSITE colour bucket before vs after f.
    Returns [(a, b, frame, colour_a_before, colour_b_before)].
    """
    swaps, used = [], set()
    ids = sorted(survivors)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            common = sorted(set(footframes.get(a, {})) & set(footframes.get(b, {})))
            if len(common) < 10:
                continue
            fa, fb = footframes[a], footframes[b]
            fmin, dmin = min(((f, math.hypot(fa[f][0] - fb[f][0], fa[f][1] - fb[f][1]))
                              for f in common), key=lambda x: x[1])
            if dmin > cross_px:
                continue
            # crossing must be well inside both tracks
            sa, ea = min(fa), max(fa)
            sb, eb = min(fb), max(fb)
            if min(fmin - sa, ea - fmin, fmin - sb, eb - fmin) < min_side_frames:
                continue
            (ca_b, na_b) = _half_hsv(crops / f"track_{a:04d}", -10**9, fmin)
            (ca_a, na_a) = _half_hsv(crops / f"track_{a:04d}", fmin, 10**9)
            (cb_b, nb_b) = _half_hsv(crops / f"track_{b:04d}", -10**9, fmin)
            (cb_a, nb_a) = _half_hsv(crops / f"track_{b:04d}", fmin, 10**9)
            if min(na_b, na_a, nb_b, nb_a) < min_side_crops:
                continue
            ca_b, ca_a = _color_team(ca_b), _color_team(ca_a)
            cb_b, cb_a = _color_team(cb_b), _color_team(cb_a)
            if None in (ca_b, ca_a, cb_b, cb_a):
                continue
            if ca_b != ca_a and cb_b != cb_a and ca_b == cb_a and cb_b == ca_a:
                if a in used or b in used:
                    continue
                used.add(a); used.add(b)
                swaps.append((a, b, fmin, ca_b, cb_b))
    return swaps


def _chroma_feat(hsv):
    """Torso colour -> (cx, cy, v).

    Hue is converted to a saturation-weighted chroma vector so that it stays
    continuous across the 0/180 wrap-around, and so that greys (black, white,
    silver) collapse toward the origin instead of landing on an arbitrary hue.
    Those achromatic kits are then separated by `v` (value/brightness) instead.
    Ch.6 of the DIP course: hue/saturation carry the colour, value carries the
    intensity -- keeping them on separate axes is the whole point of HSV.
    """
    Hc, S, V = hsv
    th = math.radians(float(Hc) * 2.0)      # OpenCV hue is 0..180 == 0..360 deg
    s, v = float(S) / 255.0, float(V) / 255.0
    return (s * math.cos(th), s * math.sin(th), v)


def learn_kit_buckets(survivors, color, *, min_per_side=3, min_sep=0.15, k=3):
    """Learn the two kit colours FROM THIS VIDEO instead of hard-coding them.

    k-means over `_chroma_feat`, seeded farthest-point-first so the result is
    deterministic (no RNG). Returns {tid: "A"|"B"} for the tracks belonging to
    the two LARGEST clusters, or None when the kits are not cleanly separable.

    k=3, not 2, on purpose: goalkeepers wear a colour deliberately unlike either
    outfield kit, so with k=2 the keeper's cluster captures one whole seed and
    every outfield player collapses into the other. The third cluster absorbs
    the keeper (and any stray official); tracks outside the two biggest clusters
    are simply left with the team the classifier gave them.

    This is the general form of the GREEN/WHITE rule -- it works for blue-vs-black
    or pink-vs-white equally well, because nothing about the kit is assumed.
    """
    pts = [(tid, _chroma_feat(color[tid])) for tid in survivors if color.get(tid)]
    if len(pts) < 2 * min_per_side:
        return None

    def d2(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    k = max(2, min(k, len(pts)))
    centres = [max(pts, key=lambda p: p[1][2])[1]]                 # brightest torso
    while len(centres) < k:                                        # farthest-point seeding
        centres.append(max(pts, key=lambda p: min(d2(p[1], c) for c in centres))[1])

    for _ in range(30):                                            # Lloyd iterations
        groups = [[] for _ in centres]
        for _tid, f in pts:
            groups[min(range(k), key=lambda i: d2(f, centres[i]))].append(f)
        moved = [tuple(sum(c) / len(g) for c in zip(*g)) if g else centres[i]
                 for i, g in enumerate(groups)]
        if all(d2(a, b) < 1e-9 for a, b in zip(moved, centres)):
            break
        centres = moved

    assign = {tid: min(range(k), key=lambda i: d2(f, centres[i])) for tid, f in pts}
    counts = Counter(assign.values())
    if len(counts) < 2:
        return None
    (a_id, a_n), (b_id, b_n) = counts.most_common(2)
    if min(a_n, b_n) < min_per_side:
        return None                                                # one side too small
    if math.sqrt(d2(centres[a_id], centres[b_id])) < min_sep:
        return None                                                # kits too similar
    return {tid: ("A" if c == a_id else "B")
            for tid, c in assign.items() if c in (a_id, b_id)}


def correct_teams_by_color(survivors, color, orig_team):
    """Override team_id by dominant jersey colour where colour is confident.

    Returns (corrected_team, changes). The colour->team_id mapping is learned
    from the data (whichever team_id the GREEN/WHITE crowd mostly carries), so it
    is not hard-coded to a particular video. Ambiguous (dark/red) tracks keep
    their original team. NEVER touches the per-frame classifier.
    """
    buckets = {"GREEN": Counter(), "WHITE": Counter()}
    for tid in survivors:
        cb = _color_team(color.get(tid))
        if cb and orig_team.get(tid) is not None:
            buckets[cb][orig_team[tid]] += 1
    mapping = {}
    for cb, ctr in buckets.items():
        if ctr:
            mapping[cb] = ctr.most_common(1)[0][0]

    bucket_of = {tid: _color_team(color.get(tid)) for tid in survivors}

    # guard: the two colours must map to different team_ids to trust the vote.
    # FALLBACK (2026-09-10): when the hard-coded GREEN/WHITE pair does not apply
    # to this match, learn the two kit colours from the video itself. Only runs
    # when the original path fails, so videos it already handles are unaffected.
    if len(set(mapping.values())) < 2:
        learned = learn_kit_buckets(survivors, color)
        if learned is None:
            return dict(orig_team), []
        bucket_of = learned
        buckets = {"A": Counter(), "B": Counter()}
        for tid, cb in learned.items():
            if orig_team.get(tid) is not None:
                buckets[cb][orig_team[tid]] += 1
        mapping = {cb: ctr.most_common(1)[0][0] for cb, ctr in buckets.items() if ctr}
        if len(set(mapping.values())) < 2:
            # both learned kits voted for the same team_id -> the classifier
            # collapsed. Split them explicitly onto team 0 / team 1 instead.
            mapping = {"A": 0, "B": 1}

    corrected, changes = {}, []
    for tid in survivors:
        cb = bucket_of.get(tid)
        new = mapping[cb] if cb in mapping else orig_team.get(tid)
        corrected[tid] = new
        if new is not None and orig_team.get(tid) is not None and new != orig_team[tid]:
            changes.append((tid, orig_team[tid], new, cb))
    return corrected, changes


def build_plan(run_dir: Path, *, goal_frames=None, auto_goal: bool = True) -> dict:
    rd = Path(run_dir).resolve()
    tracks = json.loads((rd / "tracks_with_teams.json").read_text(encoding="utf-8"))
    vm = json.loads((rd / "video_meta.json").read_text(encoding="utf-8"))
    rep = json.loads((rd / "match_report.json").read_text(encoding="utf-8"))
    kp = json.loads((rd / "pitch_keypoints.json").read_text(encoding="utf-8"))
    W, H = vm["width"], vm["height"]
    Hm, hull = _homography(kp)

    # event actors are sacred — never remove one
    protected = set()
    for e in rep.get("events", []):
        for k in ("actor", "target"):
            if e.get(k) is not None:
                protected.add(e[k])

    # per-track geometry from tracks_with_teams
    by_id = defaultdict(list)
    for t in tracks:
        if t["label"] == "player":
            by_id[t["track_id"]].append(t)
    geo = {}
    footframes: dict[int, dict[int, tuple]] = {}
    for tid, obs in by_id.items():
        feet = np.array([_foot(o, W, H) for o in obs])
        mfx, mfy = float(np.median(feet[:, 0])), float(np.median(feet[:, 1]))
        spread = float(math.hypot(feet[:, 0].std(), feet[:, 1].std()))
        pcx, pcy = _project(Hm, mfx, mfy)
        hull_d = cv2.pointPolygonTest(hull, (mfx, mfy), True)
        geo[tid] = dict(n=len(obs), mfx=mfx, mfy=mfy, spread=spread,
                        pcx=pcx, pcy=pcy, hull=hull_d)
        footframes[tid] = {o["frame_index"]: _foot(o, W, H) for o in obs}

    crops = rd / "player_crops"
    report_tids = {p["track_id"] for p in rep["players"]}

    # 1) trusted merger remap
    mplan = plan_merges(rd)
    remap = dict(mplan["remap"])

    # work on the post-merge survivor set
    survivors = [t for t in report_tids if t not in remap]

    # 2) referee removal by torso colour (dark + low saturation)
    #
    # GUARD (2026-09-10): a match has at most ~3 visible officials. If the dark
    # low-saturation group is larger than that, it is a TEAM KIT, not the
    # officials -- e.g. Brighton vs Fulham, where Fulham's black away kit made
    # this rule delete nine outfield players. In that case remove nobody: an
    # unfiltered referee is a far cheaper error than a deleted team.
    ref_candidates = []
    color = {}
    for tid in survivors:
        d = crops / f"track_{tid:04d}"
        if not d.is_dir():
            continue
        hsv = torso_hsv(d)
        if hsv is None:
            continue
        Hc, S, V = (float(x) for x in hsv)
        color[tid] = (Hc, S, V)
        if V < 150 and S < 60 and tid not in protected:
            ref_candidates.append(tid)

    if len(ref_candidates) > MAX_REFEREES:
        refs = []
        ref_note = (f"dark-torso group has {len(ref_candidates)} tracks "
                    f"(> {MAX_REFEREES}) -> treated as a TEAM KIT, none removed")
    else:
        refs = ref_candidates
        ref_note = ""

    # 3) off-pitch removal via homography
    offpitch = []
    for tid in survivors:
        if tid in refs or tid in protected:
            continue
        g = geo.get(tid)
        if not g:
            continue
        beyond_goal = g["pcx"] > 12800 or g["pcx"] < -800
        off_rect = not (-600 <= g["pcx"] <= PITCH_L + 600 and -600 <= g["pcy"] <= PITCH_W + 600)
        spurious = off_rect and g["hull"] < -100 and g["n"] < 40 and g["spread"] < 35
        if beyond_goal or spurious:
            offpitch.append(tid)

    # 4) goalkeeper collapse: red + near a goal line + co-located cluster
    def is_red(tid):
        c = color.get(tid)
        return c is not None and c[1] > 95 and (c[0] < 12 or c[0] > 168)

    gk_cluster = [t for t in survivors
                  if t not in refs and t not in offpitch and is_red(t)
                  and geo.get(t) and (geo[t]["pcx"] > 11000 or geo[t]["pcx"] < 1000)]
    gk_remove = []
    if len(gk_cluster) > 1:
        keeper = max(gk_cluster, key=lambda t: geo[t]["n"])
        for t in gk_cluster:
            if t != keeper and t not in protected:
                remap[t] = keeper
                gk_remove.append(t)

    drop = set(refs) | set(offpitch)
    final_survivors = [t for t in survivors if t not in drop and t not in gk_remove]

    # ID-swap correction: undo ByteTrack identity swaps at tight crossings
    swaps = detect_id_swaps(final_survivors, footframes, crops)
    # recompute distance for swapped ids from the stitched (corrected) tracks
    swap_dist = {}
    for a, b, f, _, _ in swaps:
        fa, fb = footframes.get(a, {}), footframes.get(b, {})
        def _dist(frames_pos):
            seq = [frames_pos[k] for k in sorted(frames_pos)]
            return float(sum(math.hypot(seq[i][0] - seq[i - 1][0], seq[i][1] - seq[i - 1][1])
                             for i in range(1, len(seq))))
        a_corr = {k: v for k, v in fa.items() if k < f} | {k: v for k, v in fb.items() if k >= f}
        b_corr = {k: v for k, v in fb.items() if k < f} | {k: v for k, v in fa.items() if k >= f}
        swap_dist[a] = round(_dist(a_corr), 1)
        swap_dist[b] = round(_dist(b_corr), 1)

    # team correction by jersey colour (fixes consistent SigLIP mislabels)
    orig_team = {p["track_id"]: p["team_id"] for p in rep["players"]}
    corrected_team, team_changes = correct_teams_by_color(final_survivors, color, orig_team)
    # swapped ids get the team of their *pre-crossing* (true) colour
    cmap = {"GREEN": None, "WHITE": None}
    for tid in final_survivors:
        cb = _color_team(color.get(tid))
        if cb and corrected_team.get(tid) is not None:
            cmap[cb] = corrected_team[tid]
    for a, b, f, ca_before, cb_before in swaps:
        if cmap.get(ca_before) is not None:
            corrected_team[a] = cmap[ca_before]
        if cmap.get(cb_before) is not None:
            corrected_team[b] = cmap[cb_before]
    team_changes = [(tid, orig_team[tid], corrected_team[tid],
                     _color_team(color.get(tid)) or "swap")
                    for tid in final_survivors
                    if corrected_team.get(tid) is not None
                    and orig_team.get(tid) is not None
                    and corrected_team[tid] != orig_team[tid]]

    return {
        "run_dir": str(rd),
        "raw_players": len(report_tids),
        "after_merge": len(survivors),
        "refs": sorted(refs),
        "ref_note": ref_note,
        "offpitch": sorted(offpitch),
        "gk_cluster": sorted(gk_cluster),
        "gk_keeper": (max(gk_cluster, key=lambda t: geo[t]["n"]) if len(gk_cluster) > 1 else None),
        "final_count": len(final_survivors),
        "final_survivors": sorted(final_survivors),
        "merge_remap": mplan["remap"],
        "gk_remap": {t: remap[t] for t in gk_remove},
        "drop": sorted(drop),
        "remap": remap,
        "color": color,
        "geo": geo,
        "protected": sorted(protected),
        "corrected_team": corrected_team,
        "team_changes": team_changes,
        "swaps": swaps,
        "swap_dist": swap_dist,
        "fps": vm.get("fps", 25.0),
        "goal_frames": list(goal_frames or []),
        "auto_goal": auto_goal,
    }


def recompute_possession(run_dir: Path, corrected_team: dict, remap: dict):
    """Real possession: share of FRAMES each team holds the ball.

    The writer computes possession as the share of `touch` *events* per team,
    which on a clip with two touches yields "50.0 / 50.0" or "100 / 0" -- a count
    of two events, not time on the ball.

    Here a frame is credited to the team of the nearest player whose foot is
    within `own_radius` player-heights of the ball. Using the player's own bbox
    height as the ruler makes the radius scale-free, so it means the same thing
    at 720p and 1080p, zoomed in or out (RC-3). Returns {} if inputs are missing.
    """
    rd = Path(run_dir)
    tf = rd / "tracks_with_teams.json"
    if not tf.exists():
        return {}
    try:
        obs = json.loads(tf.read_text(encoding="utf-8"))
        vm = json.loads((rd / "video_meta.json").read_text(encoding="utf-8"))
        W, H = vm["width"], vm["height"]
        own_radius = 1.6                       # in player-heights (~2.9 m)

        by_frame = defaultdict(lambda: {"ball": None, "players": []})
        for o in obs:
            f = o["frame_index"]
            if o.get("label") == "ball":
                by_frame[f]["ball"] = o
            elif o.get("track_id") is not None:
                by_frame[f]["players"].append(o)

        held = Counter()
        for f, d in by_frame.items():
            b = d["ball"]
            if b is None or not d["players"]:
                continue
            bx = (b["bbox"]["x1"] + b["bbox"]["x2"]) / 2 * W
            by = b["bbox"]["y2"] * H
            best, best_d = None, 1e18
            for p in d["players"]:
                bb = p["bbox"]
                ph = (bb["y2"] - bb["y1"]) * H
                if ph < 10:
                    continue
                px = (bb["x1"] + bb["x2"]) / 2 * W
                py = bb["y2"] * H
                dist = math.hypot(px - bx, py - by) / ph      # in player-heights
                if dist < best_d:
                    best, best_d = p, dist
            if best is None or best_d > own_radius:
                continue
            tid = best["track_id"]
            tid = remap.get(tid, tid)
            team = corrected_team.get(tid, best.get("team_id"))
            if team is not None:
                held[team] += 1

        total = sum(held.values())
        if not total:
            return {}
        return {str(t): round(n / total * 100, 1) for t, n in sorted(held.items())}
    except Exception:
        return {}


def _ball_pitch(run_dir: Path):
    """{frame_index: (pitch_x_cm, pitch_y_cm)} for the ball, via best-frame H.
    Returns {} if pitch keypoints or tracks are unavailable."""
    rd = Path(run_dir)
    kpf = rd / "pitch_keypoints.json"
    tf = rd / "tracks.json"
    if not (kpf.exists() and tf.exists()):
        return {}
    try:
        kp = json.loads(kpf.read_text(encoding="utf-8"))
        Hm, _ = _homography(kp)
        vm = json.loads((rd / "video_meta.json").read_text(encoding="utf-8"))
        W, H = vm["width"], vm["height"]
        out = {}
        for t in json.loads(tf.read_text(encoding="utf-8")):
            if t.get("label") == "ball":
                b = t["bbox"]
                cx = (b["x1"] + b["x2"]) / 2 * W
                cy = (b["y1"] + b["y2"]) / 2 * H
                out[t["frame_index"]] = _project(Hm, cx, cy)
        return out
    except Exception:
        return {}


def mark_goals(rep: dict, run_dir: Path, fps: float, manual_frames, auto: bool):
    """Relabel a `shot` event as `goal`. Manual frames are authoritative; the
    auto pass marks a shot whose ball reaches the goal mouth within ~3s
    (approximate — homography is unreliable for airborne balls)."""
    events = rep.setdefault("events", [])

    def _shots():
        return [e for e in events if e.get("type") == "shot"]

    for gf in manual_frames:
        near = [e for e in _shots() if abs(e.get("frame", -10**9) - gf) <= 1.5 * fps]
        if near:
            best = min(near, key=lambda e: abs(e["frame"] - gf))
            best["type"] = "goal"; best["_goal"] = "manual"
        else:  # no shot detected there — insert a standalone goal event
            events.append({"type": "goal", "frame": int(gf),
                           "time_s": round(gf / fps, 2),
                           "actor": None, "actor_label": None,
                           "target": None, "target_label": None,
                           "details": {"source": "manual"}})

    if auto:
        ball = _ball_pitch(run_dir)
        if ball:
            L = 12000.0
            for e in _shots():
                f = e.get("frame", 0)
                win = [ball[k] for k in ball if f < k <= f + int(3 * fps)]
                if any((px < 700 or px > L - 700) and 2500 <= py <= 4500 for px, py in win):
                    e["type"] = "goal"; e["_goal"] = "auto"

    rep["events"] = sorted(events, key=lambda e: e.get("frame", 0))


def write_report(run_dir: Path, plan: dict) -> Path:
    rd = Path(run_dir).resolve()
    rep = json.loads((rd / "match_report.json").read_text(encoding="utf-8"))
    remap = plan["remap"]
    drop = set(plan["drop"])

    keep = {}
    for p in rep["players"]:
        tid = p["track_id"]
        if tid in drop:
            continue
        if tid in remap:  # folded into a keeper
            continue
        keep[tid] = dict(p)
    # fold merged/gk stats into keepers (skip dropped refs/offpitch)
    for p in rep["players"]:
        tid = p["track_id"]
        if tid in remap and tid not in drop:
            root = remap[tid]
            if root in keep:
                for k in ("touches", "passes", "shots", "distance_px"):
                    keep[root][k] = (keep[root].get(k, 0) or 0) + (p.get(k, 0) or 0)
    rep["players"] = sorted(keep.values(), key=lambda p: p["track_id"])

    # apply colour-corrected team_id (+ rewrite the "T<n>#.." label prefix)
    corrected = plan.get("corrected_team", {})
    for p in rep["players"]:
        tid = p["track_id"]
        if tid in corrected and corrected[tid] is not None and corrected[tid] != p["team_id"]:
            p["team_id"] = corrected[tid]
            lbl = p.get("label", "")
            suffix = lbl.split("#", 1)[1] if "#" in lbl else str(tid)
            p["label"] = f"T{corrected[tid] + 1}#{suffix}"

    # apply corrected distance for ID-swap-stitched tracks
    swap_dist = plan.get("swap_dist", {})
    for p in rep["players"]:
        if p["track_id"] in swap_dist:
            p["distance_px"] = swap_dist[p["track_id"]]

    # ID-swap: for events at/after the crossing frame, swap the two ids so each
    # event references the physically-correct player.
    swaps = plan.get("swaps", [])
    for e in rep.get("events", []):
        for a, b, f, _, _ in swaps:
            if e.get("frame", -1) >= f:
                for k in ("actor", "target"):
                    if e.get(k) == a:
                        e[k] = b
                    elif e.get(k) == b:
                        e[k] = a

    # remap event ids, then re-derive pass/interception (and long_ball/clearance)
    # from the corrected teams, mirroring events.py:411/730.
    team_of = {p["track_id"]: p["team_id"] for p in rep["players"]}
    PAIR = {"pass": "interception", "interception": "pass",
            "long_ball": "clearance", "clearance": "long_ball"}
    for e in rep.get("events", []):
        for k in ("actor", "target"):
            if e.get(k) in remap:
                e[k] = remap[e[k]]
        et = e.get("type")
        a, t = e.get("actor"), e.get("target")
        if et in PAIR and a in team_of and t in team_of:
            same = team_of[a] == team_of[t]
            kept = et in ("pass", "long_ball")  # "same-team" member of the pair
            want = (et if kept == same else PAIR[et])
            if want != et:
                e["type"] = want
                e["_corrected_from"] = et

    # keep event labels consistent with the corrected player labels
    label_of = {p["track_id"]: p.get("label") for p in rep["players"]}
    for e in rep.get("events", []):
        if e.get("actor") in label_of:
            e["actor_label"] = label_of[e["actor"]]
        if e.get("target") in label_of:
            e["target_label"] = label_of[e["target"]]

    # real possession (frames on the ball) instead of the writer's touch-count share
    poss = recompute_possession(rd, plan.get("corrected_team", {}), plan.get("remap", {}))
    if poss:
        rep["possession"] = poss

    # mark goals (manual frames authoritative; conservative auto pass)
    mark_goals(rep, rd, plan.get("fps", 25.0),
               plan.get("goal_frames", []), plan.get("auto_goal", True))
    # refresh labels for any goal event that kept actor/target
    for e in rep.get("events", []):
        if e.get("actor") in label_of:
            e["actor_label"] = label_of[e["actor"]]
        if e.get("target") in label_of:
            e["target_label"] = label_of[e["target"]]

    # recompute summary counts from the (corrected) events
    if "summary" in rep:
        cnt = Counter(e.get("type") for e in rep.get("events", []))
        rep["summary"].update({
            "total_touches": cnt.get("touch", 0),
            "total_passes": cnt.get("pass", 0),
            "total_long_balls": cnt.get("long_ball", 0),
            "total_clearances": cnt.get("clearance", 0),
            "total_interceptions": cnt.get("interception", 0),
            "total_shots": cnt.get("shot", 0),
            "total_goals": cnt.get("goal", 0),
        })

    # pass network: drop removed ids, fold merges, refresh node label/team
    pn = rep.get("pass_network")
    if pn:
        nodes = {}
        for n in pn.get("nodes", []):
            if n["id"] in drop:
                continue
            nid = remap.get(n["id"], n["id"])
            node = {**n, "id": nid}
            if nid in label_of:
                node["label"] = label_of[nid]
            if nid in team_of:
                node["team_id"] = team_of[nid]
            nodes.setdefault(nid, node)
        edges = {}
        for ed in pn.get("edges", []):
            f, t = ed["from"], ed["to"]
            if f in drop or t in drop:
                continue
            f, t = remap.get(f, f), remap.get(t, t)
            edges[(f, t)] = edges.get((f, t), 0) + ed.get("count", 0)
        pn["nodes"] = list(nodes.values())
        pn["edges"] = [{"from": f, "to": t, "count": c} for (f, t), c in edges.items()]

    out = rd / "match_report_merged.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return out


def print_plan(plan: dict) -> None:
    print(f"raw players          : {plan['raw_players']}")
    print(f"after tracklet merge : {plan['after_merge']}")
    print(f"referees removed     : {plan['refs']}")
    if plan.get("ref_note"):
        print(f"  ! referee filter   : {plan['ref_note']}")
    print(f"off-pitch removed    : {plan['offpitch']}")
    print(f"GK cluster           : {plan['gk_cluster']}  -> keep #{plan['gk_keeper']}")
    print(f"protected (events)   : {plan['protected']}")
    for a, b, f, ca, cb in plan.get("swaps", []):
        print(f"ID-swap fixed        : #{a}<->#{b} at frame {f}  (#{a} true={ca}, #{b} true={cb})")
    tc = plan.get("team_changes", [])
    if tc:
        print(f"team corrected (colour vote): {len(tc)}")
        for tid, old, new, cb in tc:
            print(f"    #{tid}: T{old + 1} -> T{new + 1}  ({cb})")
    print(f"FINAL player count   : {plan['final_count']}")
    print(f"final survivors      : {plan['final_survivors']}")


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("--")]
    # --goal-frame N (repeatable) marks the shot near frame N as a goal
    goal_frames = []
    for i, a in enumerate(argv):
        if a == "--goal-frame" and i + 1 < len(argv):
            try:
                goal_frames.append(int(argv[i + 1]))
            except ValueError:
                pass
    auto_goal = "--no-auto-goal" not in argv
    rd = Path(args[0] if args else "outputs/20260608_224414")
    plan = build_plan(rd, goal_frames=goal_frames, auto_goal=auto_goal)
    print_plan(plan)
    if goal_frames or not auto_goal:
        print(f"goal frames (manual): {goal_frames}   auto-goal: {auto_goal}")
    if "--apply" in sys.argv:
        out = write_report(rd, plan)
        print(f"\nWrote {out}")
        print("Open outputs/viewer.html and load this file to compare with match_report.json")
