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
    # guard: the two colours must map to different team_ids to trust the vote
    if len(set(mapping.values())) < 2:
        return dict(orig_team), []
    corrected, changes = {}, []
    for tid in survivors:
        cb = _color_team(color.get(tid))
        new = mapping[cb] if cb in mapping else orig_team.get(tid)
        corrected[tid] = new
        if new is not None and orig_team.get(tid) is not None and new != orig_team[tid]:
            changes.append((tid, orig_team[tid], new, cb))
    return corrected, changes


def build_plan(run_dir: Path) -> dict:
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
    refs = []
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
            refs.append(tid)

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
    }


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
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rd = Path(args[0] if args else "outputs/20260608_224414")
    plan = build_plan(rd)
    print_plan(plan)
    if "--apply" in sys.argv:
        out = write_report(rd, plan)
        print(f"\nWrote {out}")
        print("Open outputs/viewer.html and load this file to compare with match_report.json")
