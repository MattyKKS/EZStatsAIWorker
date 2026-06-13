"""Appearance + motion aware tracklet merger (post-clustering).

Runs AFTER cluster-teams / apply-team-clusters so it can use team_id and the
per-track jersey colour in appearance_manifest.json. Collapses fragments left
by ByteTrack when a player is occluded and reappears with a new id.

A candidate fragment B is merged into A only if ALL hold:
  - they never co-exist in the same frame (real simultaneous tracks = 2 players)
  - same team_id (majority)
  - jersey colour is compatible (circular hue distance), when both known
  - B starts within `max_gap_frames` of A ending
  - B's first foot position is within `max_predicted_dist_px` of A's
    velocity-extrapolated position across the gap

Conservative by design: when in doubt it does NOT merge. Dry-run prints the
proposed merges so thresholds can be tuned before anything is written.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# geometry / colour helpers
# ---------------------------------------------------------------------------
def _foot_px(obs: dict, W: int, H: int) -> tuple[float, float]:
    b = obs["bbox"]
    return ((b["x1"] + b["x2"]) / 2 * W, b["y2"] * H)


def _hue_dist(a: list | None, b: list | None) -> float | None:
    """Circular hue distance in OpenCV scale (0-180). None if either missing."""
    if not a or not b:
        return None
    d = abs(a[0] - b[0]) % 180.0
    return min(d, 180.0 - d)


def _end_velocity(obs_list: list[dict], W: int, H: int, k: int = 8) -> tuple[float, float]:
    """Mean per-frame foot velocity over the last k observations."""
    tail = obs_list[-k:] if len(obs_list) >= 2 else obs_list
    if len(tail) < 2:
        return (0.0, 0.0)
    x0, y0 = _foot_px(tail[0], W, H)
    x1, y1 = _foot_px(tail[-1], W, H)
    df = tail[-1]["frame_index"] - tail[0]["frame_index"]
    if df <= 0:
        return (0.0, 0.0)
    return ((x1 - x0) / df, (y1 - y0) / df)


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------
def _absorb_duplicates(
    tl: dict[int, dict],
    *,
    dup_overlap_frac: float,
    dup_dist_px: float,
) -> dict[int, int]:
    """Absorb a short track that shadows a longer one (same player double-detected).

    Returns remap {ghost_tid -> keeper_tid}. The keeper is the LONGER track; the
    ghost coexists with it (high temporal overlap) at near-identical foot
    position. Team is allowed to differ because a ghost is often mislabelled.
    """
    ids = sorted(tl, key=lambda t: -tl[t]["n"])  # longest first = keepers
    foot = {t: {o["frame_index"]: o["_foot"] for o in tl[t]["obs"]} for t in tl}
    remap: dict[int, int] = {}
    for i, keeper in enumerate(ids):
        if keeper in remap:
            continue
        for ghost in ids[i + 1:]:
            if ghost in remap:
                continue
            common = tl[keeper]["frames"] & tl[ghost]["frames"]
            if not common:
                continue
            if len(common) / tl[ghost]["n"] < dup_overlap_frac:
                continue
            md = sum(
                math.hypot(foot[keeper][f][0] - foot[ghost][f][0],
                           foot[keeper][f][1] - foot[ghost][f][1])
                for f in common
            ) / len(common)
            if md <= dup_dist_px:
                remap[ghost] = keeper
    return remap


def plan_merges(
    run_dir: Path,
    *,
    max_gap_frames: int = 75,
    max_predicted_dist_px: float = 150.0,
    color_hue_tol: float = 22.0,
    max_overlap_frames: int = 2,
    cap_vel_px: float = 40.0,
    dup_overlap_frac: float = 0.5,
    dup_dist_px: float = 40.0,
) -> dict:
    """Compute (but do not apply) the merge mapping. Returns a plan dict."""
    run_dir = Path(run_dir).resolve()
    tracks = json.loads((run_dir / "tracks_with_teams.json").read_text(encoding="utf-8"))
    vm = json.loads((run_dir / "video_meta.json").read_text(encoding="utf-8"))
    W, H = vm["width"], vm["height"]

    hsv: dict[int, list | None] = {}
    am_path = run_dir / "appearance_manifest.json"
    if am_path.exists():
        for e in json.loads(am_path.read_text(encoding="utf-8")).get("tracks", []):
            hsv[e["track_id"]] = e.get("upper_body_hsv")

    by_id: dict[int, list[dict]] = defaultdict(list)
    for t in tracks:
        if t["label"] == "player":
            by_id[t["track_id"]].append(t)

    tl: dict[int, dict] = {}
    for tid, obs in by_id.items():
        obs.sort(key=lambda o: o["frame_index"])
        for o in obs:
            o["_foot"] = _foot_px(o, W, H)
        tl[tid] = {
            "tid": tid,
            "obs": obs,
            "start": obs[0]["frame_index"],
            "end": obs[-1]["frame_index"],
            "frames": {o["frame_index"] for o in obs},
            "team": Counter(o.get("team_id") for o in obs).most_common(1)[0][0],
            "start_foot": _foot_px(obs[0], W, H),
            "end_foot": _foot_px(obs[-1], W, H),
            "vel": _end_velocity(obs, W, H),
            "n": len(obs),
        }

    orig_n = len(tl)
    orig_team = Counter(d["team"] for d in tl.values())

    # Pass 1: absorb co-temporal duplicate ghosts (short track shadowing a longer one)
    dup_remap = _absorb_duplicates(tl, dup_overlap_frac=dup_overlap_frac, dup_dist_px=dup_dist_px)
    dup_merges = [
        {"from": g, "into": k, "team": tl[g]["team"], "gap": None,
         "pred_dist_px": None, "hue_dist": None, "kind": "dup"}
        for g, k in dup_remap.items()
    ]
    for ghost in dup_remap:
        tl.pop(ghost, None)

    # Pass 2: sequential (occlusion) merge on the survivors
    order = sorted(tl.values(), key=lambda d: d["start"])
    parent: dict[int, int] = {d["tid"]: d["tid"] for d in order}

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merges: list[dict] = []
    # representative state per root (updates as fragments are appended)
    rep = {d["tid"]: dict(d) for d in order}

    for cand in order:
        ctid = cand["tid"]
        if root(ctid) != ctid:
            continue  # already merged into an earlier root
        best = None
        for cur in order:
            rtid = cur["tid"]
            if rtid == ctid or root(rtid) != rtid:
                continue
            R = rep[rtid]
            if cand["start"] <= R["end"]:
                continue  # cand not strictly after this root's current end
            # temporal exclusivity
            if len(R["frames"] & cand["frames"]) > max_overlap_frames:
                continue
            gap = cand["start"] - R["end"]
            if gap > max_gap_frames:
                continue
            if R["team"] != cand["team"]:
                continue
            hd = _hue_dist(hsv.get(rtid), hsv.get(ctid))
            if hd is not None and hd > color_hue_tol:
                continue
            vx, vy = R["vel"]
            vx = max(-cap_vel_px, min(cap_vel_px, vx))
            vy = max(-cap_vel_px, min(cap_vel_px, vy))
            px = R["end_foot"][0] + vx * gap
            py = R["end_foot"][1] + vy * gap
            dist = math.hypot(px - cand["start_foot"][0], py - cand["start_foot"][1])
            if dist > max_predicted_dist_px:
                continue
            score = dist + gap * 0.5  # prefer closest, smallest-gap continuation
            if best is None or score < best[0]:
                best = (score, rtid, gap, dist, hd)
        if best is not None:
            _, rtid, gap, dist, hd = best
            parent[ctid] = rtid
            R = rep[rtid]
            # append cand's observations to the representative
            R["obs"] = sorted(R["obs"] + cand["obs"], key=lambda o: o["frame_index"])
            R["frames"] |= cand["frames"]
            R["end"] = cand["end"]
            R["end_foot"] = cand["end_foot"]
            R["vel"] = _end_velocity(R["obs"], W, H)
            R["n"] += cand["n"]
            merges.append({
                "from": ctid, "into": rtid, "team": cand["team"],
                "gap": gap, "pred_dist_px": round(dist, 1),
                "hue_dist": (round(hd, 1) if hd is not None else None),
                "kind": "occlusion",
            })

    seq_remap = {tid: root(tid) for tid in parent if root(tid) != tid}
    # final remap: duplicate ghosts point at the root of their keeper
    final_remap: dict[int, int] = {g: seq_remap.get(k, k) for g, k in dup_remap.items()}
    final_remap.update(seq_remap)
    surviving_roots = {root(d["tid"]) for d in order}
    teams_after = Counter(rep[r]["team"] for r in surviving_roots)
    return {
        "run_dir": str(run_dir),
        "players_before": orig_n,
        "players_after": orig_n - len(final_remap),
        "per_team_before": dict(orig_team),
        "per_team_after": dict(teams_after),
        "remap": final_remap,
        "merges": dup_merges + merges,
    }


def print_plan(plan: dict) -> None:
    print(f"players: {plan['players_before']} -> {plan['players_after']}")
    print(f"per-team before: {plan['per_team_before']}   after: {plan['per_team_after']}")
    print(f"{len(plan['merges'])} merges:")
    for m in plan["merges"]:
        if m["kind"] == "dup":
            print(f"  #{m['from']:>4} -> #{m['into']:<4}  team T{m['team']}  [duplicate ghost]")
        else:
            print(f"  #{m['from']:>4} -> #{m['into']:<4}  team T{m['team']}  gap={m['gap']:>3}f  "
                  f"predDist={m['pred_dist_px']:>5}px  hueD={m['hue_dist']}  [occlusion]")


def apply_to_match_report(run_dir: Path, plan: dict | None = None) -> Path:
    """Non-destructive: write match_report_merged.json with the merge applied.

    Merges absorbed players into their keeper (summing stats), drops the
    absorbed entries, and remaps event / pass-network track ids. Leaves the
    original match_report.json untouched.
    """
    run_dir = Path(run_dir).resolve()
    if plan is None:
        plan = plan_merges(run_dir)
    remap = plan["remap"]
    rep = json.loads((run_dir / "match_report.json").read_text(encoding="utf-8"))

    # players: keep survivors, fold absorbed stats into their keeper
    survivors = {p["track_id"]: dict(p) for p in rep["players"] if p["track_id"] not in remap}
    for p in rep["players"]:
        if p["track_id"] in remap:
            root = remap[p["track_id"]]
            if root in survivors:
                for k in ("touches", "passes", "shots", "distance_px"):
                    survivors[root][k] = (survivors[root].get(k, 0) or 0) + (p.get(k, 0) or 0)
    rep["players"] = sorted(survivors.values(), key=lambda p: p["track_id"])

    # events: remap actor / target ids
    for e in rep.get("events", []):
        if e.get("actor") in remap:
            e["actor"] = remap[e["actor"]]
        if e.get("target") in remap:
            e["target"] = remap[e["target"]]

    # pass network: dedupe nodes, combine edges
    pn = rep.get("pass_network")
    if pn:
        nodes = {}
        for n in pn.get("nodes", []):
            nid = remap.get(n["id"], n["id"])
            nodes.setdefault(nid, {**n, "id": nid})
        edges: dict = {}
        for ed in pn.get("edges", []):
            f = remap.get(ed["from"], ed["from"])
            t = remap.get(ed["to"], ed["to"])
            edges[(f, t)] = edges.get((f, t), 0) + ed.get("count", 0)
        pn["nodes"] = list(nodes.values())
        pn["edges"] = [{"from": f, "to": t, "count": c} for (f, t), c in edges.items()]

    out = run_dir / "match_report_merged.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rd = args[0] if args else "outputs/20260608_224414"
    plan = plan_merges(Path(rd))
    print_plan(plan)
    if "--apply" in sys.argv:
        out = apply_to_match_report(Path(rd), plan)
        print(f"\nWrote {out}")
        print(f"players in report: {plan['players_before']} -> {plan['players_after']}")
        print("Open outputs/viewer.html and load this file to compare with match_report.json")
