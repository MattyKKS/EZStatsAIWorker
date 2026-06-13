"""Post-clustering jersey-colour outlier check (dry-run reporter).

Reads team_id (tracks_with_teams.json) + per-track jersey colour
(appearance_manifest.json `upper_body_hsv`). Computes a robust colour centroid
per team, then flags each player track as:
  - OK           : close to its assigned team's colour
  - REASSIGN->Tx : clearly closer to the OTHER team's colour (mis-cluster)
  - REFEREE?     : far from BOTH teams (dark/odd kit — likely referee/noise)

Does NOT touch the SigLIP classifier. Dry-run only: prints findings so we
decide before applying. HSV is OpenCV scale (H 0-180, S/V 0-255).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def _hue_circ(h1: float, h2: float) -> float:
    d = abs(h1 - h2) % 180.0
    return min(d, 180.0 - d)  # 0-90


def _color_dist(a: list, b: list) -> float:
    """Normalised colour distance: hue(circular), saturation, value each 0-100."""
    dh = _hue_circ(a[0], b[0]) / 90.0 * 100.0
    ds = abs(a[1] - b[1]) / 255.0 * 100.0
    dv = abs(a[2] - b[2]) / 255.0 * 100.0
    return (dh * dh + ds * ds + dv * dv) ** 0.5


def _centroid(colors: list[list]) -> list:
    """Robust per-channel median (hue via the value closest to others)."""
    hs = [c[0] for c in colors]
    # circular-ish: pick hue median by minimising summed circular dist
    best = min(hs, key=lambda h: sum(_hue_circ(h, o) for o in hs)) if hs else 0.0
    return [best, median(c[1] for c in colors), median(c[2] for c in colors)]


def analyze_team_outliers(
    run_dir: Path,
    *,
    own_far_thresh: float = 35.0,   # combined colour dist beyond which "far" from own team
    ref_both_thresh: float = 35.0,  # far from BOTH teams by this => referee/noise
) -> dict:
    run_dir = Path(run_dir).resolve()
    tracks = json.loads((run_dir / "tracks_with_teams.json").read_text(encoding="utf-8"))
    hsv: dict[int, list | None] = {}
    am = run_dir / "appearance_manifest.json"
    if am.exists():
        for e in json.loads(am.read_text(encoding="utf-8")).get("tracks", []):
            hsv[e["track_id"]] = e.get("upper_body_hsv")

    team_of: dict[int, int] = {}
    by_id: dict[int, list] = defaultdict(list)
    for t in tracks:
        if t["label"] == "player":
            by_id[t["track_id"]].append(t)
    for tid, obs in by_id.items():
        team_of[tid] = Counter(o.get("team_id") for o in obs).most_common(1)[0][0]

    # robust team centroids from tracks WITH colour
    team_colors: dict[int, list[list]] = defaultdict(list)
    for tid, tm in team_of.items():
        if hsv.get(tid):
            team_colors[tm].append(hsv[tid])
    centroid = {tm: _centroid(cols) for tm, cols in team_colors.items() if cols}

    findings = []
    for tid, tm in team_of.items():
        c = hsv.get(tid)
        if not c or tm not in centroid:
            findings.append({"tid": tid, "team": tm, "verdict": "no-color", "hsv": c})
            continue
        teams = sorted(centroid)
        dists = {t: _color_dist(c, centroid[t]) for t in teams}
        d_own = dists[tm]
        d_other = min((dists[t] for t in teams if t != tm), default=999.0)
        other = min((t for t in teams if t != tm), key=lambda t: dists[t], default=None)
        if d_own <= own_far_thresh:
            verdict = "OK"
        elif other is not None and d_other < d_own and d_other <= own_far_thresh:
            verdict = f"REASSIGN->T{other}"
        elif d_own > own_far_thresh and d_other > ref_both_thresh:
            verdict = "REFEREE?"
        else:
            verdict = "borderline"
        findings.append({
            "tid": tid, "team": tm, "verdict": verdict, "hsv": c,
            "d_own": round(d_own, 1), "d_other": round(d_other, 1),
            "nobs": len(by_id[tid]),
        })
    return {"centroid": {t: [round(x, 1) for x in v] for t, v in centroid.items()},
            "findings": findings}


def print_report(rep: dict) -> None:
    print("team colour centroids (H,S,V):")
    for t, c in rep["centroid"].items():
        print(f"  T{t}: {c}")
    order = {"REFEREE?": 0, "REASSIGN->T0": 1, "REASSIGN->T1": 1, "borderline": 2, "no-color": 3, "OK": 4}
    print("\ntid  team nobs  hsv                    dOwn  dOther  verdict")
    for f in sorted(rep["findings"], key=lambda f: order.get(f["verdict"], 9)):
        if f["verdict"] in ("no-color",):
            print(f"  {f['tid']:>4} T{f['team']}  {f.get('nobs','?'):>4}  {str(f['hsv']):<22} --     --    {f['verdict']}")
        else:
            print(f"  {f['tid']:>4} T{f['team']}  {f['nobs']:>4}  {str(f['hsv']):<22} {f['d_own']:>5} {f['d_other']:>6}   {f['verdict']}")


if __name__ == "__main__":
    import sys
    rd = sys.argv[1] if len(sys.argv) > 1 else "outputs/20260608_224414"
    print_report(analyze_team_outliers(Path(rd)))
