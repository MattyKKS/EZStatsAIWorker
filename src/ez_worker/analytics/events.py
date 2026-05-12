from __future__ import annotations

import math
from collections import defaultdict
from enum import Enum

from ez_worker.schemas import Event, TrackObservation, VideoMeta

# Ball cannot physically exceed 50 m/s — used to reject tracking noise
_MAX_BALL_SPEED_CMS = 5000.0
# Typical player bounding-box height in pixels at standard broadcast midfield distance.
# Used to normalize pixel distances so they are consistent regardless of camera zoom/pan.
# e.g. a player 140 px tall is roughly 2× closer than one at 70 px, so their raw pixel
# distance should be halved before comparing against the possession threshold.
_STANDARD_PLAYER_HEIGHT_PX = 70.0


class _Phase(Enum):
    IDLE      = "idle"
    POSSESSED = "possessed"   # owner confirmed for >= poss_min_frames
    IN_FLIGHT = "in_flight"   # ball launched, awaiting reception or shot timeout


def detect_events(
    tracks: list[TrackObservation],
    video: VideoMeta,
    ball_track_id: int,
    # ── Footovision Layer 2: optional pitch-coordinate lookup tables ────────
    # ball_pitch_pos[frame]        = (x_cm, y_cm)  — Kalman-filtered from homography
    # player_pitch_pos[frame][tid] = (x_cm, y_cm)
    ball_pitch_pos: dict[int, tuple[float, float]] | None = None,
    player_pitch_pos: dict[int, dict[int, tuple[float, float]]] | None = None,
    # ── Possession thresholds ──────────────────────────────────────────────
    possession_distance_threshold_cm: float = 200.0,  # 2 m — real-world
    possession_distance_threshold_px: float = 50.0,   # pixel fallback (tighter: 65→50)
    possession_min_seconds: float = 0.15,             # 150 ms to confirm ownership
    # ── Pass thresholds ────────────────────────────────────────────────────
    pass_min_speed_cms: float = 250.0,      # 2.5 m/s — minimum for a pass (cm/s)
    pass_min_speed_px_per_s: float = 200.0, # pixel fallback (raised: 150→200 to cut false passes)
    # ── Shot thresholds ────────────────────────────────────────────────────
    shot_min_speed_cms: float = 1200.0,      # 12 m/s — minimum for a shot (cm/s)
    shot_min_speed_px_per_s: float = 600.0,  # pixel fallback
    shot_no_catch_seconds: float = 1.0,      # no reception within 1 s = shot
    # ── Direct-possession pass (slow/blocked ball, no IN_FLIGHT phase) ─────
    ball_direction_change_min_deg: float = 35.0,  # raised from 20→35 to reduce false direct-passes
    # ── High ball / clearance detection ────────────────────────────────────
    clearance_min_flight_seconds: float = 1.2,   # flight longer than this = clearance candidate
    clearance_min_arc_frac: float = 0.04,         # ball y-range > 4% frame height = arc detected
    # ── Scene change frame indices (e.g. from histogram detector) ──────────
    scene_changes: set[int] | None = None,
) -> list[Event]:
    """
    Three-phase event detection following Footovision's architecture.

    IDLE
     └─ ball within possession_dist of Player A for >= poss_min_frames
        ──► POSSESSED(owner=A), emit TOUCH

    POSSESSED(owner=A)
     ├─ ball leaves A at speed >= pass_min_speed ──► IN_FLIGHT
     ├─ ball near different Player B directly (slow takeover)
     │   + direction change >= ball_direction_change_min_deg
     │   ──► emit PASS/INTERCEPTION(A→B), POSSESSED(B)
     └─ scene change ──► IDLE

    IN_FLIGHT(last_owner=A)
     ├─ ball received by Player B for >= poss_min_frames
     │   ├─ B.team == A.team ──► emit PASS(A→B), POSSESSED(B)
     │   └─ B.team != A.team ──► emit INTERCEPTION(A→B), POSSESSED(B)
     ├─ no reception after shot_no_catch_seconds
     │   AND launch speed >= shot_min_speed ──► emit SHOT(A), IDLE
     └─ scene change ──► IDLE
    """
    use_pitch = ball_pitch_pos is not None and player_pitch_pos is not None

    poss_min_frames = max(1, int(possession_min_seconds * video.fps))
    shot_no_catch_f = max(1, int(shot_no_catch_seconds * video.fps))

    # Active speed thresholds — switch between cm/s and px/s units
    pass_speed  = pass_min_speed_cms  if use_pitch else (pass_min_speed_px_per_s  / video.fps)
    shot_speed  = shot_min_speed_cms  if use_pitch else (shot_min_speed_px_per_s  / video.fps)
    poss_dist   = possession_distance_threshold_cm if use_pitch else possession_distance_threshold_px

    by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for obs in tracks:
        by_frame[obs.frame_index].append(obs)
    sorted_frames = sorted(by_frame)

    ball_vel = _compute_ball_velocity(
        by_frame, sorted_frames, ball_track_id, video, ball_pitch_pos, use_pitch
    )

    # ── State machine variables ────────────────────────────────────────────
    phase        = _Phase.IDLE
    owner_tid:   int | None = None
    owner_team:  int | None = None

    candidate_tid:   int | None = None
    candidate_count: int        = 0

    flight_frame: int | None = None
    flight_speed: float      = 0.0
    flight_ball_y: list[float] = []

    vel_history: list[tuple[float, float]] = []
    vel_window = max(3, int(video.fps * 0.25))  # 250 ms velocity history

    scene_changes = scene_changes or set()
    events: list[Event] = []

    for frame_index in sorted_frames:
        frame_obs = by_frame[frame_index]

        # Scene change: broadcast cut resets possession context entirely
        if frame_index in scene_changes:
            phase       = _Phase.IDLE
            owner_tid   = owner_team = None
            candidate_tid   = None
            candidate_count = 0
            flight_frame    = None
            vel_history.clear()
            continue

        ball    = _get_ball(frame_obs, ball_track_id)
        players = [o for o in frame_obs if o.label in ("player", "goalkeeper")]

        vel = ball_vel.get(frame_index)
        if vel is not None:
            vel_history.append(vel)
            if len(vel_history) > vel_window:
                vel_history.pop(0)
        speed = math.hypot(*vel) if vel else 0.0

        # ══════════════════════════════════════════════════════════════════
        # Phase: IN_FLIGHT — waiting for reception or shot timeout
        # ══════════════════════════════════════════════════════════════════
        if phase == _Phase.IN_FLIGHT:
            # Track ball arc for clearance/long_ball detection
            if ball is not None:
                flight_ball_y.append(ball.bbox.cy * video.height)

            if ball is not None and players:
                closest, d = _closest_player(
                    ball, players, frame_index, video,
                    use_pitch, player_pitch_pos, ball_pitch_pos
                )
                if d <= poss_dist:
                    # Player is close — accumulate reception candidate
                    if candidate_tid == closest.track_id:
                        candidate_count += 1
                    else:
                        candidate_tid   = closest.track_id
                        candidate_count = 1

                    if candidate_count >= poss_min_frames:
                        # Confirmed reception — check if this was a high-arc long ball
                        flight_s = (frame_index - (flight_frame or frame_index)) / video.fps
                        is_high = _is_high_arc(flight_ball_y, video.height * clearance_min_arc_frac)
                        if flight_s >= clearance_min_flight_seconds and is_high:
                            etype = "long_ball"
                        else:
                            etype = "pass" if closest.team_id == owner_team else "interception"
                        speed_key = "launch_speed_cms" if use_pitch else "launch_speed_pxs"
                        events.append(Event(
                            frame_index=frame_index,
                            time_seconds=round(frame_index / video.fps, 2),
                            event_type=etype,
                            actor_track_id=owner_tid,
                            target_track_id=closest.track_id,
                            details={
                                speed_key: round(flight_speed, 1),
                                "flight_frames": frame_index - (flight_frame or frame_index),
                            },
                        ))
                        phase           = _Phase.POSSESSED
                        owner_tid       = closest.track_id
                        owner_team      = closest.team_id
                        candidate_tid   = None
                        candidate_count = 0
                        flight_frame    = None
                        flight_ball_y   = []
                        continue
                else:
                    # Different player or ball still free — reset reception candidate
                    candidate_tid   = None
                    candidate_count = 0

            # Shot / clearance timeout: no reception and enough time has passed
            if flight_frame is not None and frame_index - flight_frame >= shot_no_catch_f:
                flight_s  = (frame_index - flight_frame) / video.fps
                is_high   = _is_high_arc(flight_ball_y, video.height * clearance_min_arc_frac)
                speed_key = "launch_speed_cms" if use_pitch else "launch_speed_pxs"
                if flight_s >= clearance_min_flight_seconds and is_high:
                    # High arc with no receiver = clearance
                    events.append(Event(
                        frame_index=flight_frame,
                        time_seconds=round(flight_frame / video.fps, 2),
                        event_type="clearance",
                        actor_track_id=owner_tid,
                        details={speed_key: round(flight_speed, 1), "flight_seconds": round(flight_s, 2)},
                    ))
                elif flight_speed >= shot_speed:
                    # Fast flat ball with no receiver = shot
                    events.append(Event(
                        frame_index=flight_frame,
                        time_seconds=round(flight_frame / video.fps, 2),
                        event_type="shot_attempt",
                        actor_track_id=owner_tid,
                        details={speed_key: round(flight_speed, 1)},
                    ))
                phase           = _Phase.IDLE
                owner_tid       = owner_team = None
                flight_frame    = None
                flight_ball_y   = []
                candidate_tid   = None
                candidate_count = 0

            continue  # remain in IN_FLIGHT until resolved or timed out

        # ══════════════════════════════════════════════════════════════════
        # Phase: IDLE or POSSESSED
        # ══════════════════════════════════════════════════════════════════
        if ball is None or not players:
            continue

        closest, d = _closest_player(
            ball, players, frame_index, video,
            use_pitch, player_pitch_pos, ball_pitch_pos
        )

        if d > poss_dist:
            # Ball away from all players
            if phase == _Phase.POSSESSED and speed >= pass_speed:
                # Ball left owner with real speed → enter IN_FLIGHT
                phase        = _Phase.IN_FLIGHT
                flight_frame = frame_index
                flight_speed = speed
                flight_ball_y   = []
                candidate_tid   = None
                candidate_count = 0
            elif phase == _Phase.POSSESSED:
                # Ball drifted away slowly (rolled out of bounds, etc.)
                phase     = _Phase.IDLE
                owner_tid = owner_team = None
            continue

        # Ball is within possession zone of closest player
        if candidate_tid == closest.track_id:
            candidate_count += 1
        else:
            candidate_tid   = closest.track_id
            candidate_count = 1

        if candidate_count < poss_min_frames:
            continue

        # ── Possession confirmed ────────────────────────────────────────
        if phase == _Phase.IDLE:
            phase      = _Phase.POSSESSED
            owner_tid  = closest.track_id
            owner_team = closest.team_id
            dist_key = "distance_cm" if use_pitch else "distance_px"
            events.append(Event(
                frame_index=frame_index,
                time_seconds=round(frame_index / video.fps, 2),
                event_type="touch",
                actor_track_id=closest.track_id,
                details={dist_key: round(d, 1)},
            ))

        elif phase == _Phase.POSSESSED and closest.track_id != owner_tid:
            # Pressing check: if the current owner is still within possession distance,
            # the "closest" player is just a presser — don't flip ownership.
            owner_obs = next((p for p in players if p.track_id == owner_tid), None)
            if owner_obs is not None:
                _, owner_d = _closest_player(
                    ball, [owner_obs], frame_index, video,
                    use_pitch, player_pitch_pos, ball_pitch_pos,
                )
                if owner_d <= poss_dist:
                    continue  # owner still in control; ignore presser

            # Owner is out of possession zone — allow direct takeover
            dir_change = _direction_change_deg(vel_history)
            if dir_change >= ball_direction_change_min_deg:
                etype = "pass" if closest.team_id == owner_team else "interception"
                events.append(Event(
                    frame_index=frame_index,
                    time_seconds=round(frame_index / video.fps, 2),
                    event_type=etype,
                    actor_track_id=owner_tid,
                    target_track_id=closest.track_id,
                    details={"direction_change_deg": round(dir_change, 1)},
                ))
            owner_tid  = closest.track_id
            owner_team = closest.team_id

    return _dedupe_dense_events(events)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_ball(
    frame_obs: list[TrackObservation], ball_track_id: int
) -> TrackObservation | None:
    return (
        next((o for o in frame_obs if o.label == "ball" and o.track_id == ball_track_id), None)
        or next((o for o in frame_obs if o.label == "ball"), None)
    )


def _closest_player(
    ball: TrackObservation,
    players: list[TrackObservation],
    frame_index: int,
    video: VideoMeta,
    use_pitch: bool,
    player_pitch_pos: dict | None,
    ball_pitch_pos: dict | None,
) -> tuple[TrackObservation, float]:
    """Return (closest_player, distance) in cm when pitch coords are available, else px."""
    if use_pitch and player_pitch_pos is not None and ball_pitch_pos is not None:
        bpos = ball_pitch_pos.get(frame_index)
        fpos = player_pitch_pos.get(frame_index, {})
        if bpos is not None:
            def dist_cm(p: TrackObservation) -> float:
                pp = fpos.get(p.track_id)
                return math.hypot(bpos[0] - pp[0], bpos[1] - pp[1]) if pp is not None else 1e9
            closest = min(players, key=dist_cm)
            return closest, dist_cm(closest)

    # Pixel-space fallback — normalised by player box height to correct for camera pan/zoom.
    # A player whose box is 140 px tall is ~2× closer in real space than one at 70 px, so
    # their raw pixel distance is scaled down by the same factor before comparing thresholds.
    def dist_px(p: TrackObservation) -> float:
        dx = (p.bbox.cx - ball.bbox.cx) * video.width
        dy = (p.bbox.cy - ball.bbox.cy) * video.height
        raw = math.hypot(dx, dy)
        p_h = (p.bbox.y2 - p.bbox.y1) * video.height
        if p_h >= 15.0:
            return raw * (_STANDARD_PLAYER_HEIGHT_PX / p_h)
        return raw

    closest = min(players, key=dist_px)
    return closest, dist_px(closest)


def _compute_ball_velocity(
    by_frame: dict[int, list[TrackObservation]],
    sorted_frames: list[int],
    ball_track_id: int,
    video: VideoMeta,
    ball_pitch_pos: dict[int, tuple[float, float]] | None,
    use_pitch: bool,
) -> dict[int, tuple[float, float]]:
    """
    Returns velocity per frame.
    - cm/s when pitch positions are available
    - px/frame when falling back to pixel space
    """
    vels: dict[int, tuple[float, float]] = {}
    prev_pos: tuple[float, float] | None = None
    prev_f:   int | None                 = None

    for f in sorted_frames:
        ball = (
            next((o for o in by_frame[f] if o.label == "ball" and o.track_id == ball_track_id), None)
            or next((o for o in by_frame[f] if o.label == "ball"), None)
        )
        if ball is None:
            continue

        if use_pitch and ball_pitch_pos is not None:
            curr = ball_pitch_pos.get(f)
        else:
            curr = (ball.bbox.cx * video.width, ball.bbox.cy * video.height)

        if curr is not None and prev_pos is not None and prev_f is not None:
            gap_frames = max(f - prev_f, 1)
            if use_pitch:
                # cm/s: divide by elapsed seconds
                gap_s = gap_frames / video.fps
                vx = (curr[0] - prev_pos[0]) / gap_s
                vy = (curr[1] - prev_pos[1]) / gap_s
                if math.hypot(vx, vy) <= _MAX_BALL_SPEED_CMS:
                    vels[f] = (vx, vy)
            else:
                # px/frame: divide by gap in frames
                vx = (curr[0] - prev_pos[0]) / gap_frames
                vy = (curr[1] - prev_pos[1]) / gap_frames
                vels[f] = (vx, vy)

        if curr is not None:
            prev_pos, prev_f = curr, f

    return vels


def _direction_change_deg(vel_history: list[tuple[float, float]]) -> float:
    if len(vel_history) < 2:
        return 0.0
    mid    = max(len(vel_history) // 2, 1)
    before = vel_history[:mid]
    after  = vel_history[mid:]
    vb = (sum(v[0] for v in before) / len(before), sum(v[1] for v in before) / len(before))
    va = (sum(v[0] for v in after)  / len(after),  sum(v[1] for v in after)  / len(after))
    mb = math.hypot(*vb)
    ma = math.hypot(*va)
    if mb < 1e-6 or ma < 1e-6:
        return 0.0
    cos_a = max(-1.0, min(1.0, (vb[0] * va[0] + vb[1] * va[1]) / (mb * ma)))
    return math.degrees(math.acos(cos_a))


def _is_high_arc(y_positions: list[float], min_arc_px: float) -> bool:
    """True if ball followed a parabolic arc: went up (y decreased) then came down.
    The peak (minimum y in image-down coords) must occur in the middle 15-85% of flight.
    """
    if len(y_positions) < 5:
        return False
    y_range = max(y_positions) - min(y_positions)
    if y_range < min_arc_px:
        return False
    min_idx = y_positions.index(min(y_positions))
    mid_lo  = int(len(y_positions) * 0.15)
    mid_hi  = int(len(y_positions) * 0.85)
    return mid_lo <= min_idx <= mid_hi


def _dedupe_dense_events(events: list[Event], min_frame_gap: int = 6) -> list[Event]:
    filtered:  list[Event]                               = []
    last_seen: dict[tuple[str, int | None, int | None], int] = {}
    for event in events:
        key    = (event.event_type, event.actor_track_id, event.target_track_id)
        last_f = last_seen.get(key)
        if last_f is None or event.frame_index - last_f >= min_frame_gap:
            filtered.append(event)
            last_seen[key] = event.frame_index
    return filtered
