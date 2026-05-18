from __future__ import annotations

import math
from collections import defaultdict
from enum import Enum

from ez_worker.schemas import Event, TrackObservation, VideoMeta

# Ball cannot physically exceed 50 m/s — reject tracking noise in pitch mode
_MAX_BALL_SPEED_CMS = 5000.0
# ~60 m/s max × 40 px/m at midfield zoom / 25fps ≈ 96 px/frame.
# Anything above this is a tracker jump, not a real kick.
_MAX_BALL_SPEED_PX_FRAME = 100.0
# Typical player bbox height at standard broadcast midfield distance.
_STANDARD_PLAYER_HEIGHT_PX = 70.0


class _Phase(Enum):
    IDLE      = "idle"
    POSSESSED = "possessed"
    IN_FLIGHT = "in_flight"


def detect_events(
    tracks: list[TrackObservation],
    video: VideoMeta,
    ball_track_id: int,
    ball_pitch_pos: dict[int, tuple[float, float]] | None = None,
    player_pitch_pos: dict[int, dict[int, tuple[float, float]]] | None = None,
    # ── Possession ─────────────────────────────────────────────────────────
    possession_distance_threshold_cm: float = 200.0,
    possession_distance_threshold_px: float = 50.0,
    possession_min_seconds: float = 0.15,
    # ── Pass / Shot speeds ──────────────────────────────────────────────────
    pass_min_speed_cms: float = 250.0,
    pass_min_speed_px_per_s: float = 150.0,
    shot_min_speed_cms: float = 1200.0,
    shot_min_speed_px_per_s: float = 600.0,
    shot_no_catch_seconds: float = 1.0,
    # ── Direct-possession pass (direction change) ───────────────────────────
    ball_direction_change_min_deg: float = 55.0,
    # ── Clearance / high ball ───────────────────────────────────────────────
    clearance_min_flight_seconds: float = 1.2,
    clearance_min_arc_frac: float = 0.04,
    # ── Quality gates ───────────────────────────────────────────────────────
    # Min frames ball must be in flight before any reception is accepted
    pass_min_flight_frames: int = 3,
    # Min frames owner held ball BEFORE the kick (evaluated at launch time)
    owner_min_possession_frames: int = 6,
    # Min ball travel distance (launch pos → reception pos, normalized px)
    pass_min_ball_travel_px: float = 60.0,
    # Disabled by default (1.0 = always pass). Kept as tuning knob only.
    reception_decel_fraction: float = 1.0,
    # ── Scene changes ───────────────────────────────────────────────────────
    scene_changes: set[int] | None = None,
    # ── Explicit track exclusions (e.g. referees not caught by source_label) ─
    excluded_track_ids: set[int] | None = None,
    # Min ball speed (px/s) required to emit a TOUCH from IDLE state.
    # Prevents false touches when the ball is static near multiple players (kickoff).
    touch_min_ball_speed_px_per_s: float = 30.0,
) -> list[Event]:
    """
    Three-phase event detection state machine.

    IDLE → POSSESSED: ball near player for poss_min_frames → emit TOUCH
    POSSESSED → IN_FLIGHT: ball leaves at speed >= pass_min_speed (launch logged)
    IN_FLIGHT → POSSESSED: receiver confirmed AND all quality gates pass → emit PASS/INTERCEPTION
    IN_FLIGHT → IDLE: timeout with no receiver + fast launch → emit SHOT or CLEARANCE
    POSSESSED → POSSESSED: direct takeover (direction change + avg speed + ownership) → emit PASS
    """
    use_pitch = ball_pitch_pos is not None and player_pitch_pos is not None

    poss_min_frames = max(1, int(possession_min_seconds * video.fps))
    shot_no_catch_f = max(1, int(shot_no_catch_seconds * video.fps))

    pass_speed  = pass_min_speed_cms if use_pitch else (pass_min_speed_px_per_s / video.fps)
    shot_speed  = shot_min_speed_cms if use_pitch else (shot_min_speed_px_per_s / video.fps)
    poss_dist   = possession_distance_threshold_cm if use_pitch else possession_distance_threshold_px
    touch_speed = touch_min_ball_speed_px_per_s / video.fps  # px/frame threshold

    by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    for obs in tracks:
        by_frame[obs.frame_index].append(obs)
    sorted_frames = sorted(by_frame)

    ball_vel = _compute_ball_velocity(
        by_frame, sorted_frames, ball_track_id, video, ball_pitch_pos, use_pitch
    )

    # ── State variables ────────────────────────────────────────────────────
    phase       = _Phase.IDLE
    owner_tid:  int | None = None
    owner_team: int | None = None
    owner_since_frame: int | None = None  # when this owner confirmed possession
    # True if current owner received the ball from IN_FLIGHT (vs loose ball or direction-change).
    # Used to require longer possession before a shot is credited.
    owner_from_flight: bool = False
    # Track previous owner to suppress rapid back-and-forth ping-pong events.
    prev_owner_tid: int | None = None

    candidate_tid:   int | None = None
    candidate_count: int        = 0

    flight_frame: int | None   = None
    flight_speed: float        = 0.0
    flight_ball_y: list[float] = []
    # Ball's normalized position at IN_FLIGHT entry — used for ball travel distance check
    launch_ball_cx: float | None = None
    launch_ball_cy: float | None = None

    vel_history: list[tuple[float, float]] = []
    vel_window = max(3, int(video.fps * 0.25))  # 250 ms

    scene_changes = scene_changes or set()
    events: list[Event] = []

    for frame_index in sorted_frames:
        frame_obs = by_frame[frame_index]

        if frame_index in scene_changes:
            phase             = _Phase.IDLE
            owner_tid         = owner_team = None
            owner_since_frame = None
            owner_from_flight = False
            prev_owner_tid    = None
            candidate_tid     = None
            candidate_count   = 0
            flight_frame      = None
            launch_ball_cx    = launch_ball_cy = None
            vel_history.clear()
            continue

        ball    = _get_ball(frame_obs, ball_track_id)
        players = [
            o for o in frame_obs
            if o.label in ("player", "goalkeeper")
            and o.source_label != "referee"
            and (excluded_track_ids is None or o.track_id not in excluded_track_ids)
        ]

        vel = ball_vel.get(frame_index)
        if vel is not None:
            vel_history.append(vel)
            if len(vel_history) > vel_window:
                vel_history.pop(0)
        speed = math.hypot(*vel) if vel else 0.0

        # ══════════════════════════════════════════════════════════════════
        # Phase: IN_FLIGHT
        # ══════════════════════════════════════════════════════════════════
        if phase == _Phase.IN_FLIGHT:
            if ball is not None:
                flight_ball_y.append(ball.bbox.cy * video.height)

            if ball is not None and players:
                closest, d = _closest_player(
                    ball, players, frame_index, video,
                    use_pitch, player_pitch_pos, ball_pitch_pos
                )
                if d <= poss_dist:
                    # Deceleration gate: ball must slow below fraction of launch speed
                    # before reception frames accumulate. Prevents fly-through misdetections.
                    decel_thresh = flight_speed * reception_decel_fraction
                    ball_decelerating = (speed <= decel_thresh) or (flight_speed < pass_speed * 2)

                    if ball_decelerating:
                        if candidate_tid == closest.track_id:
                            candidate_count += 1
                        else:
                            candidate_tid   = closest.track_id
                            candidate_count = 1
                    else:
                        # Ball still moving fast through this player's zone — reset
                        candidate_tid   = None
                        candidate_count = 0

                    if candidate_count >= poss_min_frames:
                        flight_elapsed = frame_index - (flight_frame or frame_index)

                        # Gate 1: self-reception
                        if closest.track_id == owner_tid:
                            phase             = _Phase.POSSESSED
                            owner_since_frame = frame_index
                            owner_from_flight = True
                            flight_frame      = None
                            flight_ball_y     = []
                            launch_ball_cx    = launch_ball_cy = None
                            candidate_tid     = None
                            candidate_count   = 0
                            continue

                        # Gate 2: minimum flight duration
                        if flight_elapsed < pass_min_flight_frames:
                            continue

                        # Gate 3: owner must have held ball long enough BEFORE kicking
                        # Evaluated at launch time (flight_frame), not reception time.
                        owned_at_launch = (flight_frame or frame_index) - (owner_since_frame or flight_frame or frame_index)
                        if owned_at_launch < owner_min_possession_frames:
                            # Silent possession transfer — no event
                            phase             = _Phase.POSSESSED
                            prev_owner_tid    = owner_tid
                            owner_tid         = closest.track_id
                            owner_team        = closest.team_id
                            owner_since_frame = frame_index
                            owner_from_flight = True
                            flight_frame      = None
                            flight_ball_y     = []
                            launch_ball_cx    = launch_ball_cy = None
                            candidate_tid     = None
                            candidate_count   = 0
                            continue

                        # Gate 4: ball must have traveled meaningful distance
                        if launch_ball_cx is not None and not use_pitch:
                            curr_bx = (ball.bbox.x1 + ball.bbox.x2) / 2 * video.width
                            curr_by = (ball.bbox.y1 + ball.bbox.y2) / 2 * video.height
                            raw_travel = math.hypot(
                                curr_bx - launch_ball_cx * video.width,
                                curr_by - launch_ball_cy * video.height,  # type: ignore[operator]
                            )
                            p_h = (closest.bbox.y2 - closest.bbox.y1) * video.height
                            ball_travel = (
                                raw_travel * (_STANDARD_PLAYER_HEIGHT_PX / p_h)
                                if p_h >= 15.0 else raw_travel
                            )
                            if ball_travel < pass_min_ball_travel_px:
                                phase             = _Phase.POSSESSED
                                prev_owner_tid    = owner_tid
                                owner_tid         = closest.track_id
                                owner_team        = closest.team_id
                                owner_since_frame = frame_index
                                owner_from_flight = True
                                flight_frame      = None
                                flight_ball_y     = []
                                launch_ball_cx    = launch_ball_cy = None
                                candidate_tid     = None
                                candidate_count   = 0
                                continue

                        # All gates passed — confirmed reception
                        flight_s = flight_elapsed / video.fps
                        is_high  = _is_high_arc(flight_ball_y, video.height * clearance_min_arc_frac)
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
                                "flight_frames": flight_elapsed,
                            },
                        ))
                        phase             = _Phase.POSSESSED
                        prev_owner_tid    = owner_tid
                        owner_tid         = closest.track_id
                        owner_team        = closest.team_id
                        owner_since_frame = frame_index
                        owner_from_flight = True
                        candidate_tid     = None
                        candidate_count   = 0
                        flight_frame      = None
                        flight_ball_y     = []
                        launch_ball_cx    = launch_ball_cy = None
                        continue
                else:
                    candidate_tid   = None
                    candidate_count = 0

            # Shot / clearance timeout.
            # Fire only when: (a) ball has truly stopped moving (speed < 30 px/s = ~static),
            # (b) ball disappeared, or (c) 8-second hard cap exceeded.
            # This lets slow rolling passes reach their target without timing out early.
            if flight_frame is not None and frame_index - flight_frame >= shot_no_catch_f:
                _ball_truly_stopped = (ball is None) or (speed < 30.0 / video.fps)
                _hard_cap = frame_index - flight_frame >= int(video.fps * 8.0)
                if _ball_truly_stopped or _hard_cap:
                    flight_s  = (frame_index - flight_frame) / video.fps
                    is_high   = _is_high_arc(flight_ball_y, video.height * clearance_min_arc_frac)
                    speed_key = "launch_speed_cms" if use_pitch else "launch_speed_pxs"

                    # Require the shooter to have genuinely possessed the ball.
                    # Players who received from a flight ball and immediately re-kicked
                    # (owner_from_flight=True) need longer possession to earn a shot credit —
                    # prevents false shots when a ball grazes a player mid-flight.
                    owned_at_launch = flight_frame - (owner_since_frame or flight_frame)
                    min_for_shot = owner_min_possession_frames * 2 if owner_from_flight else owner_min_possession_frames
                    fire_event = owned_at_launch >= min_for_shot

                    if fire_event:
                        if flight_s >= clearance_min_flight_seconds and is_high:
                            events.append(Event(
                                frame_index=flight_frame,
                                time_seconds=round(flight_frame / video.fps, 2),
                                event_type="clearance",
                                actor_track_id=owner_tid,
                                details={speed_key: round(flight_speed, 1), "flight_seconds": round(flight_s, 2)},
                            ))
                        elif flight_speed >= shot_speed:
                            events.append(Event(
                                frame_index=flight_frame,
                                time_seconds=round(flight_frame / video.fps, 2),
                                event_type="shot_attempt",
                                actor_track_id=owner_tid,
                                details={speed_key: round(flight_speed, 1)},
                            ))
                    phase             = _Phase.IDLE
                    owner_tid         = owner_team = None
                    owner_since_frame = None
                    owner_from_flight = False
                    prev_owner_tid    = None
                    flight_frame      = None
                    flight_ball_y     = []
                    launch_ball_cx    = launch_ball_cy = None
                    candidate_tid     = None
                    candidate_count   = 0

            continue

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
            if phase == _Phase.POSSESSED and speed >= pass_speed:
                # Store ball position at launch for travel distance check
                if ball is not None:
                    launch_ball_cx = (ball.bbox.x1 + ball.bbox.x2) / 2
                    launch_ball_cy = (ball.bbox.y1 + ball.bbox.y2) / 2
                else:
                    launch_ball_cx = launch_ball_cy = None
                phase        = _Phase.IN_FLIGHT
                flight_frame = frame_index
                flight_speed = speed
                flight_ball_y   = []
                candidate_tid   = None
                candidate_count = 0
            elif phase == _Phase.POSSESSED:
                phase             = _Phase.IDLE
                owner_tid         = owner_team = None
                owner_since_frame = None
            continue

        # Ball within possession zone
        if candidate_tid == closest.track_id:
            candidate_count += 1
        else:
            candidate_tid   = closest.track_id
            candidate_count = 1

        if candidate_count < poss_min_frames:
            continue

        # ── Possession confirmed ────────────────────────────────────────
        if phase == _Phase.IDLE:
            # Reject if ball is rolling past this player (not toward them).
            # Handles kickoff bystanders: ball moves right toward T10 while T19
            # stands to its left — dot(vel, ball→T19) is negative → skip.
            if speed >= touch_speed and vel is not None:
                b2p_x = (closest.bbox.cx - (ball.bbox.x1 + ball.bbox.x2) / 2) * video.width
                b2p_y = (closest.bbox.cy - (ball.bbox.y1 + ball.bbox.y2) / 2) * video.height
                if vel[0] * b2p_x + vel[1] * b2p_y < 0:
                    candidate_tid   = None
                    candidate_count = 0
                    continue

            phase             = _Phase.POSSESSED
            owner_tid         = closest.track_id
            owner_team        = closest.team_id
            owner_since_frame = frame_index
            owner_from_flight = False
            # Only emit TOUCH when ball is actually moving (not static near a bystander).
            if speed >= touch_speed:
                dist_key = "distance_cm" if use_pitch else "distance_px"
                events.append(Event(
                    frame_index=frame_index,
                    time_seconds=round(frame_index / video.fps, 2),
                    event_type="touch",
                    actor_track_id=closest.track_id,
                    details={dist_key: round(d, 1)},
                ))

        elif phase == _Phase.POSSESSED and closest.track_id != owner_tid:
            # Pressing check: owner still in possession zone = presser, not takeover.
            # Exception: if candidate is dramatically closer than owner (dominant possession),
            # the candidate clearly has the ball — don't defer to the old owner.
            owner_obs = next((p for p in players if p.track_id == owner_tid), None)
            if owner_obs is not None:
                _, owner_d = _closest_player(
                    ball, [owner_obs], frame_index, video,
                    use_pitch, player_pitch_pos, ball_pitch_pos,
                )
                dominant = d * 2.5 < owner_d  # candidate >2.5× closer = clear takeover
                if owner_d <= poss_dist and not dominant:
                    continue

            # Owner lost the ball — check for direct-possession pass
            dir_change = _direction_change_deg(vel_history)
            avg_speed  = (
                sum(math.hypot(*v) for v in vel_history) / len(vel_history)
                if vel_history else 0.0
            )
            owned_frames = frame_index - (owner_since_frame or frame_index)

            # Suppress ping-pong: if the new "owner" is whoever JUST had the ball
            # before the current owner (rapid back-and-forth bounce), skip the event.
            # owned_frames is how long current owner held — if it's less than 1 second
            # and the ball is bouncing back to the previous owner, it's tracking noise.
            _is_pingpong = (
                closest.track_id == prev_owner_tid
                and owned_frames < int(video.fps)
            )
            if (
                dir_change >= ball_direction_change_min_deg
                and avg_speed >= pass_speed
                and owned_frames >= owner_min_possession_frames
                and not _is_pingpong
            ):
                etype = "pass" if closest.team_id == owner_team else "interception"
                events.append(Event(
                    frame_index=frame_index,
                    time_seconds=round(frame_index / video.fps, 2),
                    event_type=etype,
                    actor_track_id=owner_tid,
                    target_track_id=closest.track_id,
                    details={"direction_change_deg": round(dir_change, 1)},
                ))
            # Backdate possession start to when player first became candidate —
            # a one-touch pass confirmation at frame N means ball was with them
            # since frame N-(candidate_count-1).
            prev_owner_tid    = owner_tid
            owner_tid         = closest.track_id
            owner_team        = closest.team_id
            owner_since_frame = frame_index - (candidate_count - 1)
            owner_from_flight = False

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
    """Return (closest_player, distance) in cm (pitch) or normalized px (pixel fallback)."""
    if use_pitch and player_pitch_pos is not None and ball_pitch_pos is not None:
        bpos = ball_pitch_pos.get(frame_index)
        fpos = player_pitch_pos.get(frame_index, {})
        if bpos is not None:
            def dist_cm(p: TrackObservation) -> float:
                pp = fpos.get(p.track_id)
                return math.hypot(bpos[0] - pp[0], bpos[1] - pp[1]) if pp is not None else 1e9
            closest = min(players, key=dist_cm)
            return closest, dist_cm(closest)

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


def _nearest_players(
    ball: TrackObservation,
    players: list[TrackObservation],
    frame_index: int,
    video: VideoMeta,
    use_pitch: bool,
    player_pitch_pos: dict | None,
    ball_pitch_pos: dict | None,
    n: int = 5,
) -> list[tuple[TrackObservation, float]]:
    """Return up to n players sorted by distance from ball (closest first)."""
    if not players:
        return []
    scored = [
        (p, _closest_player(ball, [p], frame_index, video, use_pitch, player_pitch_pos, ball_pitch_pos)[1])
        for p in players
    ]
    scored.sort(key=lambda x: x[1])
    return scored[:n]


def _compute_ball_velocity(
    by_frame: dict[int, list[TrackObservation]],
    sorted_frames: list[int],
    ball_track_id: int,
    video: VideoMeta,
    ball_pitch_pos: dict[int, tuple[float, float]] | None,
    use_pitch: bool,
) -> dict[int, tuple[float, float]]:
    """Returns velocity per frame: cm/s (pitch mode) or px/frame (pixel fallback)."""
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

        curr = (
            ball_pitch_pos.get(f) if (use_pitch and ball_pitch_pos) else
            ((ball.bbox.x1 + ball.bbox.x2) / 2 * video.width,
             (ball.bbox.y1 + ball.bbox.y2) / 2 * video.height)
        )

        if curr is not None and prev_pos is not None and prev_f is not None:
            gap_frames = max(f - prev_f, 1)
            if use_pitch:
                gap_s = gap_frames / video.fps
                vx = (curr[0] - prev_pos[0]) / gap_s
                vy = (curr[1] - prev_pos[1]) / gap_s
                if math.hypot(vx, vy) <= _MAX_BALL_SPEED_CMS:
                    vels[f] = (vx, vy)
            else:
                vx = (curr[0] - prev_pos[0]) / gap_frames
                vy = (curr[1] - prev_pos[1]) / gap_frames
                if math.hypot(vx, vy) <= _MAX_BALL_SPEED_PX_FRAME:
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
