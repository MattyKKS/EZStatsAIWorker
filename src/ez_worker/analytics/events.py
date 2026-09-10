from __future__ import annotations

import math
from collections import defaultdict
from enum import Enum

from ez_worker.schemas import Event, TrackObservation, VideoMeta

# Ball cannot physically exceed 50 m/s — reject tracking noise in pitch mode
_MAX_BALL_SPEED_CMS = 5000.0
# ~60 m/s max × 40 px/m at midfield zoom / 25fps ≈ 96 px/frame.
_MAX_BALL_SPEED_PX_FRAME = 100.0
# Typical player bbox height at standard broadcast midfield distance.
_STANDARD_PLAYER_HEIGHT_PX = 70.0
# Velocity smoothing window (frames). Median over this window before any speed gate.
_VEL_SMOOTH_WINDOW = 5


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
    possession_distance_threshold_px: float = 80.0,   # ~2 m in normalised px
    possession_min_seconds: float = 0.16,              # lowered: 0.25 → 0.16 (4 frames); faster re-acquisition after clearances
    # ── Pass / Shot speeds ──────────────────────────────────────────────────
    pass_min_speed_cms: float = 250.0,
    pass_min_speed_px_per_s: float = 350.0,            # raised: 150 → 350
    shot_min_speed_cms: float = 1200.0,
    shot_min_speed_px_per_s: float = 600.0,
    shot_no_catch_seconds: float = 2.5,               # raised: 1.0 → 2.5
    # ── Direct-possession pass (POSSESSED→POSSESSED) ─────────────────────
    ball_direction_change_min_deg: float = 55.0,
    enable_direct_possession_path: bool = False,       # disabled — too noisy
    # ── Clearance / high ball ───────────────────────────────────────────────
    clearance_min_flight_seconds: float = 1.2,
    clearance_min_arc_frac: float = 0.04,
    # ── Quality gates ───────────────────────────────────────────────────────
    pass_min_flight_frames: int = 4,                   # lowered: 6 → 4; catches short passes ≥ 0.16s in flight
    owner_min_possession_frames: int = 5,              # lowered: 8 → 5 (0.2s; first-touch path handles very short holds)
    pass_min_ball_travel_px: float = 60.0,
    reception_decel_fraction: float = 0.65,            # enabled: 1.0 → 0.65
    # ── Scene changes ───────────────────────────────────────────────────────
    scene_changes: set[int] | None = None,
    excluded_track_ids: set[int] | None = None,
    touch_min_ball_speed_px_per_s: float = 80.0,      # raised: 30 → 80
    possession_by_frame: dict[int, int | None] | None = None,
    allow_aerial_contacts: bool = True,
) -> list[Event]:
    """
    Three-phase event detection state machine.

    IDLE → POSSESSED: ball near player feet for poss_min_frames → emit TOUCH
    POSSESSED → IN_FLIGHT: ball leaves at speed >= pass_min_speed (launch logged)
    IN_FLIGHT → POSSESSED: receiver confirmed (grounded ball), all quality gates → emit PASS/INTERCEPTION
    IN_FLIGHT → IDLE: timeout with no receiver + fast launch → emit SHOT or CLEARANCE
    POSSESSED → POSSESSED: disabled by default (enable_direct_possession_path=True to re-enable)

    Key improvements over v1:
    - Distance uses player FEET (y2) and ball BOTTOM (y2), not bbox centers
    - Velocity smoothed over 5-frame median window before speed gates
    - Airborne ball check: possession/reception gates ignored when ball pixel
      bottom is significantly above nearby player feet — prevents airborne
      detector noise from triggering false passes
    - Direct-possession path disabled (was the biggest source of false events)
    - All thresholds raised to require stronger evidence
    """
    use_pitch = ball_pitch_pos is not None and player_pitch_pos is not None

    poss_min_frames = max(1, int(possession_min_seconds * video.fps))
    shot_no_catch_f = max(1, int(shot_no_catch_seconds * video.fps))

    pass_speed  = pass_min_speed_cms if use_pitch else (pass_min_speed_px_per_s / video.fps)
    shot_speed  = shot_min_speed_cms if use_pitch else (shot_min_speed_px_per_s / video.fps)
    poss_dist   = possession_distance_threshold_cm if use_pitch else possession_distance_threshold_px
    # In pitch mode speed is cm/s; keep touch threshold in matching units.
    # 80 px/s ÷ 40 px/m × 100 cm/m ≈ 200 cm/s — gentle touch / rolling ball.
    touch_speed = 200.0 if use_pitch else (touch_min_ball_speed_px_per_s / video.fps)

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
    owner_since_frame: int | None = None
    owner_from_flight: bool = False
    prev_owner_tid: int | None = None

    candidate_tid:   int | None = None
    candidate_count: int        = 0

    flight_frame: int | None   = None
    flight_speed: float        = 0.0
    flight_ball_y: list[float] = []
    launch_ball_cx: float | None = None
    launch_ball_cy: float | None = None

    vel_history: list[tuple[float, float]] = []
    vel_window = max(3, int(video.fps * 0.25))  # 250 ms

    airborne_streak: int = 0
    _max_airborne_idle   = max(3, int(2.0 * video.fps))   # 2s — misdetection / stands cap
    _max_airborne_flight = max(3, int(6.0 * video.fps))   # 6s — genuine clearance flight

    # Possession linger: when POSSESSED and ball drifts far at low speed (tracker
    # gap / bouncing ball), hold possession for up to _poss_linger_max frames so a
    # reappearing fast ball can still be attributed to the same owner.
    poss_linger: int       = 0
    _poss_linger_max       = max(15, int(0.6 * video.fps))  # ~15 frames at 25fps — covers tracker gaps
    # When IN_FLIGHT starts via the linger path the first N frames of flight_ball_y
    # are missing — arc data is unreliable, so we skip long_ball classification.
    _linger_launch: bool   = False

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
            airborne_streak   = 0
            poss_linger       = 0
            _linger_launch    = False
            vel_history.clear()
            continue

        ball    = _get_ball(frame_obs, ball_track_id)
        if ball is None or ball.is_interpolated:
            candidate_tid = None
            candidate_count = 0
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
        if possession_by_frame is not None:
            possession_by_frame[frame_index] = None
            owner_obs = next((p for p in players if p.track_id == owner_tid), None)
            if phase == _Phase.POSSESSED and ball is not None and not ball.is_interpolated and owner_obs is not None:
                _, owner_distance = _closest_player(ball, [owner_obs], frame_index, video,
                                                    use_pitch, player_pitch_pos, ball_pitch_pos)
                if owner_distance <= poss_dist:
                    possession_by_frame[frame_index] = owner_team

        # Airborne flag for this frame — used to suppress possession/reception
        # when the ball pixel is floating above player feet.  Always computed
        # in pixel space (regardless of use_pitch) because in pitch mode the
        # ground-plane homography projects an airborne ball to a WRONG pitch
        # position, causing missed receptions and spurious possession changes.
        ball_airborne = (
            ball is not None
            and players
            and _is_ball_airborne(ball, players, video)
        )
        # Temporal cap: clears false airborne latch.
        # 6s cap during IN_FLIGHT (genuine clearances can stay aloft that long);
        # 2s cap in all other phases (stands / pixel misdetection).
        _airborne_cap = _max_airborne_flight if phase == _Phase.IN_FLIGHT else _max_airborne_idle
        if ball_airborne:
            airborne_streak += 1
            if airborne_streak > _airborne_cap:
                ball_airborne = False
        else:
            airborne_streak = 0

        # Height ratio: 0.0 = ball at feet, 1.0 = ball at head-top, >1.0 = above head.
        # Computed whenever ball is airborne — used to distinguish play-height contacts
        # (knee / waist / chest / header, height < 0.85) from truly in-flight balls
        # (height >= 0.85, ball above head level, no contact possible).
        _ball_hr       = (
            _ball_height_ratio(ball, players, video)
            if (ball_airborne and ball is not None and players)
            else 0.0
        )
        # >= 1.10: ball is above full-arm reach (no contact possible for a standing player).
        # Headers happen at hr ≈ 0.85-1.05; this threshold keeps them in the play-height zone.
        _truly_airborne = ball_airborne and (not allow_aerial_contacts or _ball_hr >= 1.10)

        # ══════════════════════════════════════════════════════════════════
        # Phase: IN_FLIGHT
        # ══════════════════════════════════════════════════════════════════
        if phase == _Phase.IN_FLIGHT:
            if ball is not None:
                flight_ball_y.append(ball.bbox.cy * video.height)

            # Estimate landing frame via parabolic arc symmetry; relax thresholds
            # in a ±1.2s window around the predicted landing to catch reception even
            # when the ball decelerates slowly or the closest-player flickers.
            _predicted_landing = _predict_landing_frame(
                flight_ball_y, flight_frame if flight_frame is not None else frame_index, video.fps
            )
            _landing_window = int(1.2 * video.fps)
            _near_landing   = (
                _predicted_landing is not None
                and abs(frame_index - _predicted_landing) <= _landing_window
            )
            # After shot_no_catch_f frames in flight the ball must be descending —
            # treat the same as near-landing: halve the confirmation window and
            # relax the deceleration gate so slow approaches still fire.
            _long_flight    = (
                flight_frame is not None
                and frame_index - flight_frame >= shot_no_catch_f
            )
            # Relax thresholds when:
            #   (a) ball is grounded (not ball_airborne), OR
            #   (b) ball is airborne but at play height with nearby players
            #       (_truly_airborne=False AND _ball_hr > 0.0 means a player is
            #        close enough to measure height — not an isolated mid-arc ball).
            # Case (b) catches goalkeepers receiving a ball that is still officially
            # "airborne" as it descends into their feet.
            # Case (a) prevents pixel-projection false receptions when the ball is
            # high in a clearance arc with no nearby players (_ball_hr would be 0.0).
            _grounded_or_playheight = (
                not ball_airborne
                or (not _truly_airborne and _ball_hr > 0.0)
            )
            _landing_relaxed = (_near_landing or _long_flight) and _grounded_or_playheight
            _eff_poss_min = max(1, poss_min_frames // 2) if _landing_relaxed else poss_min_frames

            if ball is not None and players:
                # When ball is airborne in pitch mode, homography projects to a
                # wrong ground position — fall back to pixel distance.
                # For play-height aerial (header/volley), use horizontal-only pixel
                # distance since the vertical gap to player feet is expected and large.
                if ball_airborne and use_pitch:
                    closest, d = _closest_player(
                        ball, players, frame_index, video, False, None, None
                    )
                    _eff_poss_dist = possession_distance_threshold_px
                else:
                    closest, d = _closest_player(
                        ball, players, frame_index, video,
                        use_pitch, player_pitch_pos, ball_pitch_pos
                    )
                    _eff_poss_dist = poss_dist
                # Near landing or long-running flight (AND ball at play height):
                # double the distance threshold — ball may still be 1-2 m away.
                if _landing_relaxed:
                    _eff_poss_dist *= 2
                if d <= _eff_poss_dist and not ball.is_interpolated:
                    _fc_thresh   = 30.0 if use_pitch else 20.0
                    _flight_close = d <= _fc_thresh and not ball_airborne

                    if _flight_close:
                        if speed <= flight_speed * reception_decel_fraction:
                            candidate_count = candidate_count + 1 if candidate_tid == closest.track_id else 1
                            candidate_tid = closest.track_id
                        else:
                            candidate_tid   = None
                            candidate_count = 0
                    else:
                        decel_thresh = flight_speed * reception_decel_fraction
                        # When flight is long or near predicted landing AND ball is
                        # at play height (not truly airborne), accept slow approaches
                        # as decelerating — covers goalkeeper distributions where the
                        # ball barely drops below the standard decel threshold.
                        # Cap at 1500 cm/s (15 m/s): a ball still moving faster than
                        # that cannot be received — it's either still in flight or a
                        # tracking artifact near a player mid-arc.
                        _max_land_speed = min(flight_speed * 1.2, 1500.0)
                        _landing_catch = (
                            _landing_relaxed
                            and d <= 100.0
                            and speed <= _max_land_speed
                        )
                        ball_decelerating = (
                            (speed <= decel_thresh)
                            or (flight_speed < pass_speed * 2)
                            or _landing_catch
                        )

                        if ball_decelerating and (not ball_airborne or not _truly_airborne):
                            # Count reception frames when ball is grounded OR at play
                            # height (knee/waist/chest/head) — only block when ball is
                            # well above head level (_truly_airborne).

                            if candidate_tid == closest.track_id:
                                candidate_count += 1
                            else:
                                # Directional filter (Priority 4): when considering a NEW
                                # candidate, skip them if they are clearly in the wrong
                                # direction (>120° off the launch-to-current travel vector).
                                # Applied only for NEW candidates so that a receiver who was
                                # correctly accumulated is never dropped because the ball
                                # slightly overshot their position.
                                _skip_dir = False
                                if launch_ball_cx is not None:
                                    _ball_cx  = (ball.bbox.x1 + ball.bbox.x2) / 2 * video.width
                                    _ball_cy  = ball.bbox.cy * video.height
                                    _travel_x = _ball_cx - launch_ball_cx * video.width
                                    _travel_y = _ball_cy - launch_ball_cy * video.height
                                    _travel_m = math.hypot(_travel_x, _travel_y)
                                    if _travel_m > 50:  # need 50px travel to have a reliable direction
                                        _cand_dx = (closest.bbox.x1 + closest.bbox.x2) / 2 * video.width - _ball_cx
                                        _cand_dy = closest.bbox.cy * video.height - _ball_cy
                                        _cand_m  = math.hypot(_cand_dx, _cand_dy) + 1e-6
                                        _cos = (_travel_x * _cand_dx + _travel_y * _cand_dy) / (_travel_m * _cand_m)
                                        if _cos < -0.5:  # >120 degrees — clearly wrong direction
                                            _skip_dir = True

                                if not _skip_dir:
                                    # Evidence belongs to this receiver only.
                                    candidate_count = 1
                                    candidate_tid   = closest.track_id
                        elif not ball_decelerating:
                            candidate_tid   = None
                            candidate_count = 0
                        # else: ball_airborne — don't reset or increment

                    if candidate_count >= _eff_poss_min:
                        flight_elapsed = frame_index - (flight_frame if flight_frame is not None else frame_index)

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

                        # Gate 3: owner must have held ball long enough BEFORE kicking.
                        # First-touch or short-possession kicks use a 1-frame minimum:
                        # (a) owner_from_flight=True: received via pass and immediately redirects
                        # (b) owned_at_launch ≤ 4 frames: goalkeeper or quick first touch
                        # 1 frame (not 2) because the zone-exit fix can set owned_at_launch=1
                        # when a first-touch player kicks on the very next frame.
                        launch = flight_frame if flight_frame is not None else frame_index
                        owned_at_launch = launch - (owner_since_frame if owner_since_frame is not None else launch)
                        _quick_kick = owner_from_flight or (owned_at_launch <= 4)
                        _min_own = max(1, int(video.fps * 0.04)) if _quick_kick else owner_min_possession_frames
                        if owned_at_launch < _min_own:
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
                        # Skip long_ball when flight started via linger: the first
                        # several frames of flight_ball_y are missing (ball was lost
                        # during the linger gap), so the arc measurement is unreliable.
                        if flight_s >= clearance_min_flight_seconds and is_high and not _linger_launch:
                            # Same team → intentional long ball; different team → clearance
                            etype = "long_ball" if closest.team_id == owner_team else "clearance"
                        else:
                            etype = "pass" if closest.team_id == owner_team else "interception"
                        if owner_team is None or closest.team_id is None:
                            etype = "ball_transfer"
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
                        _linger_launch    = False
                        continue
                else:
                    candidate_tid   = None
                    candidate_count = 0

            # Shot / clearance timeout.
            if flight_frame is not None and frame_index - flight_frame >= shot_no_catch_f:
                # Fire timeout when ball is rolling slowly (< 25% of pass speed) OR
                # hard-cap at 5s to prevent the machine being stuck in IN_FLIGHT for
                # an entire passing sequence while the ball keeps rolling.
                _ball_truly_stopped = (ball is None) or (speed < pass_speed * 0.25)
                _hard_cap = frame_index - flight_frame >= int(video.fps * 5.0)
                if _ball_truly_stopped or _hard_cap:
                    flight_s  = (frame_index - flight_frame) / video.fps
                    is_high   = _is_high_arc(flight_ball_y, video.height * clearance_min_arc_frac)
                    speed_key = "launch_speed_cms" if use_pitch else "launch_speed_pxs"

                    owned_at_launch = flight_frame - (owner_since_frame if owner_since_frame is not None else flight_frame)
                    _quick_kick_t   = owner_from_flight or (owned_at_launch <= 4)
                    min_for_shot    = max(1, int(video.fps * 0.04)) if _quick_kick_t else owner_min_possession_frames
                    fire_event      = owned_at_launch >= min_for_shot

                    if fire_event:
                        # Clearance requires a fast launch — a slow drifting ball
                        # with a long flight time is not a clearance.
                        _min_clearance_speed = pass_speed * 4  # ~1000 cm/s at 250 base
                        if flight_s >= clearance_min_flight_seconds and is_high and flight_speed >= _min_clearance_speed:
                            events.append(Event(
                                frame_index=flight_frame,
                                time_seconds=round(flight_frame / video.fps, 2),
                                event_type="clearance",
                                actor_track_id=owner_tid,
                                details={speed_key: round(flight_speed, 1), "flight_seconds": round(flight_s, 2)},
                            ))
                        elif flight_speed >= shot_speed and not _quick_kick_t:
                            # Quick redirects (owner_from_flight or ≤4 frames hold) are
                            # mid-field deflections, not deliberate shots — suppress shot_attempt.
                            events.append(Event(
                                frame_index=flight_frame,
                                time_seconds=round(flight_frame / video.fps, 2),
                                event_type="shot_attempt",
                                actor_track_id=owner_tid,
                                details={speed_key: round(flight_speed, 1)},
                            ))
                        elif _quick_kick_t and flight_speed >= pass_speed:
                            # Quick first-touch redirect — emit as touch rather than shot_attempt.
                            events.append(Event(
                                frame_index=flight_frame,
                                time_seconds=round(flight_frame / video.fps, 2),
                                event_type="touch",
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
                    # Pre-load streak so ground-mode possession fires immediately after
                    # the timeout: the ball is landing, not still in flight.
                    airborne_streak   = _max_airborne_idle + 1
                    _linger_launch    = False

            continue

        # ══════════════════════════════════════════════════════════════════
        # Phase: IDLE or POSSESSED
        # ══════════════════════════════════════════════════════════════════
        if ball is None or ball.is_interpolated or not players:
            continue

        # When ball is airborne, homography projects it to a WRONG pitch position
        # (camera sees the ball elevated, not on the ground plane).  Use pixel-space
        # distance as the reliable fallback for the possession-zone-exit check.
        if ball_airborne:
            if not use_pitch:
                # Pixel mode: original behaviour — speed gate guards the transition.
                if phase == _Phase.POSSESSED and speed >= pass_speed:
                    launch_ball_cx = (ball.bbox.x1 + ball.bbox.x2) / 2
                    launch_ball_cy = (ball.bbox.y1 + ball.bbox.y2) / 2
                    _linger_launch = False
                    phase        = _Phase.IN_FLIGHT
                    flight_frame = frame_index
                    flight_speed = speed
                    flight_ball_y   = []
                    candidate_tid   = None
                    candidate_count = 0
            else:
                # Pitch mode: cm distance is unreliable when airborne.
                # Pixel distance tells us if the ball has genuinely left the owner.
                _, d_px = _closest_player(ball, players, frame_index, video, False, None, None)
                if phase == _Phase.POSSESSED and d_px > possession_distance_threshold_px:
                    launch_ball_cx = (ball.bbox.x1 + ball.bbox.x2) / 2
                    launch_ball_cy = (ball.bbox.y1 + ball.bbox.y2) / 2
                    _linger_launch = False
                    phase        = _Phase.IN_FLIGHT
                    flight_frame = frame_index
                    flight_speed = speed
                    flight_ball_y   = []
                    candidate_tid   = None
                    candidate_count = 0
            if _truly_airborne:
                # Ball is well above head level — no aerial contact possible.
                continue
            # Ball is at play height (knee / waist / chest / header) — fall through
            # to possession logic below, but use pixel-space distance because the
            # homography ground-plane projection is unreliable for elevated balls.

        if ball_airborne and not _truly_airborne:
            # Play-height aerial contact path: horizontal-only pixel distance so the
            # vertical gap (ball at head height, feet at ground) does not swamp the
            # metric.  Height normalization is kept for perspective correction.
            closest, d = _closest_player(ball, players, frame_index, video, False, None, None, aerial=True)
            _near_ph        = (closest.bbox.y2 - closest.bbox.y1) * video.height
            _idle_poss_dist = max(40.0, _near_ph * 0.50)
        else:
            closest, d = _closest_player(
                ball, players, frame_index, video,
                use_pitch, player_pitch_pos, ball_pitch_pos
            )
            _idle_poss_dist = poss_dist

        # When POSSESSED, the zone-exit decision must use the OWNER's distance to
        # the ball, not the closest-player distance.  Without this, a ball flying
        # THROUGH another player's zone (e.g. player 12 at 84 cm while owner 537
        # is 264 cm away) would keep d low, block the IN_FLIGHT trigger, and let
        # that bystander silently steal possession via candidate accumulation.
        _d_for_exit = d
        if phase == _Phase.POSSESSED and owner_tid is not None:
            _owner_obs = next((p for p in players if p.track_id == owner_tid), None)
            if _owner_obs is not None:
                _aerial_mode = ball_airborne and not _truly_airborne
                _, _d_for_exit = _closest_player(
                    ball, [_owner_obs], frame_index, video,
                    False if _aerial_mode else use_pitch,
                    None if _aerial_mode else player_pitch_pos,
                    None if _aerial_mode else ball_pitch_pos,
                    aerial=_aerial_mode,
                )

        if _d_for_exit > _idle_poss_dist:
            if phase == _Phase.POSSESSED and speed >= pass_speed:
                if ball is not None:
                    launch_ball_cx = (ball.bbox.x1 + ball.bbox.x2) / 2
                    launch_ball_cy = (ball.bbox.y1 + ball.bbox.y2) / 2
                else:
                    launch_ball_cx = launch_ball_cy = None
                # If we were in the linger zone when speed appeared, the first
                # several frames of flight are missing — mark arc data as unreliable.
                _linger_launch = poss_linger > 0
                poss_linger    = 0
                phase          = _Phase.IN_FLIGHT
                flight_frame   = frame_index
                flight_speed   = speed
                flight_ball_y   = []
                candidate_tid   = None
                candidate_count = 0
            elif phase == _Phase.POSSESSED:
                # Possession linger: tracker sometimes loses the ball for a handful
                # of frames right at the moment of a kick (speed=0 while ball is
                # already in flight).  Hold possession for _poss_linger_max frames;
                # if the ball reappears at pass speed → IN_FLIGHT from same owner.
                poss_linger += 1
                if poss_linger >= _poss_linger_max:
                    phase             = _Phase.IDLE
                    owner_tid         = owner_team = None
                    owner_since_frame = None
                    poss_linger       = 0
            continue

        # Ball within possession zone — reset any linger counter
        poss_linger = 0

        # Speed guard (aerial only): block candidate accumulation only for true
        # shots/clearances (>= shot_speed).  Headers and volleys typically arrive
        # at 500–1100 cm/s and must be allowed through.  The old threshold
        # (pass_speed * 2 = 500 cm/s) incorrectly blocked all aerial contacts.
        if ball_airborne and not _truly_airborne and speed >= shot_speed:
            continue

        if candidate_tid == closest.track_id:
            candidate_count += 1
        else:
            candidate_tid   = closest.track_id
            candidate_count = 1

        # Close-contact fast path: ball physically at the player bypasses frame-count.
        # Threshold depends on measurement space:
        #   pitch cm (ground): 80 cm  (~player's feet / shin zone)
        #   pixel (ground):    40 px
        #   pixel (aerial):    25% of player bbox height (~17-20 px at midfield zoom)
        if ball_airborne and not _truly_airborne:
            _close_contact_thresh = max(15.0, _near_ph * 0.25)
        else:
            _close_contact_thresh = 80.0 if use_pitch else 40.0
        _close_contact = d <= _close_contact_thresh
        if _close_contact:
            # Fast-track to confirmation, but only when the ball is moving slowly
            # enough to be genuinely controlled (aerial contacts always fast-track;
            # ground contacts skip fast-track when ball is still fast — this filters
            # out 1-frame ball→shoe tracker glitches where the ball snaps to a
            # player's foot while still moving at pass speed).
            if candidate_count >= max(2, round(video.fps * 0.08)) and (ball_airborne or speed < pass_speed * 3):
                candidate_count = poss_min_frames

        if candidate_count < poss_min_frames:
            continue

        # ── Possession confirmed ────────────────────────────────────────
        if phase == _Phase.IDLE:
            if speed >= touch_speed and vel is not None and not _close_contact:
                # Velocity direction check: skip it when ball is in close contact
                # (ball at player's feet may be moving away because it was just
                # kicked — that IS a real touch). Only filter balls that are still
                # far away and clearly exiting the zone without being touched.
                b2p_x = (closest.bbox.cx - (ball.bbox.x1 + ball.bbox.x2) / 2) * video.width
                b2p_y = (closest.bbox.cy - (ball.bbox.y1 + ball.bbox.y2) / 2) * video.height
                if vel[0] * b2p_x + vel[1] * b2p_y < 0:
                    candidate_tid   = None
                    candidate_count = 0
                    continue

            phase             = _Phase.POSSESSED
            owner_tid         = closest.track_id
            owner_team        = closest.team_id
            # Back-date to when the ball first entered the possession zone, not when
            # confirmed.  Gate 3 measures actual possession duration — candidate_count
            # frames have already elapsed, so subtract them from the confirmation frame.
            # Exception: close-contact fast path forces candidate_count to poss_min_frames
            # even though possession only started this frame — don't back-date in that case.
            owner_since_frame = frame_index if _close_contact else frame_index - (candidate_count - 1)
            owner_from_flight = False
            if speed >= touch_speed:
                # Classify contact type by ball height ratio.
                # Aerial contacts use pixel distance — mark key accordingly.
                _aerial = ball_airborne and not _truly_airborne
                if _aerial:
                    # Height thresholds from anthropometric data (fraction of player height):
                    # header >= 0.75 (head), chest_control >= 0.50, volley < 0.50 (knee/shin)
                    if _ball_hr >= 0.75:
                        _contact_type = "header"
                    elif _ball_hr >= 0.50:
                        _contact_type = "chest_control"
                    else:
                        _contact_type = "volley"
                    dist_key = "distance_px"
                else:
                    _contact_type = "touch"
                    dist_key      = "distance_cm" if use_pitch else "distance_px"
                events.append(Event(
                    frame_index=frame_index,
                    time_seconds=round(frame_index / video.fps, 2),
                    event_type=_contact_type,
                    actor_track_id=closest.track_id,
                    details={dist_key: round(d, 1)},
                ))

        elif phase == _Phase.POSSESSED and closest.track_id != owner_tid:
            # ── Direct-possession pass (POSSESSED→POSSESSED) ───────────
            # Disabled by default: in broadcast football this path fires on
            # every dribble touch, ricochet, and bounce. Only re-enable with
            # enable_direct_possession_path=True when the data is clean.
            if enable_direct_possession_path:
                owner_obs = next((p for p in players if p.track_id == owner_tid), None)
                if owner_obs is not None:
                    _, owner_d = _closest_player(
                        ball, [owner_obs], frame_index, video,
                        use_pitch, player_pitch_pos, ball_pitch_pos,
                    )
                    dominant = d * 2.5 < owner_d
                    if owner_d <= _idle_poss_dist and not dominant:
                        continue

                dir_change = _direction_change_deg(vel_history)
                avg_speed  = (
                    sum(math.hypot(*v) for v in vel_history) / len(vel_history)
                    if vel_history else 0.0
                )
                owned_frames = frame_index - (owner_since_frame if owner_since_frame is not None else frame_index)
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
            else:
                # Path B disabled: just update possession silently
                owner_obs = next((p for p in players if p.track_id == owner_tid), None)
                if owner_obs is not None:
                    _, owner_d = _closest_player(
                        ball, [owner_obs], frame_index, video,
                        use_pitch, player_pitch_pos, ball_pitch_pos,
                    )
                    dominant = d * 2.5 < owner_d
                    if owner_d <= _idle_poss_dist and not dominant:
                        continue

            prev_owner_tid    = owner_tid
            owner_tid         = closest.track_id
            owner_team        = closest.team_id
            owner_since_frame = frame_index - (candidate_count - 1)
            owner_from_flight = False
            poss_linger       = 0

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
    aerial: bool = False,
) -> tuple[TrackObservation, float]:
    """Return (closest_player, distance) in cm (pitch) or normalised px (pixel fallback).

    aerial=True uses horizontal-only pixel distance, ignoring the vertical gap between
    ball bottom and player feet.  Correct for headers/volleys where the vertical gap
    is expected (ball at head height ≠ ball far away).  Height normalization is kept
    so perspective (far players appear smaller) is still corrected.
    """
    if use_pitch and not aerial and player_pitch_pos is not None and ball_pitch_pos is not None:
        bpos = ball_pitch_pos.get(frame_index)
        fpos = player_pitch_pos.get(frame_index, {})
        if bpos is not None:
            def dist_cm(p: TrackObservation) -> float:
                pp = fpos.get(p.track_id)
                return math.hypot(bpos[0] - pp[0], bpos[1] - pp[1]) if pp is not None else 1e9
            closest = min(players, key=dist_cm)
            return closest, dist_cm(closest)

    def dist_px(p: TrackObservation) -> float:
        ball_x = (ball.bbox.x1 + ball.bbox.x2) / 2 * video.width
        ball_y = ball.bbox.y2 * video.height
        p_x    = (p.bbox.x1 + p.bbox.x2) / 2 * video.width
        p_y    = p.bbox.y2 * video.height
        # Aerial contacts: ball is at head/chest height so ball_y is above player
        # feet by ~player_height.  Only horizontal proximity matters for a header
        # or volley.  Ground contacts: full 2D distance (feet-to-ball-bottom).
        raw = abs(p_x - ball_x) if aerial else math.hypot(p_x - ball_x, p_y - ball_y)
        p_h = (p.bbox.y2 - p.bbox.y1) * video.height
        if p_h >= 15.0:
            return raw * (_STANDARD_PLAYER_HEIGHT_PX / p_h)
        return raw

    closest = min(players, key=dist_px)
    return closest, dist_px(closest)


def _is_ball_airborne(
    ball: TrackObservation,
    players: list[TrackObservation],
    video: VideoMeta,
    airborne_frac: float = 0.20,
) -> bool:
    """True if ball bottom-edge is significantly above nearby player feet.

    Uses the player bbox height as a perspective-normalised ruler:
    if ball.y2 is more than airborne_frac × avg_player_height above avg player
    feet (y2), the ball is treated as airborne and excluded from possession
    and reception logic.

    At standard broadcast zoom: 20% of 70px ≈ 14px ≈ ~35 cm off the ground.
    """
    ball_cx_px = (ball.bbox.x1 + ball.bbox.x2) / 2 * video.width
    ball_y2_px = ball.bbox.y2 * video.height

    nearby: list[tuple[float, float, float]] = []
    for p in players:
        p_cx_px = (p.bbox.x1 + p.bbox.x2) / 2 * video.width
        p_h_px  = (p.bbox.y2 - p.bbox.y1) * video.height
        if p_h_px >= 15.0 and abs(p_cx_px - ball_cx_px) < p_h_px * 3:
            feet = p.bbox.y2 * video.height
            distance = math.hypot(p_cx_px - ball_cx_px, feet - ball_y2_px) / p_h_px
            nearby.append((distance, feet, p_h_px))

    if not nearby:
        return False

    # Averaging players at different depths marks ground balls as airborne.
    _, avg_feet_y, avg_player_h = min(nearby)

    # In image coords: smaller y = higher in frame = higher physically.
    # Airborne if ball bottom is above avg feet by > airborne_frac * player height.
    return ball_y2_px < avg_feet_y - airborne_frac * avg_player_h


def _ball_height_ratio(
    ball: TrackObservation,
    players: list[TrackObservation],
    video: VideoMeta,
) -> float:
    """Ball centre height as a fraction of the nearest player's bbox height.

    0.0  = ball centre at player feet (ground level)
    0.40 = knee height
    0.55 = waist height
    0.75 = chest / shoulder height
    1.00 = top of player head
    >1.0 = ball above head (truly in flight, unreachable)

    Uses the same nearby-player pool as _is_ball_airborne (players within
    3× player-height horizontal distance).  Returns 0.0 if no players nearby.
    """
    ball_cx_px = (ball.bbox.x1 + ball.bbox.x2) / 2 * video.width
    ball_cy_px = (ball.bbox.y1 + ball.bbox.y2) / 2 * video.height

    nearby: list[tuple[float, float, float]] = []
    for p in players:
        p_cx_px = (p.bbox.x1 + p.bbox.x2) / 2 * video.width
        p_h_px  = (p.bbox.y2 - p.bbox.y1) * video.height
        if p_h_px >= 15.0 and abs(p_cx_px - ball_cx_px) < p_h_px * 3:
            feet = p.bbox.y2 * video.height
            distance = math.hypot(p_cx_px - ball_cx_px, feet - ball_cy_px) / p_h_px
            nearby.append((distance, feet, p_h_px))

    if not nearby:
        return 0.0

    _, avg_feet_y, avg_player_h = min(nearby)
    if avg_player_h < 1.0:
        return 0.0

    # Image coords: smaller y = higher in frame = physically higher.
    return (avg_feet_y - ball_cy_px) / avg_player_h


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
    """Returns smoothed velocity per frame: cm/s (pitch mode) or px/frame (pixel fallback).

    Raw frame-to-frame diffs are computed first, then a 5-frame rolling median
    is applied to each velocity component independently. This removes single-frame
    detector jitter (which looks like a spurious speed spike) without smearing
    the true kick impulse across many frames.
    """
    raw_vels: dict[int, tuple[float, float]] = {}
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
             ball.bbox.y2 * video.height)  # use bottom edge (ground contact)
        )

        if curr is not None and prev_pos is not None and prev_f is not None:
            gap_frames = max(f - prev_f, 1)
            if use_pitch:
                gap_s = gap_frames / video.fps
                vx = (curr[0] - prev_pos[0]) / gap_s
                vy = (curr[1] - prev_pos[1]) / gap_s
                if math.hypot(vx, vy) <= _MAX_BALL_SPEED_CMS:
                    raw_vels[f] = (vx, vy)
            else:
                vx = (curr[0] - prev_pos[0]) / gap_frames
                vy = (curr[1] - prev_pos[1]) / gap_frames
                if math.hypot(vx, vy) <= _MAX_BALL_SPEED_PX_FRAME:
                    raw_vels[f] = (vx, vy)

        if curr is not None:
            prev_pos, prev_f = curr, f

    # ── 5-frame rolling median smoothing ──────────────────────────────────
    # Eliminates single-frame detector jumps that otherwise look like kick
    # speed spikes. Uses a centred window; no phase lag on the kick frame.
    frames_with_vel = sorted(raw_vels.keys())
    if len(frames_with_vel) < 3:
        return raw_vels

    half = _VEL_SMOOTH_WINDOW // 2
    smoothed: dict[int, tuple[float, float]] = {}
    for i, f in enumerate(frames_with_vel):
        lo = max(0, i - half)
        hi = min(len(frames_with_vel), i + half + 1)
        window_vx = sorted(raw_vels[frames_with_vel[j]][0] for j in range(lo, hi))
        window_vy = sorted(raw_vels[frames_with_vel[j]][1] for j in range(lo, hi))
        mid = len(window_vx) // 2
        smoothed[f] = (window_vx[mid], window_vy[mid])
    return smoothed


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


def _predict_landing_frame(
    flight_ball_y: list[float],
    flight_frame: int,
    fps: float,
) -> int | None:
    """Estimate the frame at which a kicked ball will land using parabolic symmetry.

    Ball y in pixel coords is smaller when the ball is higher in the frame.
    The arc peak = the minimum y value. Assuming a symmetric parabola:
        landing_frame ≈ flight_frame + 2 * frames_to_peak
    Returns None if there is insufficient data to make a reliable estimate.
    """
    min_pts = max(4, int(fps * 0.5))
    if len(flight_ball_y) < min_pts:
        return None
    min_y    = min(flight_ball_y)
    peak_idx = flight_ball_y.index(min_y)
    if peak_idx < 3:
        return None
    post_peak = flight_ball_y[peak_idx:]
    if len(post_peak) < 3:
        return None
    # Only trust the prediction if the ball has clearly started descending
    if post_peak[-1] <= min_y + 20:
        return None
    return flight_frame + 2 * peak_idx


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
