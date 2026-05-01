from __future__ import annotations

import cv2
import numpy as np

# Standard pitch dimensions
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Minimap display size in pixels
MAP_W = 320
MAP_H = int(MAP_W * PITCH_WIDTH_M / PITCH_LENGTH_M)  # ~207px — maintains aspect ratio

# Colours (BGR)
PITCH_GREEN = (34, 139, 34)
LINE_WHITE = (255, 255, 255)
TEAM_COLORS: dict[int | None, tuple[int, int, int]] = {
    0: (60, 80, 220),    # Team A — red-ish
    1: (220, 180, 60),   # Team B — blue-ish
    2: (160, 160, 160),  # Other
    None: (160, 160, 160),
}


def _m_to_px(x_m: float, y_m: float) -> tuple[int, int]:
    px = int(x_m / PITCH_LENGTH_M * MAP_W)
    py = int(y_m / PITCH_WIDTH_M * MAP_H)
    return px, py


def _draw_pitch_lines(canvas: np.ndarray) -> None:
    T = 1  # line thickness

    # Outer boundary
    cv2.rectangle(canvas, (0, 0), (MAP_W - 1, MAP_H - 1), LINE_WHITE, T)

    # Halfway line
    mid_x = MAP_W // 2
    cv2.line(canvas, (mid_x, 0), (mid_x, MAP_H), LINE_WHITE, T)

    # Centre circle (~9.15m radius)
    r_px = int(9.15 / PITCH_LENGTH_M * MAP_W)
    cx, cy = MAP_W // 2, MAP_H // 2
    cv2.circle(canvas, (cx, cy), r_px, LINE_WHITE, T)
    cv2.circle(canvas, (cx, cy), 2, LINE_WHITE, -1)

    # Penalty areas (16.5m deep, 40.32m wide)
    pa_depth = int(16.5 / PITCH_LENGTH_M * MAP_W)
    pa_top = int((PITCH_WIDTH_M / 2 - 20.16) / PITCH_WIDTH_M * MAP_H)
    pa_bot = int((PITCH_WIDTH_M / 2 + 20.16) / PITCH_WIDTH_M * MAP_H)
    # Left penalty area
    cv2.rectangle(canvas, (0, pa_top), (pa_depth, pa_bot), LINE_WHITE, T)
    # Right penalty area
    cv2.rectangle(canvas, (MAP_W - pa_depth, pa_top), (MAP_W, pa_bot), LINE_WHITE, T)

    # Penalty spots
    ps_x_left = int(11.0 / PITCH_LENGTH_M * MAP_W)
    ps_x_right = int(94.0 / PITCH_LENGTH_M * MAP_W)
    ps_y = MAP_H // 2
    cv2.circle(canvas, (ps_x_left, ps_y), 2, LINE_WHITE, -1)
    cv2.circle(canvas, (ps_x_right, ps_y), 2, LINE_WHITE, -1)


def draw_minimap(
    player_positions_m: list[tuple[float, float]],
    team_ids: list[int | None],
) -> np.ndarray:
    """
    Draw a bird's-eye pitch minimap with coloured player dots.
    Returns a (MAP_H, MAP_W, 3) BGR image.
    """
    canvas = np.full((MAP_H, MAP_W, 3), PITCH_GREEN, dtype=np.uint8)
    _draw_pitch_lines(canvas)

    for (x_m, y_m), team_id in zip(player_positions_m, team_ids):
        if x_m < 0 or x_m > PITCH_LENGTH_M or y_m < 0 or y_m > PITCH_WIDTH_M:
            continue
        px, py = _m_to_px(x_m, y_m)
        color = TEAM_COLORS.get(team_id, TEAM_COLORS[None])
        cv2.circle(canvas, (px, py), 5, color, -1)
        cv2.circle(canvas, (px, py), 5, LINE_WHITE, 1)

    return canvas


def overlay_minimap(
    frame: np.ndarray,
    minimap: np.ndarray,
    alpha: float = 0.85,
    margin: int = 10,
) -> np.ndarray:
    """Composite minimap onto bottom-centre of the frame."""
    fh, fw = frame.shape[:2]
    mh, mw = minimap.shape[:2]

    x = (fw - mw) // 2
    y = fh - mh - margin

    if x < 0 or y < 0:
        return frame

    roi = frame[y:y + mh, x:x + mw]
    blended = cv2.addWeighted(minimap, alpha, roi, 1 - alpha, 0)
    frame[y:y + mh, x:x + mw] = blended
    # Thin white border
    cv2.rectangle(frame, (x, y), (x + mw, y + mh), (255, 255, 255), 1)
    return frame
