"""Is the ball in play? Decided from evidence of a stoppage -- never from "the tracker lost the ball".

The baseline says D whenever the ball is not tracked. That conflates two unrelated things: the
ball is missing on 41% of graded frames, mostly *during* play (duels, crowded boxes, balls in the
air). Every spurious D splits a real possession in two, and spellF1 then loses both halves.

Evidence used here, all automatic:

- **ball moving inside the pitch**: travelling more than 2 m in a second, in bounds. The
  strongest sign of play, and the one that separates a patient build-up from a stoppage.
- **collective speed**: when the referee stops play, players stop running -- but slow possession
  play is just as slow, so this is weak evidence, relative to the half's median pace.
- **ball beyond a line**: if the tracker sees it out, it is out.
- **Smart Tagging SET PIECES**: a restart happens inside each window; the kick is located on the
  ball track, frames before it lean dead and the second after it leans alive.
- **Kick Off (Start)**: nothing before the kick-off of a half is play.

A two-state Viterbi with a cost for entering/leaving a stoppage turns this into clean intervals.
"""

import numpy as np
import pandas as pd

from solution.params import Params
from solution.viterbi import viterbi

FORCED = 1e3  # evidence large enough that no switch cost can overrule it


def speed_threshold(values):
    """Reference speed for "slower than usual": the median of the half, NaN ignored.

    Not a split between two populations -- on the real match the smoothed collective speed is a
    single skewed hump (diagnose.py), so any two-class cut such as Otsu lands in the bulk of play
    and calls most of the match dead. The median only says "slower than this half's typical
    pace", which is why speed is weak evidence and the ball's own motion decides.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def ball_speed_mps(x, y, pitch, fps):
    """Apparent ball speed between consecutive frames, m/s. NaN if either frame lacks the ball."""
    length, width = pitch
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    speed = np.full(len(x), np.nan)
    if len(x) > 1:
        speed[1:] = np.hypot(np.diff(x) * length, np.diff(y) * width) * fps
    return speed


def ball_moving(x, y, pitch, fps, window_s=1.0, min_m=2.0):
    """True where the ball, seen inside the pitch now and `window_s` ago, travelled > min_m."""
    length, width = pitch
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lag = max(1, int(round(window_s * fps)))
    moving = np.zeros(len(x), dtype=bool)
    if len(x) <= lag:
        return moving
    with np.errstate(invalid="ignore"):
        inside = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
        distance = np.hypot((x[lag:] - x[:-lag]) * length, (y[lag:] - y[:-lag]) * width)
        moving[lag:] = inside[lag:] & inside[:-lag] & (distance > min_m)
    return moving


def detect_restart(ball_speed, start, end, kick_speed=3.0, n_frames=3):
    """First frame in [start, end) where the ball moves faster than `kick_speed` for `n_frames`
    frames running -- the kick that restarts play. None if the window has no such frame."""
    start, end = max(0, int(start)), min(len(ball_speed), int(end))
    if end - start < n_frames:
        return None
    with np.errstate(invalid="ignore"):
        fast = (np.nan_to_num(ball_speed[start:end], nan=0.0) > kick_speed).astype(int)
    running = np.convolve(fast, np.ones(n_frames, dtype=int), mode="valid")
    hits = np.flatnonzero(running == n_frames)
    return start + int(hits[0]) if len(hits) else None


def dead_evidence(speed_smooth, threshold, x, y, pitch, params=None, moving=None):
    """Per-frame log-odds that the ball is dead, before any set-piece information.

    `moving` (bool per frame, from ball_moving) marks the ball travelling inside the pitch, the
    strongest sign of play there is: it outweighs slow players, so a patient build-up is not
    mistaken for a stoppage.
    """
    p = params or Params()
    length, width = pitch
    scale = max(p.speed_scale_frac * threshold, 1e-6) if np.isfinite(threshold) else np.inf
    with np.errstate(invalid="ignore"):
        evidence = p.w_speed * np.tanh(np.nan_to_num((threshold - speed_smooth) / scale, nan=0.0))
        mx, my = p.oob_margin_m / length, p.oob_margin_m / width
        out = (x < -mx) | (x > 1 + mx) | (y < -my) | (y > 1 + my)
    evidence = evidence + p.w_oob * out
    if moving is not None:
        evidence = evidence - p.w_move * np.asarray(moving, dtype=bool)
    return evidence + 0.0  # no negative zeros


def smooth_speed(collective_speed, fps, params=None):
    """Centred rolling median of the collective speed over params.speed_smooth_s seconds."""
    p = params or Params()
    window = max(1, int(round(p.speed_smooth_s * fps)) | 1)
    return (pd.Series(np.asarray(collective_speed, dtype=float))
            .rolling(window, center=True, min_periods=1).median().to_numpy())


def _is_set_piece(event):
    return (event.get("code") or "").strip().upper() == "SET PIECES"


def _is_kickoff_start(event):
    return "kick off (start)" in (event.get("tags", {}).get("Type") or "").strip().lower()


def dead_mask(collective_speed, x, y, blocks, events, pitch, fps, params=None):
    """Boolean array: True where the ball is judged out of play. Only graded blocks are decided.

    `blocks` is [(half, first_frame, last_frame)] with last inclusive, as
    games.half_video_blocks() returns it. `events` is the Smart Tagging list from
    starter/load_smart_tagging.load(), or None to run without it.
    """
    p = params or Params()
    n = len(collective_speed)
    smooth = smooth_speed(collective_speed, fps, p)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ball_speed = ball_speed_mps(x, y, pitch, fps)
    moving = ball_moving(x, y, pitch, fps, p.move_window_s, p.move_min_m)

    dead = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        stop = last + 1
        threshold = speed_threshold(smooth[first:stop])
        evidence = dead_evidence(smooth[first:stop], threshold, x[first:stop], y[first:stop],
                                 pitch, p, moving[first:stop]) + p.dead_prior

        for event in events or []:
            if not _is_set_piece(event):
                continue
            begin, end = int(event["start_frame"]), int(event["end_frame"])
            if end <= first or begin > last:
                continue
            restart = detect_restart(ball_speed, max(begin, first), min(end, stop),
                                     p.kick_speed_mps, p.kick_frames)
            if _is_kickoff_start(event):
                # Before the kick (or, if it was not found, before the window): no play yet.
                # Clamped: the first half is graded from the kick-off itself, so its window
                # starts before the block and a negative slice end would cover the whole half.
                evidence[:max(0, (restart if restart is not None else begin) - first)] = FORCED
            if restart is None:
                continue
            evidence[max(begin, first) - first:restart - first] += p.w_setpiece
            evidence[restart - first:min(restart + fps, stop) - first] -= p.w_setpiece

        emissions = np.column_stack([np.zeros(stop - first), evidence])
        dead[first:stop] = viterbi(emissions, p.dead_switch_cost) == 1
    return dead
