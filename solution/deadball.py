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
    """Detect motion by *distance travelled*, not just net displacement.

    The old implementation missed a ball that travelled and returned to its
    starting point. Here we sum valid consecutive steps over the window.
    Small steps (below 4 cm) are ignored to reduce tracker jitter; jumps
    suggesting over 45 m/s are discarded as potential tracking glitches.

    We require at least 70% valid consecutive steps and the ball visible
    inside the pitch at the current frame. Missing tracking alone is NEVER
    taken as evidence of a stoppage.

    These thresholds are physical heuristics, not fitted to ground truth.
    """
    length, width = pitch
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x i y han de ser vectors de la mateixa longitud")
    if fps <= 0:
        raise ValueError("fps ha de ser positiu")

    n = len(x)
    lag = max(1, int(round(window_s * fps)))
    moving = np.zeros(n, dtype=bool)
    if n <= lag:
        return moving

    with np.errstate(invalid="ignore"):
        inside = (np.isfinite(x) & np.isfinite(y) &
                  (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1))
        steps = np.hypot(np.diff(x) * length, np.diff(y) * width)
        valid = (inside[:-1] & inside[1:] & np.isfinite(steps) &
                 (steps * fps <= 45.0))

    # La distancia d'un pas només compta si és plausible i prou gran
    # per no confondre oscil·lacions minúscules amb moviment real.
    travelled_steps = np.where(valid & (steps >= 0.04), steps, 0.0)
    distance_sum = np.r_[0.0, np.cumsum(travelled_steps)]
    valid_sum = np.r_[0, np.cumsum(valid.astype(np.int64))]

    travelled_window = distance_sum[lag:] - distance_sum[:-lag]
    valid_fraction = (valid_sum[lag:] - valid_sum[:-lag]) / lag
    moving[lag:] = (inside[lag:] & (valid_fraction >= 0.70) &
                    (travelled_window > min_m))
    return moving


def detect_restart(ball_speed, start, end, kick_speed=3.0, n_frames=3):
    """Find a credible restart within [start, end), or return None.

    A restart requires a change from a recently *observed stationary ball*
    to sustained fast motion. Merely finding three fast frames is not enough:
    the ball could have been moving before the Smart Tagging interval began.

    Require three consecutive low-speed measurements (<= 1 m/s, or a third
    of kick_speed if lower) in a short look-back, with the last measured speed
    before the kick also low. This is deliberately conservative: if the ball
    is missing around the restart, return None rather than invent the kick.
    """
    values = np.asarray(ball_speed, dtype=float)
    start, end = max(0, int(start)), min(len(values), int(end))
    n_frames = int(n_frames)
    if n_frames < 1:
        raise ValueError("n_frames ha de ser com a mínim 1")
    if end - start < n_frames:
        return None

    finite = np.isfinite(values)
    fast = finite & (values > kick_speed)
    calm = finite & (values <= min(1.0, kick_speed / 3.0))
    min_calm = max(3, n_frames)
    lookback = max(10, 4 * min_calm)

    for frame in range(start, end - n_frames + 1):
        if not np.all(fast[frame:frame + n_frames]):
            continue
        # No és un inici de moviment si ja anava ràpid abans del candidat.
        if frame > 0 and fast[frame - 1]:
            continue
        prior_start = max(0, frame - lookback)
        recent_calm = calm[prior_start:frame]
        if len(recent_calm) < min_calm:
            continue
        # Una ratxa lenta abans de la puntada, no un únic frame sorollós.
        stable = np.convolve(recent_calm.astype(int),
                             np.ones(min_calm, dtype=int), mode="valid")
        if not np.any(stable == min_calm):
            continue
        # La darrera velocitat observada abans de la puntada ha de ser baixa.
        previous_valid = np.flatnonzero(finite[prior_start:frame])
        if not len(previous_valid) or not recent_calm[previous_valid[-1]]:
            continue
        return frame
    return None


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


def dead_mask(collective_speed, x, y, blocks, events, pitch, fps, params=None, ball_speed=None):
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
    # Reutilitzem la velocitat calculada al preprocessament si està disponible.
    if ball_speed is None:
        ball_speed = ball_speed_mps(x, y, pitch, fps)
    else:
        ball_speed = np.asarray(ball_speed, dtype=float)
        if len(ball_speed) != n:
            raise ValueError("ball_speed ha de tenir una dada per frame")
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
