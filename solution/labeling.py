"""Per-frame features in, H / A / D / X labels out. The decision layer of the pipeline.

Contract with the feature builder (the preprocessing notebook, or any possession.py that
replaces it): a DataFrame with one row per video frame 0..N-1, in order, with these columns.
Names match solution/01_preprocessing.ipynb, so its CSV plugs in unchanged.

    ball_x_norm, ball_y_norm      ball position, 0-1 pitch coordinates, NaN when not tracked
    home_nearest_m                distance ball -> nearest home player (m), NaN when unknown
    away_nearest_m                distance ball -> nearest away player (m), NaN when unknown
    home_mean_speed_mps           mean speed of tracked home players (m/s)
    away_mean_speed_mps           mean speed of tracked away players (m/s)

Order of decisions:
  1. outside the graded halves             -> X (never scored)
  2. ball out of play (deadball.py)        -> D
  3. in play: which team (teams.py)        -> H / A
  4. in play but the ball is never seen    -> X (the scorer carries the previous label)
"""

import os
import sys

import numpy as np

from solution.deadball import dead_mask
from solution.params import Params
from solution.teams import AWAY, HOME, segment_teams, team_evidence, tactical_team_evidence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import games  # noqa: E402

REQUIRED_COLUMNS = ("ball_x_norm", "ball_y_norm", "home_nearest_m", "away_nearest_m",
                    "home_mean_speed_mps", "away_mean_speed_mps")


def check_columns(features):
    missing = [c for c in REQUIRED_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError("features are missing columns {} (contract in solution/labeling.py)"
                         .format(missing))


def collective_speed(features):
    """Mean of the two team means, ignoring a team with no tracked player."""
    speeds = features[["home_mean_speed_mps", "away_mean_speed_mps"]].to_numpy(dtype=float)
    counts = np.isfinite(speeds).sum(axis=1)
    total = np.nansum(speeds, axis=1)
    return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)


def graded_mask(n, blocks):
    mask = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        mask[first:last + 1] = True
    return mask


def label_frames(features, events=None, blocks=None, pitch=None, fps=None, params=None, use_tactical_tags=True):
    """Array of 'H' / 'A' / 'D' / 'X', one per row of `features`."""
    check_columns(features)
    p = params or Params()
    blocks = blocks if blocks is not None else games.half_video_blocks(games.GAME)
    pitch = pitch or games.GAMES[games.GAME]["pitch"]
    fps = fps or games.GAMES[games.GAME]["fps"]
    n = len(features)

    x = features["ball_x_norm"].to_numpy(dtype=float)
    y = features["ball_y_norm"].to_numpy(dtype=float)
    # No recalculem la velocitat de la pilota si ja la tenim a les features.
    speed_col = (features["ball_apparent_speed_mps"].to_numpy(dtype=float)
                 if "ball_apparent_speed_mps" in features.columns else None)
    dead = dead_mask(collective_speed(features), x, y, blocks, events, pitch, fps, p,
                     ball_speed=speed_col)

    graded = graded_mask(n, blocks)
    evidence = team_evidence(features["home_nearest_m"].to_numpy(dtype=float),
                             features["away_nearest_m"].to_numpy(dtype=float), p)
    # Utilitzem els tags per frame que va calcular el notebook, sense rellegir l'XML.
    if use_tactical_tags:
        evidence += tactical_team_evidence(features, p)
    teams = segment_teams(evidence, graded & ~dead, p)

    labels = np.full(n, "X", dtype="<U1")
    labels[graded & dead] = "D"
    labels[teams == HOME] = "H"
    labels[teams == AWAY] = "A"
    return labels
