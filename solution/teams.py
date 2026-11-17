"""Which team has the ball, inside stretches already known to be in play.

Input is the per-frame distance from the ball to the nearest home and nearest away player -- what
the preprocessing produces (home_nearest_m / away_nearest_m, NaN where the ball is not seen).

Two steps:

1. **Viterbi** over {home, away} with a cost per change of possession. A deflection or a lost
   duel gives the other team a few frames of evidence, not enough to pay for two switches, so it
   does not split the possession. A real interception gives sustained evidence and does.
2. **Move each change to the new team's first touch.** Between the old team's last touch and
   the new team's first one, the ball is loose or in the air and the evidence is indifferent, so
   Viterbi may put the change anywhere in that gap. The ground truth gives the ball to the new
   team when it gains control -- loose balls stay with the last team (CHALLENGE.md) -- so the
   change goes at the first touch. One second late costs a third of spellF1 (EVALUATION.md).

Frames where the ball is not tracked carry no evidence and simply continue the current
possession, which is what an occlusion inside a possession should do.
"""

import numpy as np

from solution.params import Params
from solution.viterbi import viterbi

HOME, AWAY, NONE = 0, 1, -1


def team_evidence(home_nearest_m, away_nearest_m, params=None):
    """Per-frame evidence that the home team (positive) or away team (negative) is on the ball.

    Strong only when someone is actually close to the ball: weighted by exp(-closest / r).
    A team with no tracked player near the ball is treated as infinitely far.
    """
    p = params or Params()
    home = np.where(np.isfinite(home_nearest_m), home_nearest_m, np.inf)
    away = np.where(np.isfinite(away_nearest_m), away_nearest_m, np.inf)
    with np.errstate(invalid="ignore"):
        closest = np.minimum(home, away)
        margin = np.clip(away - home, -10.0, 10.0)          # inf - inf -> NaN -> no evidence
        evidence = np.exp(-closest / p.control_radius_m) * np.tanh(margin / p.margin_scale_m)
    evidence[~np.isfinite(evidence)] = 0.0
    return evidence + 0.0


def tactical_team_evidence(features, params=None):
    """Small, bounded tactical reinforcement; never possession by itself.

    Use offensive tags only when *independent tracking evidence* already
    supports the same team with sufficient strength. In particular:

    - if the ball is missing, tactical evidence is zero;
    - if tracking supports the opposite team, tactical evidence is zero;
    - for each continuous tag interval, the total bonus is capped at a
      quarter of the cost of a possession change.

    Consequently, a long tactical tag cannot manufacture a possession
    switch in a stretch without visible ball/player proximity information.
    No parameters are fitted using the public ground-truth fragment.
    """
    p = params or Params()
    n = len(features)
    codes = ("build_up", "progression", "attacking_transition", "creating_chances")
    home = np.zeros(n, dtype=bool)
    away = np.zeros(n, dtype=bool)
    for code in codes:
        for team, accum in (("H", home), ("A", away)):
            name = f"tag_{code}_{team}"
            if name in features.columns:
                accum |= features[name].fillna(0).to_numpy(dtype=float) > 0

    # Si els dos equips tenen tags al mateix frame, no prenem partit.
    direction = (home & ~away).astype(np.int8) - (away & ~home).astype(np.int8)
    result = np.zeros(n, dtype=float)
    if p.tactical_team_weight <= 0 or not np.any(direction):
        return result

    observed = team_evidence(features["home_nearest_m"].to_numpy(dtype=float),
                             features["away_nearest_m"].to_numpy(dtype=float), p)
    # Una etiqueta només reforça una proximitat significativa del mateix equip.
    confirmed = ((direction != 0) & (observed * direction >= p.touch_evidence))
    result[confirmed] = (direction[confirmed] *
                         np.minimum(float(p.tactical_team_weight),
                                    0.25 * np.abs(observed[confirmed])))

    # Una etiqueta que dura minuts no es pot acumular indefinidament.
    # El pressupost és per cada interval ininterromput d'un mateix signe.
    budget = max(0.0, 0.25 * float(p.team_switch_cost))
    for side in (-1, 1):
        for first, stop in runs_of(direction == side):
            total = float(np.abs(result[first:stop]).sum())
            if total > budget and total > 0:
                result[first:stop] *= budget / total

    return result


def runs_of(mask):
    """[(start, stop)] half-open runs where `mask` is True."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _touches(evidence, team, threshold):
    return evidence > threshold if team == HOME else evidence < -threshold


def move_changes_to_first_touch(path, evidence, threshold):
    """Shift each change of team to the new team's first touch after the old team's last one."""
    out = path.copy()
    changes = (np.flatnonzero(path[1:] != path[:-1]) + 1).tolist()
    bounds = [0] + changes + [len(path)]
    for i, change in enumerate(changes):
        old, new = int(path[change - 1]), int(path[change])
        previous_change, next_change = bounds[i], bounds[i + 2]
        old_touches = np.flatnonzero(_touches(evidence[previous_change:change], old, threshold))
        last_old = previous_change + int(old_touches[-1]) if len(old_touches) else previous_change
        new_touches = np.flatnonzero(_touches(evidence[last_old + 1:next_change], new, threshold))
        if not len(new_touches):
            continue
        first_new = last_old + 1 + int(new_touches[0])
        if first_new > change:
            out[change:first_new] = old
        else:
            out[first_new:change] = new
    return out


def segment_teams(evidence, alive, params=None):
    """Team code per frame: HOME / AWAY inside alive runs, NONE on dead frames and on alive runs
    that never see the ball (the scorer forward-fills those; guessing would only add noise)."""
    p = params or Params()
    evidence = np.asarray(evidence, dtype=float)
    codes = np.full(len(evidence), NONE, dtype=np.int8)
    for start, stop in runs_of(alive):
        run = evidence[start:stop]
        if not np.any(run != 0):
            continue
        path = viterbi(np.column_stack([run / 2.0, -run / 2.0]), p.team_switch_cost)
        codes[start:stop] = move_changes_to_first_touch(path, run, p.touch_evidence)
    return codes
