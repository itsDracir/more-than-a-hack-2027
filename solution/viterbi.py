"""Best state sequence under per-frame evidence and a cost for every change of state.

This replaces the baseline's majority vote. A vote over +/-2 s smears every boundary by up to the
window and lets a three-frame deflection flip the label if it lands at the right moment. Viterbi
instead finds the single labelling that maximises

    sum_t emissions[t, state_t]  -  sum_t switch_cost[state_{t-1}, state_t]

so the switch cost is a direct, interpretable knob: "how much evidence does it take to believe
possession changed", in units of frames of clear evidence. That is what controls the number of
possessions, which is exactly what spellF1 grades.
"""

import numpy as np


def viterbi(emissions, switch_cost):
    """Most likely path. `emissions` is (T, K) log-evidence, higher = more likely.

    `switch_cost` is a scalar (same cost for any change) or a (K, K) matrix with
    cost[from, to]; the diagonal is ignored. Ties go to the lower state index, deterministically.
    """
    emissions = np.asarray(emissions, dtype=float)
    n_frames, n_states = emissions.shape
    if n_frames == 0:
        return np.zeros(0, dtype=np.int8)

    if np.isscalar(switch_cost):
        cost = np.full((n_states, n_states), float(switch_cost))
    else:
        cost = np.array(switch_cost, dtype=float)
    np.fill_diagonal(cost, 0.0)

    back = np.empty((n_frames, n_states), dtype=np.int8)
    score = emissions[0].copy()
    back[0] = np.arange(n_states)
    for t in range(1, n_frames):
        candidates = score[:, None] - cost          # [from, to]
        best_from = np.argmax(candidates, axis=0)
        back[t] = best_from
        score = candidates[best_from, np.arange(n_states)] + emissions[t]

    path = np.empty(n_frames, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
