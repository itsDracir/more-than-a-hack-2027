"""Possession decoder: per-frame ball evidence + a Viterbi HMM over H / A / D.

The baseline votes frame by frame inside a window. Here the possession *structure* is explicit:
a hidden state (H, A or D) that is sticky, so a possession is one segment, not a flickering run.

  emissions   visible ball -> logistic on (distance to nearest away player - distance to nearest
              home player); ball far from everyone or outside the pitch -> evidence for D.
              invisible ball -> no evidence, so the state is carried over the gap.
  transitions switching H<->A or entering D costs a fixed penalty; this is what removes the
              flicker and sets the typical possession length.

    python solution/possession.py                      # writes evaluation/results/hmm_v1.csv
    python solution/possession.py --switch 9 --k 0.6   # tune the two main knobs
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
import baseline  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data", "processed")
H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])


def extract(game=GAME, data_dir=None):
    """Stream the ATD once. Returns per-frame arrays: ball xy (m), nearest distance per team (m)."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    team = baseline.team_of_each_track(meta_path)

    ball_x = np.full(n, np.nan)
    ball_y = np.full(n, np.nan)
    dist = np.full((n, 2), np.inf)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3 or parts[2][0] == "N":
                continue
            frame = int(parts[0]) - 1
            bx, by = parts[2].split(",")[:2]
            bx, by = float(bx) * length, float(by) * width
            ball_x[frame], ball_y[frame] = bx, by
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                px, py = group.split(",")[:2]
                d = np.hypot(float(px) * length - bx, float(py) * width - by)
                if d < dist[frame, team[slot]]:
                    dist[frame, team[slot]] = d
    return ball_x, ball_y, dist, (length, width)


def load_features(refresh=False):
    """Marc's per-frame table (solution/preprocessing.ipynb) if present, else read the ATD directly.

    Returns ball x/y (m), nearest distance per team (m), pitch and a per-frame set-piece flag.
    """
    table = os.path.join(CACHE_DIR, "features_tactical.csv")
    if os.path.isfile(table) and not refresh:
        import pandas as pd
        tags = ["tag_set_pieces_H", "tag_set_pieces_A", "tag_set_pieces_unknown"]
        df = pd.read_csv(table, usecols=["ball_x_m", "ball_y_m", "home_nearest_m", "away_nearest_m"] + tags)
        dist = np.column_stack([df["home_nearest_m"].fillna(np.inf), df["away_nearest_m"].fillna(np.inf)])
        setp = (df[tags].sum(axis=1) > 0).to_numpy()
        return df["ball_x_m"].to_numpy(), df["ball_y_m"].to_numpy(), dist, GAMES[GAME]["pitch"], setp

    path = os.path.join(CACHE_DIR, "possession_features.npz")
    if not refresh and os.path.isfile(path):
        z = np.load(path)
        bx, by, dist, pitch = z["bx"], z["by"], z["dist"], tuple(z["pitch"])
    else:
        bx, by, dist, pitch = extract()
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(path, bx=bx, by=by, dist=dist, pitch=np.array(pitch))
    return bx, by, dist, pitch, np.zeros(len(bx), dtype=bool)


def emissions(bx, by, dist, pitch, setp, k=0.15, far=6.0, d_bias=-2.5, clip=10.0, sp=0.0):
    """(n, 3) log-likelihood-ish scores for H, A, D. Zeros where the ball is invisible."""
    length, width = pitch
    n = len(bx)
    em = np.zeros((n, 3))
    visible = ~np.isnan(bx)

    dh = np.minimum(dist[:, H], 40.0)
    da = np.minimum(dist[:, A], 40.0)
    margin = np.where(visible, np.clip(da - dh, -clip, clip), 0.0)  # >0: home player is closer
    em[:, H] = np.where(visible, -np.logaddexp(0, -k * margin), 0.0)
    em[:, A] = np.where(visible, -np.logaddexp(0, k * margin), 0.0)

    outside = visible & ((bx < 0) | (bx > length) | (by < 0) | (by > width))
    isolated = visible & (np.minimum(dist[:, H], dist[:, A]) > far)
    em[:, D] = np.where(outside, 0.0, np.where(isolated, -0.5, np.where(visible, d_bias, d_bias)))
    em[outside, H] = em[outside, A] = -3.0
    em[~visible, D] = -1.5  # a missing ball is mild evidence for a stoppage, not proof
    em[setp, D] += sp  # Smart Tagging set-piece interval: restarts are mostly dead ball
    return em


def viterbi(em, switch=9.0, to_dead=8.0):
    """Most likely state path. Staying is free; H<->A costs `switch`, entering/leaving D `to_dead`."""
    n = len(em)
    cost = np.full((3, 3), switch)
    cost[D, :] = to_dead
    cost[:, D] = to_dead
    np.fill_diagonal(cost, 0.0)

    score = em[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] - cost  # cand[i, j]: come from i, go to j
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + em[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(f, lab) for f, lab in enumerate(labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=float, default=0.15)
    parser.add_argument("--switch", type=float, default=25.0)
    parser.add_argument("--dead", type=float, default=8.0)
    parser.add_argument("--sp", type=float, default=0.5, help="D bonus inside set-piece tags")
    parser.add_argument("--refresh", action="store_true", help="re-read the raw tracking")
    parser.add_argument("--name", default="hmm_v1")
    args = parser.parse_args()

    bx, by, dist, pitch, setp = load_features(args.refresh)
    path = viterbi(emissions(bx, by, dist, pitch, setp, k=args.k, sp=args.sp), args.switch, args.dead)
    labels = LABELS[path]
    out = os.path.join(ROOT, "evaluation", "results", args.name + ".csv")
    write_submission(out, labels)
    print({lab: int(np.sum(labels == lab)) for lab in LABELS}, "->", os.path.relpath(out, ROOT))
