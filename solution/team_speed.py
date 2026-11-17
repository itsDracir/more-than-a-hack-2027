"""Mean player speed of each team per frame -- the main dead-ball signal.

This is the one thing the decision layer needs from the raw tracking that the per-frame feature
table (ball position, nearest player per team) does not already carry. If your feature builder
adds `home_mean_speed_mps` and `away_mean_speed_mps` itself, run.py uses those and never calls
this. Otherwise it costs one streaming pass over the ATD, cached under data/processed/.

Raw ATD line: "<frame_id>:<x,y,speed>;<x,y,speed>;...:<ball>", "NaN" for a missing slot,
frame_id counting from 1 (video frame = frame_id - 1).
"""

import numpy as np

# Faster than any sprint ever measured: a value above this is a tracking glitch, not a player.
MAX_SPEED_MPS = 12.0


def team_mean_speeds(lines, team_by_slot, n_frames, max_speed=MAX_SPEED_MPS):
    """(n_frames, 2) array of mean speed in m/s, column 0 home and 1 away. NaN where a team has
    no tracked player. `lines` is any iterable of raw ATD lines (an open file works)."""
    team = np.asarray(team_by_slot, dtype=np.int8)
    out = np.full((n_frames, 2), np.nan)
    for line in lines:
        parts = line.strip().split(":")
        if len(parts) != 3:
            continue
        frame = int(parts[0]) - 1
        if not 0 <= frame < n_frames:
            continue
        sums, counts = [0.0, 0.0], [0, 0]
        for slot, group in enumerate(parts[1].split(";")):
            fields = group.split(",")
            if len(fields) < 3:
                continue                                    # "NaN": slot not tracked
            speed = float(fields[2])
            if speed != speed:
                continue
            side = team[slot]
            sums[side] += min(max(speed, 0.0), max_speed)
            counts[side] += 1
        for side in (0, 1):
            if counts[side]:
                out[frame, side] = sums[side] / counts[side]
    return out
