"""Velocitat mitjana dels jugadors de cada equip per frame -- el senyal principal de pilota morta.

És l'única cosa que la capa de decisió necessita del tracking en brut i que la taula de features
per frame (posició de la pilota, jugador més proper de cada equip) encara no porta. Si el vostre
constructor de features ja afegeix `home_mean_speed_mps` i `away_mean_speed_mps`, run.py fa servir
aquestes i mai no crida aquest mòdul. Si no, costa una passada en streaming per l'ATD, que es desa
a data/processed/.

Línia de l'ATD en brut: "<frame_id>:<x,y,speed>;<x,y,speed>;...:<pilota>", "NaN" per a un slot
absent, frame_id comptant des d'1 (frame de vídeo = frame_id - 1).
"""

import numpy as np

# Més ràpid que qualsevol esprint mai mesurat: un valor per sobre és un error del tracking, no un
# jugador.
MAX_SPEED_MPS = 12.0


def team_mean_speeds(lines, team_by_slot, n_frames, max_speed=MAX_SPEED_MPS):
    """Array (n_frames, 2) de velocitat mitjana en m/s, columna 0 local i 1 visitant. NaN quan un
    equip no té cap jugador seguit. `lines` és qualsevol iterable de línies de l'ATD en brut (un
    fitxer obert serveix)."""
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
                continue                                    # "NaN": slot sense seguiment
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
