"""Què veu el detector de pilota morta al partit real, només amb agregats -- sense ground truth.

    python solution/diagnose.py

Imprimeix, per a cada part, els números que diuen si cada senyal de pilota morta es comporta bé:
la distribució de velocitats i la seva mediana, quina proporció de la part acaba morta i -- la
comprovació clau -- amb quina freqüència la pilota es *mou dins del camp* durant els frames
declarats morts. En una aturada real la pilota està quieta, fora o l'estan anant a buscar; si
sovint viatja durant frames "morts", el detector està confonent joc de possessió lent amb una
aturada.

Només s'imprimeixen recomptes, proporcions i quantils. Res a nivell de frame, així que la sortida
es pot enganxar a qualsevol lloc sense problema.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solution.deadball import (ball_moving, ball_speed_mps, dead_mask,  # noqa: E402
                               detect_restart, smooth_speed, speed_threshold,
                               _is_set_piece)
from solution.labeling import collective_speed  # noqa: E402
from solution.params import Params  # noqa: E402


def half_report(features, events, block, pitch, fps, params=None):
    """Diagnòstic agregat d'una part puntuada. Cada valor és un número o una tupla curta."""
    p = params or Params()
    _, first, last = block
    stop = last + 1
    speed = collective_speed(features)
    smooth = smooth_speed(speed, fps, p)[first:stop]
    x = features["ball_x_norm"].to_numpy(dtype=float)
    y = features["ball_y_norm"].to_numpy(dtype=float)

    dead = dead_mask(speed, x, y, [block], events, pitch, fps, p)[first:stop]
    moving = ball_moving(x, y, pitch, fps)[first:stop]
    xs, ys = x[first:stop], y[first:stop]
    visible = np.isfinite(xs) & np.isfinite(ys)
    with np.errstate(invalid="ignore"):
        out = visible & ((xs < 0) | (xs > 1) | (ys < 0) | (ys > 1))

    threshold = speed_threshold(smooth)
    finite = smooth[np.isfinite(smooth)]
    low, high = (np.percentile(finite, [1, 99]) if len(finite) else (0.0, 1.0))
    hist, edges = np.histogram(finite.clip(low, high), bins=12, range=(low, high))

    windows = [e for e in (events or []) if _is_set_piece(e)
               and int(e["end_frame"]) > first and int(e["start_frame"]) <= last]
    ball_speed = ball_speed_mps(x, y, pitch, fps)
    restarts = sum(detect_restart(ball_speed, max(int(e["start_frame"]), first),
                                  min(int(e["end_frame"]), stop), p.kick_speed_mps,
                                  p.kick_frames) is not None for e in windows)

    def share(mask, within):
        return float(mask[within].mean()) if within.any() else float("nan")

    return {
        "frames": int(stop - first),
        "speed_nan_share": float(np.mean(~np.isfinite(speed[first:stop]))),
        "speed_quantiles": tuple(round(float(v), 2)
                                 for v in np.percentile(finite, [5, 25, 50, 75, 95]))
        if len(finite) else (),
        "speed_threshold": round(threshold, 3),
        "below_threshold_share": float(np.mean(finite < threshold)) if len(finite) else 0.0,
        "speed_histogram": tuple(zip(np.round(edges[:-1], 2).tolist(), hist.tolist())),
        "ball_visible_share": float(visible.mean()),
        "ball_out_share_of_visible": share(out, visible),
        "set_piece_windows": len(windows),
        "restarts_found": int(restarts),
        "dead_share": float(dead.mean()),
        "dead_ball_visible_share": share(visible, dead),
        "dead_ball_moving_share": share(moving, dead),
        "alive_ball_moving_share": share(moving, ~dead),
    }


def print_report(half, report):
    print("\n== half {}  ({} frames)".format(half, report["frames"]))
    print("collective speed  NaN {:.1%}   p5/p25/p50/p75/p95 {}".format(
        report["speed_nan_share"], report["speed_quantiles"]))
    print("speed threshold   {:.3f} m/s (half median) -> {:.1%} below it".format(
        report["speed_threshold"], report["below_threshold_share"]))
    peak = max([count for _, count in report["speed_histogram"]] or [1])
    for edge, count in report["speed_histogram"]:
        marker = "  <- threshold" if edge <= report["speed_threshold"] < edge + 1e-9 + (
            report["speed_histogram"][1][0] - report["speed_histogram"][0][0]) else ""
        print("   {:>6.2f} | {:<40} {:>6}{}".format(edge, "#" * int(40 * count / peak), count,
                                                    marker))
    print("ball visible      {:.1%}   seen beyond a line {:.2%} of visible".format(
        report["ball_visible_share"], report["ball_out_share_of_visible"]))
    print("set pieces        {} windows, kick found in {}".format(
        report["set_piece_windows"], report["restarts_found"]))
    print("called dead       {:.1%} of the half".format(report["dead_share"]))
    print("  ball visible during dead frames      {:.1%}".format(
        report["dead_ball_visible_share"]))
    print("  ball MOVING in-pitch during dead     {:.1%}   (alive frames: {:.1%})".format(
        report["dead_ball_moving_share"], report["alive_ball_moving_share"]))


def main():
    import games
    from solution.run import DEFAULT_FEATURES, load_events, load_features
    n_frames = games.source_frames("tactical")
    features = load_features(DEFAULT_FEATURES, n_frames)
    events = load_events()
    pitch = games.GAMES[games.GAME]["pitch"]
    fps = games.GAMES[games.GAME]["fps"]
    for block in games.half_video_blocks(games.GAME):
        print_report(block[0], half_report(features, events, block, pitch, fps))


if __name__ == "__main__":
    main()
