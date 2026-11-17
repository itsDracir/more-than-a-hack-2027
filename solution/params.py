"""Every tunable number of the pipeline, in one place, with the reason it has the value it has.

None of these were fitted to the public five minutes -- RULES.md forbids it, and EVALUATION.md
shows it picks the wrong answer anyway. They are set from what a football match looks like and
from the units they are in. If you change one, change it for a reason you could write in the
technical document, not because the public window went up.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # -- dead ball -------------------------------------------------------------------------
    # Collective player speed is smoothed with a centred median over this many seconds: long
    # enough to ignore a jog-back, short enough to keep the edges of a stoppage sharp.
    speed_smooth_s: float = 2.0
    # Speed is compared with the half's median pace ("slower than usual"), saturating within
    # this fraction of it. It is WEAK evidence: on the real match the speed distribution is one
    # skewed hump, not two populations, and slow possession play is as slow as a stoppage.
    speed_scale_frac: float = 0.15
    w_speed: float = 0.5
    # The ball travelling more than move_min_m within move_window_s, inside the pitch, is play.
    # 2 m in 1 s is slower than any pass and faster than tracker jitter on a parked ball. It
    # outweighs slow players (w_move > w_speed), which is the point.
    move_window_s: float = 1.0
    move_min_m: float = 2.0
    w_move: float = 2.0
    # Ball seen beyond a line by more than this is out of play, whatever the players do.
    oob_margin_m: float = 0.5
    w_oob: float = 3.0
    # Inside a Smart Tagging SET PIECES window, before the detected kick: dead. Just after: alive.
    w_setpiece: float = 1.5
    # A kick is the ball moving faster than this for this many consecutive frames.
    kick_speed_mps: float = 3.0
    kick_frames: int = 3
    # Cost of entering or leaving a stoppage: ~1.6 s of full speed evidence (40 frames x 0.5).
    # Real stoppages last several seconds; this keeps a brief lull from being called dead.
    dead_switch_cost: float = 20.0
    # Per-frame lean towards play when nothing says otherwise. The ball is in play most of the
    # match and a stoppage needs a cause; without this, neutral stretches next to a stoppage are
    # absorbed into it because staying dead is free. 20 / 0.1 = 8 s of neutral frames pays to
    # get back to play.
    dead_prior: float = -0.1

    # -- which team ------------------------------------------------------------------------
    # A player within about this distance of the ball is plausibly playing it; evidence decays
    # exponentially beyond it, so a ball in flight far from everyone says almost nothing.
    control_radius_m: float = 1.5
    # How much closer one team must be for the evidence to saturate.
    margin_scale_m: float = 1.0
    # Cost of a change of possession, in units of per-frame evidence. A player clearly on the
    # ball gives ~0.7 per frame, so this asks for ~0.5 s of clear control by the other team:
    # long enough to ignore deflections and one-touch duels (CHALLENGE.md: "a simple touch of
    # the ball without control" is not a change), short enough to catch a real interception.
    team_switch_cost: float = 10.0
    # When moving a change of possession to the new team's first touch, a frame counts as a
    # touch when its evidence is at least this strong.
    touch_evidence: float = 0.2
    # Evidència suau de les etiquetes ofensives de Smart Tagging ja preprocessades (H - A).
    # No utilitzem etiquetes defensives ni considerem que un tag sigui ground truth.
    #tactical_team_weight: float = 0.04
    tactical_team_weight: float = 0.04
