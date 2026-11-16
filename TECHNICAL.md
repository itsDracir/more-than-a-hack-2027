# Technical document — More than a Hack 2027, Phase 1: Ball Possession

## Approach

Per-frame labelling (`H` / `A` / `D`) turned into a **sequence decoding** problem. The baseline votes
frame by frame inside a window; we instead model possession as a hidden state that is *sticky*,
decoded with a Viterbi HMM (`solution/possession.py`). A possession is therefore one segment with a
start and an end, which is what `spellF1` rewards.

- **Emissions (per frame).** With the ball visible, the evidence for `H` vs `A` is a logistic of the
  margin between the nearest away and nearest home player to the ball (metres, clipped to ±10 m).
  Evidence is down-weighted by `1 / (1 + (d_min / tau)^2)` (tau = 3 m): when nobody is near the ball it is in
  flight and the nearest player says little. Ball outside the pitch, or far from everyone, is evidence for `D`.
- **Invisible ball (41 % of graded frames).** No evidence, so the state is carried across the gap.
  A missing ball is mild evidence for `D`, never proof.
- **Transitions.** Staying is free; `H<->A` costs 25, entering/leaving `D` costs 8. This sets the
  typical possession length and removes the flicker.
- **Smart Tagging.** Frames inside a `SET PIECES` interval get a small bonus (0.5) for `D`.

## Assets used

| asset | use |
|---|---|
| tactical ATD tracking (players + ball) | main signal: ball position, nearest player per team |
| automatic Smart Tagging | `SET PIECES` intervals as weak evidence for dead ball |
| tactical homography, both videos, broadcast tracking | **not used** (the tracking already gives pitch coordinates) |

Per-frame features come from `solution/preprocessing.ipynb` (Marc). It reads the tracking in
streaming mode and writes `data/processed/features_tactical.csv`, which is private and never committed.

## Training data

**None.** No model is trained. No labelled data of any kind was used, from this game or any other.
The five tunable numbers (`k`, `tau`, switch cost, dead cost, set-piece bonus) were set by hand.
The public 5-minute ground truth was used only to check the pipeline and compare variants, as
intended; the full-match ground truth was not available to us.

## Possession rules implemented

- Possession goes to the team of the player nearest to the ball; no tracklet identities are used
  (the 46 "Track" slots are fragments, not persons).
- A loose ball, a pass in flight or a challenge does not change the state unless a sustained shift
  in nearest-player evidence pays the switch cost.
- Restarts: `SET PIECES` tag intervals lean towards `D`; there is no other restart logic.
- Dead ball is never assigned to a team. `X` is never emitted.

## Results (public 5 minutes, 17 possessions, so noisy)

| | spellF1 | macroF1 |
|---|---|---|
| this solution | 0.4593 | 0.567 |
| baseline_smooth125 | 0.4242 | 0.581 |
| baseline_smooth50 (published floor on full match: 0.2433) | 0.3256 | 0.621 |

The bootstrap interval on this sample is about ±0.4, so the ordering is not established. On the full
match the decoder outputs 296 possessions.

## Limitations

- Parameters were chosen on 17 public possessions; results move a lot with `tau` and the switch
  cost (spellF1 0.24–0.46 across the sweep). Expect regression to the mean on the full match.
- `D` is weak: no use of ball speed, ball height or the restart types.
- Nearest-player is a poor proxy in duels, crowded areas and for long passes.
- Mean possession on the full match is about 26 s versus 14 s on the public window, so it may be
  over-smoothed.
- Ideas with more time: ball speed and acceleration, learned emissions from other games' labels,
  using the broadcast tracking to fill ball gaps, per-half calibration of the switch cost.

## Reproduction

```bash
pip install -r requirements-lock.txt        # Python 3.14; numpy/pandas only for the decoder
# put the match data in ./data (or set MTAH_DATA_DIR)
python -m jupyter nbconvert --to notebook --execute solution/preprocessing.ipynb --output-dir <tmpdir>
python run.py                               # writes submission/submission.csv and validates it
```

The result is deterministic: running it again gives the same file.
