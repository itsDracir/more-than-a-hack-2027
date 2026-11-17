"""Dades a l'entrada, entrega a la sortida -- l'única ordre que regenera el fitxer que pugem.

    python solution/run.py --name <nom_equip>

Llegeix la taula de features per frame que escriu el constructor de features
(solution/preprocessing.ipynb escriu data/processed/features_tactical.csv; un possession.py que
produeixi les mateixes columnes funciona igual), hi afegeix les velocitats dels equips a partir de
l'ATD en brut si la taula no les té, llegeix el Smart Tagging, etiqueta cada frame i escriu
evaluation/results/<nom_equip>.csv, i després hi passa exactament la mateixa comprovació que fa el
scorer oficial.

Determinista: les mateixes dades d'entrada donen un fitxer idèntic byte a byte.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402
from solution.labeling import label_frames  # noqa: E402
from solution.params import Params  # noqa: E402
from solution.team_speed import team_mean_speeds  # noqa: E402

SPEED_COLUMNS = ["home_mean_speed_mps", "away_mean_speed_mps"]
DEFAULT_FEATURES = os.path.join(games.DATA_DIR, "processed", "features_tactical.csv")
SPEED_CACHE = os.path.join(games.DATA_DIR, "processed", "team_speed.npy")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

MISSING_FEATURES = """No feature table at {}

Build it first -- it is the per-frame table this step labels. Either run the preprocessing
notebook end to end (outputs go to data/processed/, which git ignores, so no match data ends up
in a commit):

    python -m jupyter nbconvert --to notebook --execute solution/preprocessing.ipynb --output-dir data/processed

or run your own feature builder (possession.py), as long as it writes the same columns
(see solution/labeling.py). Then run this again, or point --features at another file."""


def display_path(path):
    """Ruta relativa al repositori quan hi és a dins, absoluta si no. A Windows relpath falla amb
    una ruta en una altra unitat, i això no ha de fer petar una execució que ja ha acabat."""
    absolute = os.path.abspath(path)
    try:
        relative = os.path.relpath(absolute, ROOT)
    except ValueError:
        return absolute
    return absolute if relative.startswith("..") else relative


def load_features(path, n_frames):
    """La taula de features, una fila per frame de vídeo 0..n_frames-1 en ordre, o un error
    clar."""
    table = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)
    if "frame" not in table.columns:
        raise ValueError("{}: no 'frame' column".format(path))
    table = table.sort_values("frame").reset_index(drop=True)
    if len(table) != n_frames or not np.array_equal(table["frame"].to_numpy(),
                                                    np.arange(n_frames)):
        raise ValueError("{}: expected one row per video frame 0..{}, got {} rows"
                         .format(path, n_frames - 1, len(table)))
    if "ball_visible" in table.columns and table["ball_visible"].dtype != bool:
        table["ball_visible"] = table["ball_visible"].astype(str).str.lower() == "true"
    return table


def add_team_speeds(features, n_frames, refresh=False):
    """Omple les columnes de velocitat a partir de l'ATD en brut (una passada, en memòria cau)
    si la taula no les té."""
    if all(c in features.columns for c in SPEED_COLUMNS):
        return features
    if os.path.exists(SPEED_CACHE) and not refresh:
        speeds = np.load(SPEED_CACHE)
        print("team speeds: cache {}".format(display_path(SPEED_CACHE)))
    else:
        from starter.baseline import team_of_each_track
        meta_path, raw_path = games.require(["atd_meta", "atd_raw"])
        print("team speeds: reading {} (one pass, about a minute) ..."
              .format(os.path.basename(raw_path)))
        with open(raw_path, encoding="utf-8-sig") as handle:
            speeds = team_mean_speeds(handle, team_of_each_track(meta_path), n_frames)
        os.makedirs(os.path.dirname(SPEED_CACHE), exist_ok=True)
        np.save(SPEED_CACHE, speeds)
    features = features.copy()
    features[SPEED_COLUMNS[0]], features[SPEED_COLUMNS[1]] = speeds[:, 0], speeds[:, 1]
    return features


def load_events():
    """Intervals del Smart Tagging, o None (amb un avís) si el fitxer no hi és."""
    from starter import load_smart_tagging
    try:
        return load_smart_tagging.load()
    except Exception as error:  # fitxer absent: millor funcionar sense que no funcionar
        print("WARNING: running without Smart Tagging ({})".format(error))
        return None


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def summarise(labels, blocks):
    """El que cal revisar abans de pujar-lo -- calculat sobre el nostre fitxer, sense ground
    truth."""
    runs = [(first, last + 1) for _, first, last in blocks]
    graded = np.concatenate([labels[a:b] for a, b in runs])
    print("\ngraded frames: {}".format(len(graded)))
    for label in "HADX":
        count = int(np.sum(graded == label))
        print("  {}  {:>7}  {:5.1f}%".format(label, count, 100.0 * count / len(graded)))
    codes = pm.fill_abstentions(pm.label_codes(labels), runs)
    stats = pm.possession_stats(codes, runs)
    print("possessions: {}  mean {:.1f} s  home share {:.1%}".format(
        stats["n_spells"], stats["mean_duration_s"], stats["home_share"]))
    print("(sanity check, not a target: D is ball-out-of-play time, roughly 35-45% of a "
          "professional match; a D share far outside that means the dead-ball signal is off.)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="team name: the file is <name>.csv")
    parser.add_argument("--features", default=DEFAULT_FEATURES, help="per-frame feature table")
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--no-smart-tagging", action="store_true")
    parser.add_argument("--refresh-speeds", action="store_true",
                        help="re-read the ATD instead of using the cached team speeds")
    args = parser.parse_args(argv)

    if not os.path.exists(args.features):
        print(MISSING_FEATURES.format(display_path(args.features)), file=sys.stderr)
        return 2

    n_frames = games.source_frames("tactical")
    blocks = games.half_video_blocks(games.GAME)
    features = add_team_speeds(load_features(args.features, n_frames), n_frames,
                               args.refresh_speeds)
    events = None if args.no_smart_tagging else load_events()

    labels = label_frames(features, events, blocks, params=Params())
    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)

    problem = pm.validate_submission(pm.read_label_csv(path), n_frames)
    print("\n{}  {}".format("INVALID: " + problem if problem else "valid",
                            display_path(path)))
    summarise(labels, blocks)
    print("\nscore on the public window:  python evaluation/run_evaluation.py")
    return 1 if problem else 0


if __name__ == "__main__":
    sys.exit(main())
