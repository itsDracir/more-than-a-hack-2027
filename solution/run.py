"""Data in, submission out -- the one command that regenerates the file we upload.

    python solution/run.py --name <team_name>

Reads the per-frame feature table from solution/01_preprocessing.ipynb and the
SET PIECES intervals cached during preprocessing. No raw ATD or Smart Tagging XML is opened
at prediction time. The model produces evaluation/results/<team_name>.csv, then validates it.

Deterministic: same data in, byte-identical file out.
"""

import argparse
import json
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

SPEED_COLUMNS = ["home_mean_speed_mps", "away_mean_speed_mps"]
DEFAULT_FEATURES = os.path.join(games.DATA_DIR, "processed", "features_tactical.csv")
DEFAULT_EVENTS = os.path.join(games.DATA_DIR, "processed", "smart_tagging_events.json")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

MISSING_FEATURES = """No feature table at {}

Executa primer el notebook solution/01_preprocessing.ipynb, que guarda les features
privades a data/processed/ (o a la ruta configurada). Després executa run.py.
També pots indicar una ruta alternativa amb --features."""


def display_path(path):
    """Path relative to the repository when it is inside it, absolute otherwise. On Windows
    relpath raises for a path on another drive, which must not crash a finished run."""
    absolute = os.path.abspath(path)
    try:
        relative = os.path.relpath(absolute, ROOT)
    except ValueError:
        return absolute
    return absolute if relative.startswith("..") else relative


def load_features(path, n_frames):
    """The feature table, one row per video frame 0..n_frames-1 in order, or a clear error."""
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
    """Comprova que ja tenim les velocitats calculades al preprocessament.

    Conservem el nom per compatibilitat amb diagnose.py; no rellegim l'ATD.
    """
    missing = [c for c in SPEED_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(
            "Falten {}. Torna a executar solution/01_preprocessing.ipynb: "
            "ara calcula les velocitats mitjanes en la primera lectura de l'ATD."
            .format(missing)
        )
    if len(features) != n_frames:
        raise ValueError("Nombre incorrecte de frames")
    return features


def load_events(path=DEFAULT_EVENTS):
    """Obté els intervals de reinici preprocessats; mai obre l'XML original."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "No trobo {}. Executa solution/01_preprocessing.ipynb per generar "
            "smart_tagging_events.json, passa --events o fes servir --no-smart-tagging."
            .format(display_path(path))
        )
    with open(path, encoding="utf-8") as handle:
        events = json.load(handle)
    if not isinstance(events, list):
        raise ValueError("El JSON d'Smart Tagging ha de contenir una llista d'intervals")
    for event in events:
        if not isinstance(event, dict) or not all(
            key in event for key in ("code", "start_frame", "end_frame", "tags")
        ):
            raise ValueError("Format inesperat a smart_tagging_events.json")
    return events


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def summarise(labels, blocks):
    """What to sanity-check before uploading -- computed on our own file, no ground truth."""
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
    parser.add_argument("--events", default=DEFAULT_EVENTS,
                        help="JSON d'intervals SET PIECES produït pel preprocessament")
    parser.add_argument("--no-smart-tagging", action="store_true",
                        help="Ignora tant els intervals de reinici com les features tàctiques")
    args = parser.parse_args(argv)

    if not os.path.exists(args.features):
        print(MISSING_FEATURES.format(display_path(args.features)), file=sys.stderr)
        return 2

    n_frames = games.source_frames("tactical")
    blocks = games.half_video_blocks(games.GAME)
    features = add_team_speeds(load_features(args.features, n_frames), n_frames)
    events = None if args.no_smart_tagging else load_events(args.events)

    labels = label_frames(features, events, blocks, params=Params(),
                          use_tactical_tags=not args.no_smart_tagging)
    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)

    problem = pm.validate_submission(pm.read_label_csv(path), n_frames)
    print("\n{}  {}".format("INVALID: " + problem if problem else "valid",
                            display_path(path)))
    summarise(labels, blocks)
    print("\nscore on the public window:  python evaluation/run_evaluation.py")
    return 1 if problem else 0


if __name__ == "__main__":
    sys.exit(main())
