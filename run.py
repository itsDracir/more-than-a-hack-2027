"""Reproduce the submission from the raw match data.

    pip install -r requirements-lock.txt
    python -m jupyter nbconvert --to notebook --execute solution/preprocessing.ipynb --output-dir <tmpdir>
    python run.py

Step 2 writes data/processed/features_tactical.csv (private, never committed). This script reads it,
decodes possession and writes submission/submission.csv, then validates it with the official checker.
Outputs of the executed notebook go to a temp dir so no match data ends up in a tracked .ipynb.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import possession  # noqa: E402
import numpy as np  # noqa: E402

FEATURES = os.path.join(possession.CACHE_DIR, "features_tactical.csv")
OUT = os.path.join(ROOT, "submission", "submission.csv")


def main():
    if not os.path.isfile(FEATURES):
        sys.exit("Missing {}.\nRun the preprocessing notebook first (see the docstring).".format(FEATURES))

    bx, by, dist, pitch, setp = possession.load_features()
    em = possession.emissions(bx, by, dist, pitch, setp, k=0.3, sp=0.5, tau=3.0)
    labels = possession.LABELS[possession.viterbi(em, switch=25.0, to_dead=8.0)]
    possession.write_submission(OUT, labels)
    print({lab: int(np.sum(labels == lab)) for lab in possession.LABELS}, "->", os.path.relpath(OUT, ROOT))

    subprocess.check_call([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), OUT])


if __name__ == "__main__":
    main()
