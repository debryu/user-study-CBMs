"""Supplementary analysis of the CUB User Study participant data:
time-on-task vs. support modality, decisiveness (answer/concept-value
changes) vs. support modality, and correlations between trust, confidence,
interaction behavior, and time-on-task.

Reads temp/dataWideClean.csv and temp/dataLongClean.csv -- the official
cleaned participant data downloaded from Drive (user_study_results_cub/).
An earlier version of this pipeline self-generated these files
(cub/scripts/prepare_analysis_data.py) before an official version existed;
that script's output was verified byte-for-byte equivalent to the official
files (identical 342-participant set, identical values modulo cosmetic
condition/answer relabeling) once the official files became available, so
the exclusion criteria it documents (attention-check pass/fail,
sum(TimeTabWasLeft) > 3) are now confirmed correct for CUB, not just
assumed by analogy with emails. Shared plotting logic lives in
common/participant_analysis.py (also used by
emails/scripts/supplementary_analysis.py).

Example:
  cd cub && uv run scripts/supplementary_analysis.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.participant_analysis import run_all  # noqa: E402

CONDITION_ORDER = ["NoSupport", "LabelOnly", "NonInteractiveConcepts", "InteractiveConcepts"]
INTERACTIVE_CONDITION = "InteractiveConcepts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="temp")
    parser.add_argument("--output-dir", default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    long_df = pd.read_csv(data_dir / "dataLongClean.csv", keep_default_na=False)
    wide_df = pd.read_csv(data_dir / "dataWideClean.csv", keep_default_na=False)
    run_all(long_df, wide_df, Path(args.output_dir), CONDITION_ORDER, INTERACTIVE_CONDITION, "CUB")


if __name__ == "__main__":
    main()
