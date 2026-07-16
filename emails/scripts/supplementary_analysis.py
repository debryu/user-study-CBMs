"""Supplementary analysis of the emails User Study participant data:
time-on-task vs. support modality, decisiveness (answer/concept-value
changes) vs. support modality, and correlations between trust, confidence,
interaction behavior, and time-on-task.

Reads the published "clean" participant data (dataWideClean.csv,
dataLongClean.csv -- see emails/PARTICIPANT_DATA_REPORT.md), saves PNG
figures plus a text summary of the underlying stats (Kruskal-Wallis for
condition comparisons, Spearman for correlations -- both nonparametric,
appropriate for the skewed time/count distributions here). Shared plotting
logic lives in common/participant_analysis.py (also used by
cub/scripts/supplementary_analysis.py).

Example:
  cd emails && uv run scripts/supplementary_analysis.py
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
    parser.add_argument("--data-dir", default="temp/User Study")
    parser.add_argument("--output-dir", default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    long_df = pd.read_csv(data_dir / "dataLongClean.csv", keep_default_na=False)
    wide_df = pd.read_csv(data_dir / "dataWideClean.csv", keep_default_na=False)
    run_all(long_df, wide_df, Path(args.output_dir), CONDITION_ORDER, INTERACTIVE_CONDITION)


if __name__ == "__main__":
    main()
