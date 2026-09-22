"""Clean up the raw CUB user-study export into a HuggingFace-ready CSV.

SUPERSEDED: push_user_study_to_hub.py now reads the three official Drive
exports directly and derives StimX_TestSampleIdx itself, so this script is no
longer part of the publishing path. It also reads dataComplete.csv, whose
*internal* condition labels (None / BlackBox / FixedCBM / InteractiveCBM) are
not what we publish -- the public labels come from
dataComplete_CorrectLabelsCondition.csv. Kept because prepare_analysis_data.py
still consumes its output for the provenance derivation.

For each StimX_StimID (X = 1..10), inserts a StimX_TestSampleIdx column right
after it: the integer sample_idx that indexes into the CUB test split (see
cub/data/cub_csv/test.csv or the NWeak/cub-mirror dataset's test split), e.g.
"1_003531_Final2.png" -> 3531. Also relabels the "None" (no-support baseline)
ExperimentalCondition to "NoSupport", and drops rows where the participant
never actually started (every column blank except ID).

Example:
  cd cub && uv run scripts/prepare_user_study_dataset.py
"""

import argparse
from pathlib import Path

import pandas as pd

N_STIMULI = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="temp/dataComplete.csv")
    parser.add_argument("--output", default="temp/updated_complete.csv")
    return parser.parse_args()


def extract_id(x: str) -> int:
    if x == "":
        return -1
    return int(x.split("_")[1])


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, keep_default_na=False)

    started = df.drop(columns=["ID"]).apply(lambda row: (row != "").any(), axis=1)
    dropped = (~started).sum()
    df = df[started].reset_index(drop=True)
    print(f"Dropped {dropped} row(s) with no data beyond ID")

    for stim in range(1, N_STIMULI + 1):
        stim_id_col = f"Stim{stim}_StimID"
        idx_col = f"Stim{stim}_TestSampleIdx"
        insert_at = df.columns.get_loc(stim_id_col) + 1
        df.insert(insert_at, idx_col, df[stim_id_col].map(extract_id))

    n_none = (df["ExperimentalCondition"] == "None").sum()
    df["ExperimentalCondition"] = df["ExperimentalCondition"].replace("None", "NoSupport")
    print(f"Relabeled {n_none} row(s): ExperimentalCondition 'None' -> 'NoSupport'")

    print(df["ExperimentalCondition"].value_counts())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
