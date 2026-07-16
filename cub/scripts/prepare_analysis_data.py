"""Build cleaned wide/long participant-response tables for CUB, mirroring
emails/scripts/verify_participant_data.py's dataWideClean.csv/dataLongClean.csv
outputs -- needed because no equivalent cleaning/exclusion pipeline exists
yet for CUB (unlike emails, `cub/temp/updated_complete.csv` has no
attention-check/tab-switching exclusion applied to it at all).

IMPORTANT CAVEAT: the exclusion criteria here (attention-check pass/fail,
sum(TimeTabWasLeft) > 3) are carried over **by analogy** from the emails
study, where they were confirmed by the researcher. For CUB, the
attention-check logic is verified directly against this data (see below),
but the specific ">3" tab-switching threshold is an assumption, not
independently confirmed for CUB -- re-check with the researcher before
treating results derived from it as final.

Attention-check pass criterion (verified against this data: 355/568
participants match this exact pattern, by far the largest single group):
- AttentionCheck1 (correct answer is always "Le Conte"): ParticipantAnswer
  == "Le Conte" AND Confidence == 1 (the bipolar 1-13 confidence scale's
  "Le Conte" extreme).
- AttentionCheck2 (correct answer is always "Savannah"): ParticipantAnswer
  == "Savannah" AND Confidence == 13 (the "Savannah" extreme).

Example:
  cd cub && uv run scripts/prepare_analysis_data.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_STIMULI = 10
TRUST_REVERSED_ITEMS = {2, 5, 6, 8}  # same reversed items as emails' TrustIndex
TAB_SWITCH_THRESHOLD = 3  # see module docstring -- assumed, not confirmed for CUB


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_trust_index(df: pd.DataFrame) -> pd.Series:
    items = []
    for i in range(1, 9):
        col = to_num(df[f"TrustQuestionnaire_Q{i}"])
        if i in TRUST_REVERSED_ITEMS:
            col = 8 - col
        items.append(col)
    return pd.concat(items, axis=1).mean(axis=1)


def reconstruct_removed_participants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tab_cols = [f"Stim{i}_TimeTabWasLeft" for i in range(1, N_STIMULI + 1)]
    df["_total_tab_left"] = df[tab_cols].apply(to_num).sum(axis=1)
    df["_blank"] = (df["Finished"] != "complete")

    pass1 = (df["AttentionCheck1_ParticipantAnswer"] == "Le Conte") & (df["AttentionCheck1_Confidence"] == "1")
    pass2 = (df["AttentionCheck2_ParticipantAnswer"] == "Savannah") & (df["AttentionCheck2_Confidence"] == "13")
    df["_attention_fail"] = ~(pass1 & pass2)
    df["_tab_left_excess"] = df["_total_tab_left"] > TAB_SWITCH_THRESHOLD
    df["_keep"] = ~(df["_blank"] | df["_attention_fail"] | df["_tab_left_excess"])
    return df


def build_wide_clean(df: pd.DataFrame) -> pd.DataFrame:
    annotated = reconstruct_removed_participants(df)
    kept = annotated[annotated["_keep"]].drop(columns=["_blank", "_attention_fail", "_tab_left_excess", "_keep"])
    kept = kept.copy()
    kept["TrustIndex"] = compute_trust_index(kept)

    click_cols = [f"Stim{i}_Feature{j}_Clicks" for i in range(1, N_STIMULI + 1) for j in range(1, 7)]
    kept["TotalInteractions"] = kept[click_cols].apply(to_num).apply(lambda c: (c > 0).astype(int)).sum(axis=1)

    correct_cols = [f"Stim{i}_ParticipantAnswer" for i in range(1, N_STIMULI + 1)]
    gt_cols = [f"Stim{i}_CorrectAnswer" for i in range(1, N_STIMULI + 1)]
    n_shown = kept[correct_cols].apply(lambda c: (c != "")).sum(axis=1)
    n_correct = sum((kept[pc] == kept[gc]) & (kept[pc] != "") for pc, gc in zip(correct_cols, gt_cols))
    kept["AverageAccuracy"] = n_correct / n_shown.replace(0, np.nan)

    tab_cols = [f"Stim{i}_TimeTabWasLeft" for i in range(1, N_STIMULI + 1)]
    kept["TotalTimesTabLeft"] = kept[tab_cols].apply(to_num).sum(axis=1)

    return kept.reset_index(drop=True)


def build_long_clean(wide_clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(1, N_STIMULI + 1):
        prefix = f"Stim{i}_"
        sub = wide_clean[wide_clean[f"{prefix}StimID"] != ""]
        if sub.empty:
            continue
        record = pd.DataFrame({
            "ID": sub["ID"],
            "ExperimentalCondition": sub["ExperimentalCondition"],
            "StimulusSlot": i,
            "StimID": sub[f"{prefix}StimID"],
            "TestSampleIdx": sub[f"{prefix}TestSampleIdx"],
            "CorrectAnswer": sub[f"{prefix}CorrectAnswer"],
            "ModelAnswer": sub[f"{prefix}ModelAnswer"],
            "TimeTabWasLeft": to_num(sub[f"{prefix}TimeTabWasLeft"]),
            "TimeSpent": to_num(sub[f"{prefix}TimeSpent"]) / 60000,  # ms -> minutes, same convention as emails
            "TimesAnswerChanged": to_num(sub[f"{prefix}TimesAnswerChanged"]),
            "ParticipantAnswer": sub[f"{prefix}ParticipantAnswer"],
            "Confidence": to_num(sub[f"{prefix}Confidence"]),
        })
        click_cols = [f"{prefix}Feature{j}_Clicks" for j in range(1, 7)]
        record["Interactions"] = (sub[click_cols].apply(to_num).sum(axis=1) > 0).astype(int)
        record["ParticipantAnswerCorrect"] = (record["ParticipantAnswer"] == record["CorrectAnswer"]).astype(int)
        record["ModelAnswerCorrect"] = (sub[f"{prefix}ModelAnswer"] == sub[f"{prefix}CorrectAnswer"]).astype(int)
        for j in range(1, 7):
            record[f"Feature{j}_GT"] = sub[f"{prefix}Feature{j}_GT"].values
            record[f"Feature{j}_Detected"] = to_num(sub[f"{prefix}Feature{j}_Detected"]).values
            record[f"Feature{j}_Clicks"] = to_num(sub[f"{prefix}Feature{j}_Clicks"]).values
        rows.append(record)
    long_df = pd.concat(rows, ignore_index=True)
    # Confidence_Adjusted: same "direction-independent" fold as emails, but
    # CUB's raw Confidence scale is 1-13 (bipolar Le Conte/Savannah), not
    # emails' 1-13 legitimate/fraudulent -- same numeric range, same fold formula.
    long_df["Confidence_Adjusted"] = (long_df["Confidence"] - 7).abs()
    return long_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="temp/updated_complete.csv")
    parser.add_argument("--output-dir", default="temp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, keep_default_na=False)
    print(f"Loaded {len(df)} raw participants")

    annotated = reconstruct_removed_participants(df)
    removed = annotated[~annotated["_keep"]]
    print(f"Removed {len(removed)} participants: blank/incomplete={annotated['_blank'].sum()}, "
          f"attention_fail={annotated['_attention_fail'].sum()}, "
          f"tab_left_excess={annotated['_tab_left_excess'].sum()} (threshold={TAB_SWITCH_THRESHOLD}, assumed)")

    wide_clean = build_wide_clean(df)
    long_clean = build_long_clean(wide_clean)
    print(f"Kept {len(wide_clean)} participants -> {len(long_clean)} stimulus-rows")
    print(wide_clean["ExperimentalCondition"].value_counts())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wide_clean.to_csv(out_dir / "dataWideClean.csv", index=False)
    long_clean.to_csv(out_dir / "dataLongClean.csv", index=False)
    print(f"Saved {out_dir / 'dataWideClean.csv'} and {out_dir / 'dataLongClean.csv'}")


if __name__ == "__main__":
    main()
