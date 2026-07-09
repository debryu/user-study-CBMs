"""Verify the emails User Study participant-response CSVs before publishing
them to HuggingFace.

Checks, all hard assertions (fail loudly, not just print a diagnostic):
1. dataComplete_CorrectLabelsCondition.csv ("full") differs from
   dataComplete.csv (raw/internal) only in ExperimentalCondition's labels
   (relabeled to public names) -- no other cell changes anywhere, including
   for stimuli 263/268.
2. dataWideClean.csv ("clean_wide")'s 363 participants are a strict subset
   of dataComplete.csv's 417 -- 0 added. The 54 removed participants are
   fully explained by: blank/incomplete, OR failed AttentionCheck1/2, OR
   sum(Stim1..10_TimeTabWasLeft) > 3 (tab-switching risk) -- 0 unexplained.
3. For the 363 kept participants, every cell matches dataComplete.csv
   exactly except the known relabelings (ExperimentalCondition and the
   legitimate/fraudulent -> 0/1 answer columns) and stimulus slot 268 being
   blanked in place (not removed) -- 0 unexpected differences.
4. dataLongClean.csv ("clean_long") is a consistent long-format melt of the
   same 363 participants: TimeSpent rescaled (/60000), Confidence_Adjusted
   == abs(Confidence-7), stimulus 268 rows fully absent.

Example:
  cd emails && uv run scripts/verify_participant_data.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONDITION_RELABEL = {
    "BlackBox": "LabelOnly",
    "FixedCBM": "NonInteractiveConcepts",
    "InteractiveCBM": "InteractiveConcepts",
    "None": "NoSupport",
    "": "",
}
ANSWER_RELABEL = {"legitimate": "0", "fraudulent": "1", "": ""}


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def check_full_vs_raw(dc: pd.DataFrame, full: pd.DataFrame) -> None:
    assert list(dc.columns) == list(full.columns), "column mismatch between dataComplete.csv and full"
    assert list(dc["ID"]) == list(full["ID"]), "row order/ID mismatch"

    full_str = full.astype(str)
    dc_relabelled = dc.copy()
    dc_relabelled["ExperimentalCondition"] = dc_relabelled["ExperimentalCondition"].map(
        lambda v: CONDITION_RELABEL.get(v, v)
    )
    dc_relabelled_str = dc_relabelled.astype(str)

    diff_mask = dc_relabelled_str.values != full_str.values
    if diff_mask.any():
        rows, cols = np.where(diff_mask)
        bad_cols = sorted(set(full.columns[c] for c in cols))
        raise AssertionError(f"Unexpected diffs between full and dataComplete.csv (relabel-adjusted) in columns: {bad_cols}")
    print(f"[full vs raw] OK: identical to dataComplete.csv except ExperimentalCondition relabeling ({len(dc)} rows)")


def reconstruct_removed_participants(dc: pd.DataFrame, kept_ids: set) -> dict:
    stim_tab_cols = [f"Stim{i}_TimeTabWasLeft" for i in range(1, 11)]
    dc = dc.copy()
    dc["_total_tab_left"] = dc[stim_tab_cols].apply(to_num).sum(axis=1)
    dc["_blank"] = ~dc.drop(columns=["ID"]).apply(lambda row: (row != "").any(), axis=1)

    def attention_fail(row) -> bool:
        pass1 = row["AttentionCheck1_ParticipantAnswer"] == "legitimate" and str(row["AttentionCheck1_Confidence"]) == "13"
        pass2 = row["AttentionCheck2_ParticipantAnswer"] == "fraudulent" and str(row["AttentionCheck2_Confidence"]) == "1"
        return not (pass1 and pass2)

    dc["_attention_fail"] = dc.apply(attention_fail, axis=1)
    dc["_tab_left_excess"] = dc["_total_tab_left"] > 3

    removed = dc[~dc["ID"].isin(kept_ids)]
    unexplained = removed[~(removed["_blank"] | removed["_attention_fail"] | removed["_tab_left_excess"])]

    return {
        "removed_count": len(removed),
        "blank_count": int(removed["_blank"].sum()),
        "attention_fail_count": int(removed["_attention_fail"].sum()),
        "tab_left_excess_count": int(removed["_tab_left_excess"].sum()),
        "unexplained_ids": unexplained["ID"].tolist(),
        "max_tab_left_among_kept": dc[dc["ID"].isin(kept_ids)]["_total_tab_left"].max(),
    }


def check_kept_participants_match(dc: pd.DataFrame, wide: pd.DataFrame, kept_ids: set) -> None:
    original_cols = [c for c in dc.columns if c in wide.columns]
    dc_kept = dc[dc["ID"].isin(kept_ids)].set_index("ID").sort_index()
    wide_kept = wide[wide["ID"].isin(kept_ids)][original_cols].set_index("ID").sort_index()

    # Only the main Stim{i}_* answer columns are relabeled legitimate/fraudulent
    # -> 0/1 in dataWideClean.csv -- AttentionCheck*/PracticeTrial* keep the
    # original string values (verified directly against the source).
    answer_cols = [
        c for c in original_cols
        if c.startswith("Stim") and c.endswith(("CorrectAnswer", "ModelAnswer", "ParticipantAnswer"))
    ]

    dc_norm = dc_kept.copy()
    dc_norm["ExperimentalCondition"] = dc_norm["ExperimentalCondition"].map(lambda v: CONDITION_RELABEL.get(v, v))
    for col in answer_cols:
        dc_norm[col] = dc_norm[col].map(lambda v: ANSWER_RELABEL.get(v, v))

    # Stimulus 268 slots: blanked in wide, may still hold data in dc. Blank
    # those same slots in dc_norm before comparing, per participant.
    stim_cols_by_n = {i: [c for c in original_cols if c.startswith(f"Stim{i}_")] for i in range(1, 11)}
    for pid in dc_norm.index:
        for i in range(1, 11):
            if dc_kept.loc[pid, f"Stim{i}_StimID"] == "268":
                for c in stim_cols_by_n[i]:
                    dc_norm.loc[pid, c] = ""

    diff_mask = dc_norm.astype(str).values != wide_kept.astype(str).values
    if diff_mask.any():
        rows, cols = np.where(diff_mask)
        bad_cols = sorted(set(original_cols[c] for c in cols))
        bad_ids = sorted(set(dc_norm.index[r] for r in rows))
        raise AssertionError(
            f"Unexpected diffs for kept participants in columns {bad_cols} (IDs e.g. {bad_ids[:5]})"
        )
    print(f"[kept-participant match] OK: all {len(kept_ids)} kept participants' cells match "
          f"dataComplete.csv exactly (relabeling + stim-268-blanking accounted for)")


def check_long_consistency(wide: pd.DataFrame, long_df: pd.DataFrame) -> None:
    assert set(long_df["ID"]) == set(wide["ID"]), "clean_long participant set != clean_wide participant set"
    assert (long_df["StimID"] != "268").all() and (long_df["StimID"] != 268).all(), "StimID 268 present in clean_long"

    sample = long_df.merge(
        wide[["ID"] + [f"Stim{i}_{c}" for i in range(1, 11) for c in ("StimID", "TimeSpent", "Confidence")]],
        on="ID",
        suffixes=("", "_wide"),
    )
    checked = 0
    for i in range(1, 11):
        rows = sample[sample[f"Stim{i}_StimID"] == sample["StimID"].astype(str)]
        if rows.empty:
            continue
        expected_time = to_num(rows[f"Stim{i}_TimeSpent"]) / 60000
        actual_time = to_num(rows["TimeSpent"])
        assert np.allclose(expected_time, actual_time, atol=1e-6), f"TimeSpent scaling mismatch at Stim{i}"

        expected_conf_adj = (to_num(rows[f"Stim{i}_Confidence"]) - 7).abs()
        actual_conf_adj = to_num(rows["Confidence_Adjusted"])
        assert np.allclose(expected_conf_adj, actual_conf_adj, atol=1e-6), f"Confidence_Adjusted mismatch at Stim{i}"
        checked += len(rows)
    print(f"[clean_long consistency] OK: TimeSpent/Confidence_Adjusted verified across {checked} matched rows, "
          f"StimID 268 absent, participant set matches clean_wide")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="temp/User Study")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d = Path(args.dir)

    dc = pd.read_csv(d / "dataComplete.csv", keep_default_na=False)
    full = pd.read_csv(d / "dataComplete_CorrectLabelsCondition.csv", keep_default_na=False)
    wide = pd.read_csv(d / "dataWideClean.csv", keep_default_na=False)
    long_df = pd.read_csv(d / "dataLongClean.csv", keep_default_na=False)

    print(f"Loaded: dataComplete.csv ({len(dc)}), full ({len(full)}), "
          f"clean_wide ({len(wide)}), clean_long ({len(long_df)})")
    print()

    check_full_vs_raw(dc, full)
    print()

    kept_ids = set(wide["ID"])
    removed_report = reconstruct_removed_participants(dc, kept_ids)
    print(f"[removed-participant reconstruction] removed={removed_report['removed_count']}, "
          f"blank={removed_report['blank_count']}, attention_fail={removed_report['attention_fail_count']}, "
          f"tab_left_excess={removed_report['tab_left_excess_count']}, "
          f"max_tab_left_among_kept={removed_report['max_tab_left_among_kept']}")
    assert removed_report["removed_count"] == 54, f"expected 54 removed, got {removed_report['removed_count']}"
    assert not removed_report["unexplained_ids"], f"unexplained removed IDs: {removed_report['unexplained_ids']}"
    assert removed_report["max_tab_left_among_kept"] <= 3, "a kept participant exceeds TotalTimesTabLeft > 3"
    print("PASSED: all 54 removed participants explained, 0 unexplained.")
    print()

    check_kept_participants_match(dc, wide, kept_ids)
    print()

    check_long_consistency(wide, long_df)
    print()
    print("ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
