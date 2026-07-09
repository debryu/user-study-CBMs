# Participant response data: verification and publishing report

Companion to [`DATASET_NOTES.md`](DATASET_NOTES.md) (which covers the
263/268 model-activation issue in depth). This report covers the separate
question of *participant* data quality: whether the manually-cleaned files
handed over for publishing are correct, and what got published where.

## Files involved

All in `emails/temp/User Study/` (gitignored, not tracked):

| File | Rows | Shape | Role |
|---|---|---|---|
| `dataComplete.csv` | 417 | 1 row / participant | Raw/internal export (internal condition codenames) |
| `dataComplete_CorrectLabelsCondition.csv` | 417 | 1 row / participant | **"full"** — same data, public condition labels |
| `dataWideClean.csv` | 363 | 1 row / participant | **"clean_wide"** — filtered + derived columns |
| `dataLongClean.csv` | 3539 | 1 row / participant-stimulus | **"clean_long"** — same 363 participants, melted, main 10 stimuli only |

## What was verified

### 1. "full" matches "raw" exactly, except relabeling

`dataComplete_CorrectLabelsCondition.csv` was diffed cell-by-cell against
`dataComplete.csv`: **the only column that differs is `ExperimentalCondition`**,
relabeled to public names (`BlackBox→LabelOnly`,
`FixedCBM→NonInteractiveConcepts`, `InteractiveCBM→InteractiveConcepts`,
`None→NoSupport`). Every `Feature_GT`/`Feature_Detected` value — including
for the known-bad stimuli 263/268 — is untouched.

### 2. "full" and "clean_wide" pass the model-reproduction sanity checks

Reused the hard-assertion checks already in `scripts/train_concept_extractor2.ipynb`
(same `UI_TO_OUR_PERM` permutation, `ROUNDTRIP_TOLERANCE = 1e-3`, factored
into a shared `check_against_participants()` helper) against both files:

- **`dataComplete_CorrectLabelsCondition.csv`**: 0 unexpected activation
  mismatches (28 checked, excluding 263/268) and 0 unexpected GT mismatches
  (29 checked, excluding 268). Expected — trivial given check 1 above.
- **`dataWideClean.csv`**: 0 unexpected activation mismatches (28 checked,
  excluding 263 only) and 0 unexpected GT mismatches (29 checked, **no
  exclusion needed** — 268 never appears in the walk since its stimulus
  slots are already blanked/absent in this file).

Both print `PASSED`.

### 3. "clean_wide"/"clean_long" remove *only* the correct participants

`dataWideClean.csv` keeps 363 of `dataComplete.csv`'s 417 participants (54
removed, 0 added — verified as a strict subset). The 54 removed are fully
explained (0 unexplained remainder) by three criteria, checked against every
removed participant:

| Criterion | Count* |
|---|---|
| Blank / incomplete (no data beyond `ID`, or `Finished != 'complete'`) | overlaps below |
| Failed `AttentionCheck1`/`AttentionCheck2` (wrong answer or confidence not at the correct scale extreme) | 35 |
| `sum(Stim1..10_TimeTabWasLeft) > 3` — tab-switching risk (e.g. possible use of an external tool/LLM to answer) | 22 |

*Criteria overlap (some participants fail more than one), so counts don't
sum to 54 — the important number is **0 unexplained**. The third criterion
(`TotalTimesTabLeft > 3`) was the one initially not derivable from
`dataComplete.csv` alone; confirmed against `dataWideClean.csv`'s own
precomputed `TotalTimesTabLeft` column with **0 mismatches across all 363
kept participants**, and **all 363 kept participants have `TotalTimesTabLeft
<= 3`** (no exceptions in either direction).

For the 363 kept participants, **every cell matches `dataComplete.csv`
exactly**, except the known, accounted-for transformations: `ExperimentalCondition`
and the `Stim{i}_{CorrectAnswer,ModelAnswer,ParticipantAnswer}` relabeling
(`legitimate→0`, `fraudulent→1` — note this relabeling applies only to the
main `Stim{i}_*` answer columns, not `AttentionCheck*`/`PracticeTrial*`,
which keep the original string values), and stimulus 268's columns blanked
in place (slot position preserved, not shifted) for the 91 participants who
had it.

`dataLongClean.csv` was checked for internal consistency against
`dataWideClean.csv`: same 363-participant set, `TimeSpent` correctly
rescaled (`raw_ms / 60000`), `Confidence_Adjusted` correctly computed
(`abs(Confidence - 7)`), and stimulus 268 rows fully absent (not just
blanked) — verified across all 3539 rows.

All of the above is automated and repeatable via
`scripts/verify_participant_data.py` (`cd emails && uv run scripts/verify_participant_data.py`)
— re-run this after any future re-cleaning before re-publishing.

## Published to HuggingFace

`NWeak/CBM-user-study-emails` (private), three configs, pushed via
`scripts/push_participant_responses_to_hub.py`:

| Config | Source file | Rows |
|---|---|---|
| `full` | `dataComplete_CorrectLabelsCondition.csv` | 417 |
| `clean_wide` | `dataWideClean.csv` | 363 |
| `clean_long` | `dataLongClean.csv` | 3539 |

```python
from datasets import load_dataset
load_dataset("NWeak/CBM-user-study-emails", "full", split="test")
load_dataset("NWeak/CBM-user-study-emails", "clean_wide", split="test")
load_dataset("NWeak/CBM-user-study-emails", "clean_long", split="test")
```

This is distinct from
[`NWeak/emails-user-study`](https://huggingface.co/datasets/NWeak/emails-user-study),
which holds *our model's computed activations/predictions*, not participant
responses.

## 263/268 recap

See [`DATASET_NOTES.md`](DATASET_NOTES.md) for the full root-cause writeup.
In short: an annotator copy-paste error meant the activations shown to
participants for stimuli 263 and 268 didn't match the model's real output.
268's error also flipped which concepts were active — participants saw
materially different information, so all its observations are excluded from
`clean_wide`/`clean_long` (blanked/dropped). 263's error preserved the same
binary concept pattern, so its observations were kept (GT is correct, only
its recorded activation *magnitudes* are unreliable — still excluded from
activation-level sanity checks, but not from the published clean dataset).
