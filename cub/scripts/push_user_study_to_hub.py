"""Push the CUB user-study participant responses to the HF Hub, mirroring
emails/scripts/push_participant_responses_to_hub.py.

Reads the three official exports straight from Drive's user_study_results_cub/
(downloaded into cub/temp/) and pushes them as three configs of one repo:
- "full": 568 participants (dataComplete_CorrectLabelsCondition.csv, minus 3
  rows where the participant never started -- every column blank except ID).
- "clean_wide": 342 participants, one row each (dataWideClean.csv) --
  blank/incomplete, failed-attention-check, and excessive-tab-switching (>3)
  participants removed.
- "clean_long": the same 342 participants, one row per participant-stimulus
  (dataLongClean.csv, 3420 rows).

The clean configs are what the paper analyses: 342 CUB + 363 emails = 705
participants, 3420 + 3539 = 6959 observations.

All three use dataComplete_CorrectLabelsCondition.csv's *public* condition
labels (NoSupport / LabelOnly / NonInteractiveConcepts / InteractiveConcepts),
matching NWeak/CBM-user-study-emails. The raw dataComplete.csv's internal
labels (None / BlackBox / FixedCBM / InteractiveCBM) are deliberately not
published -- an earlier version of this script read that file by mistake.

Every config gains a TestSampleIdx column derived from the stimulus filename
("1_003531_Final2.png" -> 3531), which is the sample_idx into the CUB test
split -- see NWeak/cub-mirror. Wide configs get one per stimulus slot
(StimX_TestSampleIdx, inserted right after StimX_StimID); the long config gets
a single TestSampleIdx after StimID.

Requires being logged in first: `uv run huggingface-cli login` (or set HF_TOKEN).

Example:
  cd cub && uv run scripts/push_user_study_to_hub.py
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import DatasetCard

N_STIMULI = 10

CONFIGS = {
    "full": "dataComplete_CorrectLabelsCondition.csv",
    "clean_wide": "dataWideClean.csv",
    "clean_long": "dataLongClean.csv",
}

# Body text only -- the YAML frontmatter (including the `configs:` block that
# maps each config_name to its uploaded file paths) is managed automatically by
# Dataset.push_to_hub() on each push below, and must be preserved rather than
# overwritten by a hand-written duplicate that could drift from the real layout.
CARD_BODY_TEMPLATE = """# {repo_id}

Participant responses from the CUB concept-bottleneck user study (Le Conte's
Sparrow vs. Savannah Sparrow), 10 stimuli each plus attention checks and a
trust questionnaire. Three configs:

- **`full`** ({n_full} participants, one row each): every participant who
  started the study, as recorded.
- **`clean_wide`** ({n_clean_wide} participants, one row each): `full` filtered
  to remove participants who were (a) blank/incomplete, (b) failed
  `AttentionCheck1`/`AttentionCheck2`, or (c) switched away from the browser tab
  more than 3 times total across the 10 stimuli (a tab-switching risk
  heuristic, e.g. possible use of an external tool to answer).
- **`clean_long`** ({n_clean_long} rows, one row per participant-stimulus): the
  same {n_clean_wide} participants as `clean_wide`, melted to long format.
  `TimeSpent` is in minutes; `Confidence_Adjusted = abs(Confidence - 7)` folds
  the 1-13 confidence scale into a direction-independent 0-6 measure.

**The paper analyses the clean configs.** Together with
[`NWeak/CBM-user-study-emails`](https://huggingface.co/datasets/NWeak/CBM-user-study-emails)'s
`clean_wide`/`clean_long` they give the reported totals of 705 participants and
6,959 observations.

## Columns

- `ExperimentalCondition`: `NoSupport`, `LabelOnly`, `NonInteractiveConcepts`,
  or `InteractiveConcepts` (the same four arms, under the same names, as the
  emails study).
- `StimID` / `StimX_StimID` (X = 1..10): the stimulus filename shown.
- `TestSampleIdx` / `StimX_TestSampleIdx`: the `sample_idx` into the CUB test
  split. Join against
  [`NWeak/cub-mirror`](https://huggingface.co/datasets/NWeak/cub-mirror)'s test
  split to recover the image, concepts, and label behind any stimulus. `-1`
  means no stimulus was shown at that position.
- `FeatureN_GT` / `FeatureN_Detected` / `FeatureN_Clicks`: per-concept ground
  truth, model activation, and how many times the participant clicked it.

```python
from datasets import load_dataset

full  = load_dataset("{repo_id}", "full",       split="test")
wide  = load_dataset("{repo_id}", "clean_wide", split="test")
long_ = load_dataset("{repo_id}", "clean_long", split="test")
```

Participant IDs are sequential integers; no identifying information is present.

## Ownership and licence

**This dataset is our own work, (c) 2026 the authors, released under
CC-BY-4.0.** That covers every participant response, our model's outputs, the
experimental design, *and* the concept ground truth.

The six concepts are **not** CUB's official attributes. We defined them for
this study (warm-coloured eyebrow, warm-coloured chest, plain sides, crested
head, white throat, striped chest) and annotated the stimuli ourselves;
`FeatureN_GT` reproduces CUB's own 112 attributes in 0 of 3420 rows, because it
is not derived from them.

CUB-200-2011 contributes only two things here: the stimulus images, which are
*referenced* by `StimID`/`TestSampleIdx` but not included, and the species
label in `CorrectAnswer`. For the images and CUB's official annotations see
[`NWeak/cub-mirror`](https://huggingface.co/datasets/NWeak/cub-mirror), which
stays under CUB's non-commercial research-use terms (Wah et al., 2011) because
we do not own them.

## Citation

Produced for **[Are Concept Bottleneck Models Effective as Decision-Support
Systems?](https://arxiv.org/abs/2608.25581)** (arXiv:2608.25581) -- Bogani,
Debole, Marconato, Pugnana, Tentori, Passerini. Code:
[github.com/debryu/user-study-CBMs](https://github.com/debryu/user-study-CBMs).
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="temp", help="Directory holding the official Drive exports")
    parser.add_argument("--repo-id", default="NWeak/CBM-user-study-cub")
    parser.add_argument("--public", action="store_true", help="Push as a public repo (default: private)")
    parser.add_argument("--dry-run", action="store_true", help="Build and report the tables without pushing")
    return parser.parse_args()


def extract_id(value) -> int:
    """'1_003531_Final2.png' -> 3531; blank/missing -> -1."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return -1
    text = str(value).strip()
    if not text:
        return -1
    return int(text.split("_")[1])


def add_sample_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Insert TestSampleIdx column(s) right after the stimulus-filename column(s)."""
    if "StimID" in df.columns:  # long format: one stimulus per row
        df.insert(df.columns.get_loc("StimID") + 1, "TestSampleIdx", df["StimID"].map(extract_id))
        return df
    # Wide format: one slot per stimulus. Build every new column first and
    # reorder once, rather than calling df.insert() ten times (which fragments
    # the frame and makes pandas warn).
    new_cols = {f"Stim{s}_TestSampleIdx": df[f"Stim{s}_StimID"].map(extract_id) for s in range(1, N_STIMULI + 1)}
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    order = []
    for col in df.columns:
        if col in new_cols:
            continue
        order.append(col)
        idx_col = col.replace("_StimID", "_TestSampleIdx")
        if col.endswith("_StimID") and idx_col in new_cols:
            order.append(idx_col)
    return df[order]


def load_config(path: Path, config_name: str) -> pd.DataFrame:
    # Default NA handling (not keep_default_na=False): these exports already use
    # "NoSupport" rather than the literal string "None", so blank cells can
    # safely become real nulls -- which keeps numeric columns as int64/float64
    # on the Hub instead of coercing everything to string.
    df = pd.read_csv(path)

    if config_name == "full":
        started = df.drop(columns=["ID"]).notna().any(axis=1)
        if (~started).any():
            print(f"  dropped {(~started).sum()} row(s) with no data beyond ID: {sorted(df.loc[~started, 'ID'])}")
            df = df[started].reset_index(drop=True)

    return add_sample_idx(df)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    tables = {}
    for config_name, filename in CONFIGS.items():
        print(f"{config_name} <- {filename}")
        df = load_config(data_dir / filename, config_name)
        conditions = sorted(df["ExperimentalCondition"].dropna().unique())
        print(f"  {len(df)} rows, {df.shape[1]} cols, conditions={conditions}")
        tables[config_name] = df

    if args.dry_run:
        print("\n--dry-run: nothing pushed")
        return

    for config_name, df in tables.items():
        ds = Dataset.from_pandas(df, preserve_index=False)
        ds.push_to_hub(args.repo_id, config_name=config_name, split="test", private=not args.public)
        print(f"Pushed {len(df)} rows to {args.repo_id} (config={config_name})")

    # Load the card push_to_hub() just auto-generated (correct multi-config
    # YAML, built from what was actually uploaded) and only set license/tags
    # plus the body text, keeping the `configs:` metadata intact.
    card = DatasetCard.load(args.repo_id, repo_type="dataset")
    card.data.license = "cc-by-4.0"
    card.data["tags"] = ["concept-bottleneck-models", "cub-200-2011", "user-study"]
    card.text = CARD_BODY_TEMPLATE.format(
        repo_id=args.repo_id,
        n_full=len(tables["full"]),
        n_clean_wide=len(tables["clean_wide"]),
        n_clean_long=len(tables["clean_long"]),
    )
    card.push_to_hub(args.repo_id, repo_type="dataset")
    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
