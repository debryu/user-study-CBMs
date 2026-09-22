"""Push the emails User Study participant responses to the HF Hub as a
private dataset, mirroring cub/scripts/push_user_study_to_hub.py.

Pushes three configs into the *same* repo (verified via
verify_participant_data.py and the corresponding hard-assertion cells in
train_concept_extractor2.ipynb before publishing):
- "full": all 417 participants (dataComplete_CorrectLabelsCondition.csv).
- "clean_wide": 363 participants, one row per participant
  (dataWideClean.csv) -- blank/incomplete, failed-attention-check, and
  excessive-tab-switching (>3, tab-switch risk e.g. possible LLM use)
  participants removed.
- "clean_long": the same 363 participants, one row per participant-stimulus
  (dataLongClean.csv).

See emails/DATASET_NOTES.md and emails/PARTICIPANT_DATA_REPORT.md for the
full exclusion-criteria writeup and the 263/268 known-issue caveat.

Requires being logged in first: `uv run huggingface-cli login` (or set HF_TOKEN).

Example:
  cd emails && uv run scripts/push_participant_responses_to_hub.py
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from huggingface_hub import DatasetCard

CONFIGS = {
    "full": "dataComplete_CorrectLabelsCondition.csv",
    "clean_wide": "dataWideClean.csv",
    "clean_long": "dataLongClean.csv",
}

# Body text only -- the YAML frontmatter (including the `configs:` block
# that maps each config_name to its actual uploaded file paths) is managed
# automatically by Dataset.push_to_hub() on each of the three pushes below,
# and must be preserved, not overwritten by a hand-written duplicate here
# (which could drift from the real file layout and break config discovery).
CARD_BODY_TEMPLATE = """# {repo_id}

Participant responses from the emails concept-bottleneck user study. Three configs:

- **`full`** (417 participants, one row each): every participant, as
  originally recorded. `ExperimentalCondition`: `LabelOnly`,
  `NonInteractiveConcepts`, `InteractiveConcepts`, or `NoSupport`.
- **`clean_wide`** (363 participants, one row each): `full` filtered to
  remove participants who were (a) blank/incomplete, (b) failed
  `AttentionCheck1`/`AttentionCheck2`, or (c) switched away from the
  browser tab more than 3 times total across the 10 stimuli (a
  tab-switching risk heuristic, e.g. possible use of an external tool to
  answer). Kept participants' values are otherwise byte-identical to
  `full`, except stimulus 268's columns are blanked in place (slot position
  preserved) for participants who saw it -- see the 263/268 note below.
- **`clean_long`** (3539 rows, one row per participant-stimulus, main 10
  stimuli only): the same 363 participants as `clean_wide`, melted to long
  format. `TimeSpent` is in minutes (`raw_ms / 60000`);
  `Confidence_Adjusted = abs(Confidence - 7)` folds the 1-13 confidence
  scale into a direction-independent 0-6 measure. Stimulus 268 rows are
  fully absent here (not just blanked).

**Known issue (stimuli 263/268)**: due to a human error, the model
activations actually shown to participants for stimuli 263 and 268 did not
match the model's real output. 268 also flipped which concepts were active
(materially different information shown) -- all `full` observations for
268 are blanked out in `clean_wide`/`clean_long`. 263's error preserved the
same binary concept pattern (values differ, signs don't), so its
observations were kept; its GT is correct, only its recorded activations
are unreliable. Full root-cause writeup in this repo's `emails/DATASET_NOTES.md`
and `emails/PARTICIPANT_DATA_REPORT.md`.

This is our own contribution (participant responses, not the underlying
email stimuli), so it's licensed independently -- see
[`NWeak/emails-mirror`](https://huggingface.co/datasets/NWeak/emails-mirror)
for the email data itself and
[`NWeak/emails-user-study`](https://huggingface.co/datasets/NWeak/emails-user-study)
for our model's own computed activations/predictions (a different dataset
from this one -- that one is model output, this one is participant
responses).

## Citation

Produced for **[Are Concept Bottleneck Models Effective as Decision-Support
Systems?](https://arxiv.org/abs/2608.25581)** (arXiv:2608.25581) -- Bogani,
Debole, Marconato, Pugnana, Tentori, Passerini. Code:
[github.com/debryu/user-study-CBMs](https://github.com/debryu/user-study-CBMs).
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="temp/User Study")
    parser.add_argument("--repo-id", default="NWeak/CBM-user-study-emails")
    parser.add_argument("--public", action="store_true", help="Push as a public repo (default: private)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    for config_name, filename in CONFIGS.items():
        # Default NA handling: these files have already had the literal
        # "None" ExperimentalCondition value relabeled to "NoSupport", so
        # there's no more ambiguity with genuinely blank cells (same
        # rationale as cub/scripts/push_user_study_to_hub.py).
        df = pd.read_csv(data_dir / filename)
        ds = Dataset.from_pandas(df, preserve_index=False)
        ds.push_to_hub(args.repo_id, config_name=config_name, split="test", private=not args.public)
        print(f"Pushed {len(df)} rows to {args.repo_id} (config={config_name})")

    # Load the card push_to_hub() just auto-generated (correct multi-config
    # YAML, built from what was actually uploaded) and only add license/tags
    # plus replace the body text, keeping the `configs:` metadata intact.
    card = DatasetCard.load(args.repo_id, repo_type="dataset")
    card.data.license = "cc-by-4.0"
    card.data["tags"] = ["concept-bottleneck-models", "phishing-detection", "user-study"]
    card.text = CARD_BODY_TEMPLATE.format(repo_id=args.repo_id)
    card.push_to_hub(args.repo_id, repo_type="dataset")
    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
