"""Push the cleaned CUB user-study responses to the HF Hub as a private dataset.

Loads the CSV produced by prepare_user_study_dataset.py and pushes it as a
single table under the "test" split name -- every stimulus a participant saw
came from the CUB test split (see StimX_TestSampleIdx), so calling it "train"
(push_to_hub's default for a single Dataset) would be misleading.

Requires being logged in first: `uv run huggingface-cli login` (or set HF_TOKEN).

Example:
  cd cub && uv run scripts/push_user_study_to_hub.py
"""

import argparse

import pandas as pd
from datasets import Dataset
from huggingface_hub import DatasetCard

CARD_TEMPLATE = """---
license: cc-by-4.0
tags:
- concept-bottleneck-models
- cub-200-2011
- user-study
---

# {repo_id}

Participant responses from the CUB concept-bottleneck user study (568 participants, 10 stimuli each, plus attention checks and a trust questionnaire).

- `ExperimentalCondition`: `NoSupport`, `BlackBox`, `FixedCBM`, or `InteractiveCBM`.
- `StimX_StimID` (X = 1..10): the raw stimulus filename shown to the participant.
- `StimX_TestSampleIdx`: the `sample_idx` into the CUB test split -- join against
  [`NWeak/cub-mirror`](https://huggingface.co/datasets/NWeak/cub-mirror)'s test
  split (or `cub/data/cub_csv/test.csv`) to recover the image/concepts/label
  shown for that stimulus. `-1` means no stimulus was shown at that position.

This is our own contribution (not part of the original CUB-200-2011 release), so it's licensed independently of CUB's terms -- see [`NWeak/cub-mirror`](https://huggingface.co/datasets/NWeak/cub-mirror) for the image/label data itself.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="temp/updated_complete.csv")
    parser.add_argument("--repo-id", default="NWeak/CBM-user-study-cub")
    parser.add_argument("--public", action="store_true", help="Push as a public repo (default: private)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Default NA handling (not keep_default_na=False): by this point
    # prepare_user_study_dataset.py has already relabeled the literal "None"
    # string to "NoSupport", so there's no more ambiguity between it and a
    # genuinely blank cell -- reading normally lets numeric columns come back
    # as proper int/float (with real nulls) instead of all-string, so the
    # pushed dataset needs no dtype fixups once downloaded.
    df = pd.read_csv(args.input)

    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(args.repo_id, split="test", private=not args.public)

    card = DatasetCard(CARD_TEMPLATE.format(repo_id=args.repo_id))
    card.push_to_hub(args.repo_id, repo_type="dataset")
    print(f"Pushed {len(df)} rows to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
