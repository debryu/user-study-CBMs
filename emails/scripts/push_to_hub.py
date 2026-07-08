"""Push the preprocessed emails dataset (train/val/user_study splits, with
sentence embeddings, concept ground truth, and labels) to the HF Hub as a
private dataset, mirroring cub/scripts/push_to_hub.py.

user_study_data.parquet is pushed as the "test" split, matching how
train_concept_extractor2.ipynb actually uses it.

Requires being logged in first: `uv run huggingface-cli login` (or set HF_TOKEN).

Example:
  cd emails && uv run scripts/push_to_hub.py --repo-id NWeak/emails-mirror
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import DatasetCard

SPLIT_FILES = {
    "train": "train.parquet",
    "val": "val.parquet",
    "test": "user_study_data.parquet",
}

CARD_TEMPLATE = """---
license: other
tags:
- concept-bottleneck-models
- phishing-detection
- nlp
---

# {repo_id}

Preprocessed phishing-email concept-bottleneck data (see `emails/scripts/preprocessing.ipynb`
in [user-study-CBMs](https://github.com/debryu/user-study-CBMs)): sentence
embeddings (`all-MiniLM-L6-v2`), the full 19-dim concept ground truth
(`concept_gts`) and the 6-dim merged/filtered concept ground truth used for
modeling (`concept_gts_f` -- Fear+Authority combined, plus Urgency, Curiosity,
Neutral, Reply, Open attachment), and the task label (`label`: 0=Phishing,
1=Valid, 2=Spam).

`test` is `user_study_data.parquet` -- the 1000-email class-balanced subset
used for the concept-bottleneck user study (same split `train_concept_extractor2.ipynb`
evaluates against).

**Note:** `Sender`/`Body` contain real sender addresses and email content
pulled from public mailing-list archives and phishing-sample datasets --
published private for this reason.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory with train/val/user_study_data parquet files")
    parser.add_argument("--repo-id", default="NWeak/emails-mirror")
    parser.add_argument("--public", action="store_true", help="Push as a public repo (default: private)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    splits = {}
    for split, filename in SPLIT_FILES.items():
        df = pd.read_parquet(data_dir / filename)
        # words_bin is a leftover stratified-split helper column (train/val only)
        df = df.drop(columns=["words_bin"], errors="ignore")
        splits[split] = Dataset.from_pandas(df, preserve_index=False)
        print(f"{split}: {len(df)} rows")

    dataset_dict = DatasetDict(splits)
    dataset_dict.push_to_hub(args.repo_id, private=not args.public)

    card = DatasetCard(CARD_TEMPLATE.format(repo_id=args.repo_id))
    card.push_to_hub(args.repo_id, repo_type="dataset")
    print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
