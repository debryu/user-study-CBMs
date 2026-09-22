"""Push our model's computed activations (emails_user_study.csv, produced by
train_concept_extractor2.ipynb) to the HF Hub as a private dataset.

This is our own contribution (predicted concept activations, task predictions),
not raw email data -- kept separate from emails-mirror. Pushed under split
"test" (Dataset.push_to_hub()'s default "train" would be misleading, since
this is the held-out user-study set).

Requires being logged in first: `uv run huggingface-cli login` (or set HF_TOKEN).

Example:
  cd emails && uv run scripts/push_user_study_to_hub.py
"""

import argparse

import pandas as pd
from datasets import Dataset
from huggingface_hub import DatasetCard

CARD_TEMPLATE = """---
license: cc-by-4.0
tags:
- concept-bottleneck-models
- phishing-detection
- nlp
---

# {repo_id}

Our model's computed results for the 1000-email user-study set (see
`emails/scripts/train_concept_extractor2.ipynb` in
[user-study-CBMs](https://github.com/debryu/user-study-CBMs)): per email,
the 6 ground-truth concepts (`c1..6_gt`), the 6 predicted concept activations
(`c1..6_pred`, continuous -- tanh of the concept-SVM decision function), the
predicted task label (`task_pred`), and the ground-truth label (`task_gt`).

Concept order: Fear+Authority, Urgency, Curiosity, Neutral, Reply, Open attachment.

Also includes `Subject`/`Body`/`Sender` for readability. Join on
`Original email No.`.

## Source dataset and attribution

The emails themselves are **not ours**. They come from the
**PhishingSpamDataSet** of Toth, Bisztray and Dubniczky, released under
CC-BY-4.0 and redistributed here under that licence:

> Rebeka Toth, Tamas Bisztray, Richard A. Dubniczky.
> *Constructing and Benchmarking: a Labeled Email Dataset for Text-Based
> Phishing and Spam Detection Framework.*
> [arXiv:2511.21448](https://arxiv.org/abs/2511.21448) |
> [github.com/DataPhish/PhishingSpamDataSet](https://github.com/DataPhish/PhishingSpamDataSet)

Theirs: `Subject`, `Body`, `Sender`, `URL(s)`, `Type`, `Created by`, `Source`,
`Year`, and the `LLM detected emotion` / `LLM detected motivation`
annotations. Ours: the filtering and split, the embeddings, the 6-concept
merge, and the trained models' outputs.

Please cite Toth et al. alongside our paper if you use this data.

**Note:** `Sender` and `Body` are real email content carried over from the
source dataset.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/user_study/emails_user_study.csv")
    parser.add_argument("--repo-id", default="NWeak/emails-user-study")
    parser.add_argument("--public", action="store_true", help="Push as a public repo (default: private)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(args.repo_id, split="test", private=not args.public)

    card = DatasetCard(CARD_TEMPLATE.format(repo_id=args.repo_id))
    card.push_to_hub(args.repo_id, repo_type="dataset")
    print(f"Pushed {len(df)} rows to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
