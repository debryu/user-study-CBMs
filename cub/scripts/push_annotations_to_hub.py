"""Push our own, original annotations to their own HF dataset repo -- separate
from the CUB image/official-label mirror in push_to_hub.py.

Kept separate because these annotations are fully our contribution and can be
licensed however we want, independent of CUB-200-2011's non-commercial
research-use terms, which govern the images and official concept/class labels
published in the paired --base-repo-id dataset.

Put one CSV per split under --annotations-dir (default: data/<dataset>_annotations),
each containing an --id-column (default: image_path) matching the image_path
column in the base dataset, so the two can be joined later, e.g.:

    from datasets import load_dataset
    base = load_dataset("<base-repo-id>", split="train").to_pandas()
    extra = load_dataset("<repo-id>", split="train").to_pandas()
    joined = base.merge(extra, on="image_path")

Requires being logged in first: `uv run huggingface-cli login` (or set HF_TOKEN).

Example:
  cd cub && uv run scripts/push_annotations_to_hub.py \\
      --repo-id debryu/cub-user-study-annotations \\
      --base-repo-id debryu/cub-mirror \\
      --license cc-by-4.0
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import DatasetCard

SPLITS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cub")
    parser.add_argument("--repo-id", required=True, help="HF repo for the annotations-only dataset")
    parser.add_argument("--base-repo-id", default=None, help="The paired image/CUB-label repo, for the dataset card")
    parser.add_argument("--annotations-dir", default=None, help="Default: data/<dataset>_annotations, one <split>.csv each")
    parser.add_argument("--id-column", default="image_path", help="Column used to join back to the base dataset")
    parser.add_argument("--license", default="cc-by-4.0", help="SPDX license id for this (our own) annotation data")
    parser.add_argument("--public", action="store_true", help="Push as a public repo (default: private)")
    return parser.parse_args()


def build_split(args, split: str) -> Dataset:
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else Path("data") / f"{args.dataset}_annotations"
    csv_path = annotations_dir / f"{split}.csv"
    df = pd.read_csv(csv_path)
    if args.id_column not in df.columns:
        raise ValueError(f"{csv_path} has no '{args.id_column}' column to join on")
    return Dataset.from_pandas(df, preserve_index=False)


def build_card(args) -> str:
    base_note = (
        f"Join back to the images and official CUB labels at "
        f"[`{args.base_repo_id}`](https://huggingface.co/datasets/{args.base_repo_id}) on `{args.id_column}`."
        if args.base_repo_id
        else f"Join back to the paired image/CUB-label dataset on `{args.id_column}`."
    )
    return f"""---
license: {args.license}
tags:
- concept-bottleneck-models
- cub-200-2011
---

# {args.repo_id}

Annotations produced for this paper -- **not** part of the original
CUB-200-2011 release, and fully our own contribution. Contains no images or
official CUB labels, so it's licensed under `{args.license}` independently of
CUB-200-2011's non-commercial research-use terms.

{base_note}
"""


def main() -> None:
    args = parse_args()
    dataset_dict = DatasetDict({split: build_split(args, split) for split in SPLITS})
    dataset_dict.push_to_hub(args.repo_id, private=not args.public)

    card = DatasetCard(build_card(args))
    card.push_to_hub(args.repo_id, repo_type="dataset")
    print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
