"""Package encode_clip.py's output (images + CLIP embeddings + concepts/labels)
into a HuggingFace Dataset and push it to the Hub.

For each split, joins:
  - data/<dataset>_csv/<split>.csv        (label, class_name, concept columns, image_path)
  - data/clip_embeddings/<dataset>_<split>_<model>.pt   (CLIP embedding per row)
  - the actual image files under --data-root (referenced by image_path)
into one table per split, so a row has: image, embedding, label, class_name,
and one column per concept.

Requires being logged in first: `uv run huggingface-cli login` (or set HF_TOKEN).

Example:
  cd cub && uv run scripts/push_to_hub.py --repo-id debryu/cub-user-study
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Image

SPLITS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cub", help="Dataset name used in encode_clip.py's output filenames")
    parser.add_argument("--repo-id", required=True, help="HF dataset repo, e.g. debryu/cub-user-study")
    parser.add_argument("--data-root", default=None, help="Dataset root that image_path is relative to (default: data/<dataset>)")
    parser.add_argument("--csv-dir", default=None, help="Default: data/<dataset>_csv")
    parser.add_argument("--embeddings-dir", default=None, help="Default: data/clip_embeddings")
    parser.add_argument("--clip-model", default="ViT-L/14", help="Must match the model used in encode_clip.py")
    parser.add_argument("--public", action="store_true", help="Push as a public repo (default: private)")
    return parser.parse_args()


def build_split(args, split: str) -> Dataset:
    csv_dir = Path(args.csv_dir) if args.csv_dir else Path("data") / f"{args.dataset}_csv"
    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else Path("data") / "clip_embeddings"
    data_root = Path(args.data_root) if args.data_root else Path("data") / args.dataset
    model_tag = args.clip_model.replace("/", "%")

    df = pd.read_csv(csv_dir / f"{split}.csv")
    embeddings = torch.load(embeddings_dir / f"{args.dataset}_{split}_{model_tag}.pt", weights_only=True)
    assert len(df) == embeddings.shape[0], f"{split}: {len(df)} csv rows vs {embeddings.shape[0]} embeddings"

    extra = pd.DataFrame(
        {
            "image": [str(data_root / p) for p in df.pop("image_path")],
            "embedding": embeddings.tolist(),
        }
    )
    df = pd.concat([df, extra], axis=1)

    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds.cast_column("image", Image())


def main() -> None:
    args = parse_args()
    splits = {split: build_split(args, split) for split in SPLITS}
    dataset_dict = DatasetDict(splits)
    dataset_dict.push_to_hub(args.repo_id, private=not args.public)
    print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
