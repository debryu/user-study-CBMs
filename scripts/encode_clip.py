"""Encode a CQA dataset's images with CLIP and dump embeddings + concepts/labels.

For every split (train/val/test) this:
  1. loads the dataset via CQA's GenericDataset, applying CLIP's own preprocessing
  2. encodes every image with a CLIP vision encoder
  3. saves the resulting embedding matrix to <embeddings-dir>/<dataset>_<split>_<model>.pt
  4. saves a concepts/labels CSV to <csv-dir>/<split>.csv

Example:
  uv run scripts/encode_clip.py --dataset cub --data-root data/cub
"""

import argparse
from pathlib import Path

import clip
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from CQA.datasets import GenericDataset

SPLITS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cub", help="CQA dataset name (ds_name passed to GenericDataset)")
    parser.add_argument("--data-root", default="data/cub", help="Dataset root directory")
    parser.add_argument("--metadata-dir", default=None, help="Directory with classes.txt/concepts.txt (default: metadata/<dataset>)")
    parser.add_argument("--clip-model", default="ViT-L/14", choices=clip.available_models())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--download", action="store_true", help="Download the dataset into --data-root if missing")
    parser.add_argument("--embeddings-dir", default="clip_embeddings")
    parser.add_argument("--csv-dir", default=None, help="Default: <dataset>_csv")
    return parser.parse_args()


def load_names(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def encode_split(model, preprocess, args, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = GenericDataset(
        ds_name=args.dataset,
        split=split,
        root=args.data_root,
        transform=preprocess,
        download=args.download,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    embeddings, concepts, labels = [], [], []
    with torch.no_grad():
        for images, batch_concepts, batch_labels in tqdm(loader, desc=f"{args.dataset}/{split}"):
            features = model.encode_image(images.to(args.device))
            embeddings.append(features.float().cpu())
            concepts.append(batch_concepts)
            labels.append(batch_labels)

    return torch.cat(embeddings), torch.cat(concepts), torch.cat(labels)


def main() -> None:
    args = parse_args()
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else Path("metadata") / args.dataset
    csv_dir = Path(args.csv_dir) if args.csv_dir else Path(f"{args.dataset}_csv")
    embeddings_dir = Path(args.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_names(metadata_dir / "classes.txt")
    concept_names = load_names(metadata_dir / "concepts.txt")

    model, preprocess = clip.load(args.clip_model, device=args.device)
    model.eval()
    model_tag = args.clip_model.replace("/", "%")

    for split in SPLITS:
        embeddings, concepts, labels = encode_split(model, preprocess, args, split)

        embeddings_path = embeddings_dir / f"{args.dataset}_{split}_{model_tag}.pt"
        torch.save(embeddings, embeddings_path)

        labels_list = labels.tolist()
        df = pd.DataFrame(
            {
                "sample_idx": range(len(labels_list)),
                "split": split,
                "label": labels_list,
                "class_name": [class_names[label] for label in labels_list],
            }
        )
        concepts_df = pd.DataFrame(concepts.numpy(), columns=concept_names)
        df = pd.concat([df, concepts_df], axis=1)
        df.to_csv(csv_dir / f"{split}.csv", index=False)

        print(f"{split}: {embeddings.shape[0]} samples -> {embeddings_path} ({embeddings.shape[1]}-d), {csv_dir / f'{split}.csv'}")


if __name__ == "__main__":
    main()
