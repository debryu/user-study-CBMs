"""Encode a CQA dataset's images with CLIP and dump embeddings + concepts/labels.

For every split (train/val/test) this:
  1. loads the dataset via CQA's GenericDataset, applying CLIP's own preprocessing
  2. encodes every image with a CLIP vision encoder
  3. saves the resulting embedding matrix to <embeddings-dir>/<dataset>_<split>_<model>.pt
  4. saves a concepts/labels CSV to <csv-dir>/<split>.csv

Run from inside the experiment folder (e.g. cub/), where data/<dataset>,
metadata/<dataset>, data/clip_embeddings, and data/<dataset>_csv live.

Example:
  cd cub && uv run scripts/encode_clip.py --dataset cub
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
    parser.add_argument("--data-root", default=None, help="Dataset root directory (default: data/<dataset>)")
    parser.add_argument("--metadata-dir", default=None, help="Directory with classes.txt/concepts.txt (default: metadata/<dataset>)")
    parser.add_argument("--clip-model", default="ViT-L/14", choices=clip.available_models())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--download", action="store_true", help="Download the dataset into --data-root if missing")
    parser.add_argument("--embeddings-dir", default=None, help="Default: data/clip_embeddings")
    parser.add_argument("--csv-dir", default=None, help="Default: data/<dataset>_csv")
    return parser.parse_args()


def load_names(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def get_image_paths(dataset: GenericDataset) -> list[str] | None:
    """Image paths relative to --data-root, in the same order as dataset[idx].

    Only implemented for CQA's CUBDataset (the only backend we've inspected);
    returns None for datasets we don't know how to resolve paths for.
    """
    underlying = dataset.dataset
    if not (hasattr(underlying, "data") and hasattr(underlying, "image_dir")):
        return None
    root = Path(dataset.root)
    image_dir = Path(underlying.image_dir)
    paths = []
    for entry in underlying.data:
        parts = entry["img_path"].split("/")
        marker = parts.index("CUB_200_2011")
        local_path = image_dir.joinpath(*parts[marker + 2 :])
        paths.append(str(local_path.relative_to(root)))
    return paths


def encode_split(model, preprocess, args, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str] | None]:
    dataset = GenericDataset(
        ds_name=args.dataset,
        split=split,
        root=args.data_root,
        transform=preprocess,
        download=args.download,
    )
    image_paths = get_image_paths(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    embeddings, concepts, labels = [], [], []
    with torch.no_grad():
        for images, batch_concepts, batch_labels in tqdm(loader, desc=f"{args.dataset}/{split}"):
            features = model.encode_image(images.to(args.device))
            embeddings.append(features.float().cpu())
            concepts.append(batch_concepts)
            labels.append(batch_labels)

    return torch.cat(embeddings), torch.cat(concepts), torch.cat(labels), image_paths


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root or str(Path("data") / args.dataset)
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else Path("metadata") / args.dataset
    csv_dir = Path(args.csv_dir) if args.csv_dir else Path("data") / f"{args.dataset}_csv"
    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else Path("data") / "clip_embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_names(metadata_dir / "classes.txt")
    concept_names = load_names(metadata_dir / "concepts.txt")

    model, preprocess = clip.load(args.clip_model, device=args.device)
    model.eval()
    model_tag = args.clip_model.replace("/", "%")

    for split in SPLITS:
        embeddings, concepts, labels, image_paths = encode_split(model, preprocess, args, split)

        embeddings_path = embeddings_dir / f"{args.dataset}_{split}_{model_tag}.pt"
        torch.save(embeddings, embeddings_path)

        labels_list = labels.tolist()
        columns = {
            "sample_idx": range(len(labels_list)),
            "split": split,
            "label": labels_list,
            "class_name": [class_names[label] for label in labels_list],
        }
        if image_paths is not None:
            columns["image_path"] = image_paths
        df = pd.DataFrame(columns)
        concepts_df = pd.DataFrame(concepts.numpy(), columns=concept_names)
        df = pd.concat([df, concepts_df], axis=1)
        df.to_csv(csv_dir / f"{split}.csv", index=False)

        print(f"{split}: {embeddings.shape[0]} samples -> {embeddings_path} ({embeddings.shape[1]}-d), {csv_dir / f'{split}.csv'}")


if __name__ == "__main__":
    main()
