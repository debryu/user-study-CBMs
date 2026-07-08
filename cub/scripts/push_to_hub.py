"""Push the CUB-200-2011 image + CLIP embedding + official-label mirror to the HF Hub.

Joins, for each split:
  - data/<dataset>_csv/<split>.csv                     (label, class_name, concept columns, image_path)
  - data/clip_embeddings/<dataset>_<split>_<model>.pt   (CLIP embedding per row)
  - the actual image files under --data-root (referenced by image_path)
into one table per split: image, image_path, embedding, label (a ClassLabel,
so class names are stored in the dataset schema), class_name, one 0/1 column
per (original CUB) concept, AND a `concepts` column holding the same 112
values as a single ground-truth vector (same order as --metadata-dir's
concepts.txt, which is also uploaded to the repo as concept_names.txt,
alongside class_names.txt, so both name lists are directly retrievable).

This repo only contains images and the *original* CUB-200-2011 annotations
(class labels + the 112 official concept attributes) plus our CLIP encoding of
them, so it's published under CUB-200-2011's own non-commercial research-use
terms (see build_card() below) -- NOT a license we get to choose freely, since
we don't own the images. Any new annotations that are our own contribution
belong in a separate, separately-licensed repo -- see push_annotations_to_hub.py.
Both repos share the `image_path` column so they can be joined back together.

Requires being logged in first: `uv run huggingface-cli login` (or set HF_TOKEN).

Example:
  cd cub && uv run scripts/push_to_hub.py --repo-id debryu/cub-mirror
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Image
from huggingface_hub import DatasetCard, HfApi

SPLITS = ["train", "val", "test"]

CUB_CITATION = """@techreport{WahCUB_200_2011,
    Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
    Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
    Year = {2011},
    Institution = {California Institute of Technology},
    Number = {CNS-TR-2011-001}
}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cub", help="Dataset name used in encode_clip.py's output filenames")
    parser.add_argument("--repo-id", required=True, help="HF dataset repo, e.g. debryu/cub-mirror")
    parser.add_argument("--annotations-repo-id", default=None, help="Paired repo with our own annotations, for the dataset card")
    parser.add_argument("--metadata-dir", default=None, help="Directory with classes.txt/concepts.txt (default: metadata/<dataset>)")
    parser.add_argument("--data-root", default=None, help="Dataset root that image_path is relative to (default: data/<dataset>)")
    parser.add_argument("--csv-dir", default=None, help="Default: data/<dataset>_csv")
    parser.add_argument("--embeddings-dir", default=None, help="Default: data/clip_embeddings")
    parser.add_argument("--clip-model", default="ViT-L/14", help="Must match the model used in encode_clip.py")
    parser.add_argument("--public", action="store_true", help="Push as a public repo (default: private)")
    return parser.parse_args()


def load_names(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_split(args, split: str, class_names: list[str], concept_names: list[str]) -> Dataset:
    csv_dir = Path(args.csv_dir) if args.csv_dir else Path("data") / f"{args.dataset}_csv"
    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else Path("data") / "clip_embeddings"
    data_root = Path(args.data_root) if args.data_root else Path("data") / args.dataset
    model_tag = args.clip_model.replace("/", "%")

    df = pd.read_csv(csv_dir / f"{split}.csv")
    embeddings = torch.load(embeddings_dir / f"{args.dataset}_{split}_{model_tag}.pt", weights_only=True)
    assert len(df) == embeddings.shape[0], f"{split}: {len(df)} csv rows vs {embeddings.shape[0]} embeddings"

    # Keep image_path (stable join key for push_annotations_to_hub.py) and the
    # individual concept columns (browsable in the Hub viewer) alongside the
    # decoded image feature and a single concepts ground-truth vector (same
    # order as concept_names, handy to load straight into numpy/torch).
    extra = pd.DataFrame(
        {
            "image": [str(data_root / p) for p in df["image_path"]],
            "embedding": embeddings.tolist(),
            "concepts": df[concept_names].to_numpy().tolist(),
        }
    )
    df = pd.concat([df, extra], axis=1)

    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.cast_column("image", Image())
    ds = ds.cast_column("label", ClassLabel(names=class_names))
    return ds


def build_card(args) -> str:
    annotations_note = (
        f"New annotations that are our own contribution (not part of the original "
        f"CUB-200-2011 release) are published separately, under a more permissive "
        f"license, at [`{args.annotations_repo_id}`](https://huggingface.co/datasets/{args.annotations_repo_id}). "
        f"Join on `image_path` to combine the two."
        if args.annotations_repo_id
        else "Any new annotations that are our own contribution (not part of the "
        "original CUB-200-2011 release) are published separately, under a more "
        "permissive license, in a paired repo joinable on `image_path`."
    )
    return f"""---
license: other
license_name: cub-200-2011-research-use
license_link: https://www.vision.caltech.edu/datasets/cub_200_2011/
tags:
- concept-bottleneck-models
- cub-200-2011
- clip
---

# {args.repo_id}

CLIP (`{args.clip_model}`) embeddings of [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)
images, joined with the dataset's official class labels and 112 concept
(attribute) annotations. Each row: `image`, `image_path`, `embedding`,
`label` (a `ClassLabel`, so `ds[split].features["label"].names` gives the 200
class names), `class_name`, one 0/1 column per concept, and a `concepts`
column with the same 112 values as a single ground-truth vector. The ordered
name lists are also available as plain files in this repo: `class_names.txt`
and `concept_names.txt` (the latter is the column order used by `concepts`).

## Attribution & license

Images and the original concept/class annotations are from CUB-200-2011
(Wah et al., 2011, Caltech-UCSD). The photographs were collected from the web
and remain the property of their original copyright holders; Caltech
distributes them for **non-commercial research and educational use only**.
This mirror carries the same restriction -- do not use for commercial
purposes, and cite the original dataset:

```bibtex
{CUB_CITATION}
```

{annotations_note}
"""


def main() -> None:
    args = parse_args()
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else Path("metadata") / args.dataset
    class_names = load_names(metadata_dir / "classes.txt")
    concept_names = load_names(metadata_dir / "concepts.txt")

    splits = {split: build_split(args, split, class_names, concept_names) for split in SPLITS}
    dataset_dict = DatasetDict(splits)
    dataset_dict.push_to_hub(args.repo_id, private=not args.public)

    api = HfApi()
    api.upload_file(
        path_or_fileobj="\n".join(class_names).encode(),
        path_in_repo="class_names.txt",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj="\n".join(concept_names).encode(),
        path_in_repo="concept_names.txt",
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    card = DatasetCard(build_card(args))
    card.push_to_hub(args.repo_id, repo_type="dataset")
    print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
