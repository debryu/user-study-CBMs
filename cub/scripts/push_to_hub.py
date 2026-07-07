"""Push the CUB-200-2011 image + CLIP embedding + official-label mirror to the HF Hub.

Joins, for each split:
  - data/<dataset>_csv/<split>.csv                     (label, class_name, concept columns, image_path)
  - data/clip_embeddings/<dataset>_<split>_<model>.pt   (CLIP embedding per row)
  - the actual image files under --data-root (referenced by image_path)
into one table per split: image, image_path, embedding, label, class_name, and
one column per (original CUB) concept.

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
from datasets import Dataset, DatasetDict, Image
from huggingface_hub import DatasetCard

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

    # Keep image_path (stable join key for push_annotations_to_hub.py) alongside
    # the decoded image feature.
    extra = pd.DataFrame(
        {
            "image": [str(data_root / p) for p in df["image_path"]],
            "embedding": embeddings.tolist(),
        }
    )
    df = pd.concat([df, extra], axis=1)

    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds.cast_column("image", Image())


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
`label`, `class_name`, and one 0/1 column per concept.

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
    splits = {split: build_split(args, split) for split in SPLITS}
    dataset_dict = DatasetDict(splits)
    dataset_dict.push_to_hub(args.repo_id, private=not args.public)

    card = DatasetCard(build_card(args))
    card.push_to_hub(args.repo_id, repo_type="dataset")
    print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
