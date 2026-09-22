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
we don't own the images. Our own contributions are published separately under
a license we choose: the participant responses live in NWeak/CBM-user-study-cub,
joinable back to this repo's test split on sample_idx (see its TestSampleIdx
columns). The image_path column is kept as a stable per-image join key.

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

    # Keep image_path (stable per-image join key for downstream datasets) and the
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
        "Our own contributions (not part of the original CUB-200-2011 release) are "
        "published separately under CC-BY-4.0: the user-study participant responses "
        "at [`NWeak/CBM-user-study-cub`](https://huggingface.co/datasets/NWeak/CBM-user-study-cub), "
        "whose `TestSampleIdx` columns join against this repo's test-split `sample_idx`."
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

## Citation

Produced for **[Are Concept Bottleneck Models Effective as Decision-Support
Systems?](https://arxiv.org/abs/2608.25581)** (arXiv:2608.25581) -- Bogani,
Debole, Marconato, Pugnana, Tentori, Passerini. Code:
[github.com/debryu/user-study-CBMs](https://github.com/debryu/user-study-CBMs).
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
