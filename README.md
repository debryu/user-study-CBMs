# user-study-CBMs

Reproducible codebase for the CBM user study paper, built around [CQA](https://github.com/debryu/CQA) for dataset loading and concept-bottleneck-model tooling.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and an NVIDIA GPU (CUDA 12.1+ driver).

```bash
uv sync
```

This creates a `.venv` with a CUDA build of PyTorch, CQA (installed straight from its GitHub repo), and OpenAI's CLIP.

> CQA vendors its own copy of CLIP, but its packaging doesn't ship the BPE vocab file it needs, so it fails to import. We depend on the upstream [openai/CLIP](https://github.com/openai/CLIP) package instead for the actual encoding — CQA is still used for dataset loading (`CQA.datasets.GenericDataset`).

## Data

Datasets, CLIP embeddings, and model checkpoints are **not** committed to this repo — they're distributed via HuggingFace instead (link TBD). `data/`, `clip_embeddings/`, and `*_csv/` are gitignored.

To populate a dataset locally, place it under `data/<dataset>/` in the layout CQA expects. For CUB-200-2011:

```
data/cub/
├── CUB_200_2011/          # raw images (from Caltech)
└── class_attr_data_10/    # train.pkl / val.pkl / test.pkl (concept + label annotations)
```

If `data/cub/` is empty, pass `--download` to the encoding script below and CQA will fetch both automatically.

Class and concept names live in `metadata/cub/{classes.txt,concepts.txt}` (tracked in git — small, static reference lists).

## Encoding a dataset with CLIP

```bash
uv run scripts/encode_clip.py --dataset cub --data-root data/cub --clip-model ViT-L/14
```

For each split (`train`, `val`, `test`) this:
- encodes every image with the given CLIP model and saves the embedding matrix to `clip_embeddings/<dataset>_<split>_<model>.pt`
- saves a CSV of concepts + labels to `<dataset>_csv/<split>.csv` (e.g. `cub_csv/train.csv`)

Run `uv run scripts/encode_clip.py --help` for all options (batch size, device, output directories, etc).

## Repository layout

```
scripts/encode_clip.py     # CLIP encoding script
metadata/cub/               # class/concept name lists (tracked)
notebooks/generate_data.ipynb  # exploratory reference notebook
data/                        # datasets (gitignored, populated locally)
clip_embeddings/             # CLIP embeddings (gitignored)
cub_csv/                     # concepts/labels CSVs (gitignored)
```
