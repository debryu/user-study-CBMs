# user-study-CBMs

Reproducible codebase for the CBM user study paper, built around [CQA](https://github.com/debryu/CQA) for dataset loading and concept-bottleneck-model tooling.

The repo is organized by experiment (`cub/`, `emails/`, ...), each with its own `data/`, `metadata/`, `scripts/`, `notebooks/` — sharing a single top-level `uv` environment.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and an NVIDIA GPU (CUDA 12.1+ driver).

```bash
uv sync
```

This creates a `.venv` with a CUDA build of PyTorch, CQA (installed straight from its GitHub repo), and OpenAI's CLIP.

> CQA vendors its own copy of CLIP, but its packaging doesn't ship the BPE vocab file it needs, so it fails to import. We depend on the upstream [openai/CLIP](https://github.com/openai/CLIP) package instead for the actual encoding — CQA is still used for dataset loading (`CQA.datasets.GenericDataset`).

## Data

Datasets, CLIP embeddings, and model checkpoints are **not** committed to this repo — they're distributed via HuggingFace instead (link TBD). Every `data/`, `clip_embeddings/`, and `*_csv/` folder (wherever it occurs, e.g. `cub/data/`) is gitignored.

To populate the CUB-200-2011 dataset locally, place it under `cub/data/cub/` in the layout CQA expects:

```
cub/data/cub/
├── CUB_200_2011/          # raw images (from Caltech)
└── class_attr_data_10/    # train.pkl / val.pkl / test.pkl (concept + label annotations)
```

If `cub/data/cub/` is empty, pass `--download` to the encoding script below and CQA will fetch both automatically.

Class and concept names live in `cub/metadata/cub/{classes.txt,concepts.txt}` (tracked in git — small, static reference lists).

## Encoding a dataset with CLIP

```bash
cd cub
uv run scripts/encode_clip.py --dataset cub --clip-model ViT-L/14
```

For each split (`train`, `val`, `test`) this:
- encodes every image with the given CLIP model and saves the embedding matrix to `data/clip_embeddings/<dataset>_<split>_<model>.pt`
- saves a CSV of concepts + labels to `data/<dataset>_csv/<split>.csv` (e.g. `data/cub_csv/train.csv`)

Run `uv run scripts/encode_clip.py --help` for all options (batch size, device, output directories, etc). The output CSV also gets an `image_path` column (relative to `--data-root`) so images can be re-joined for distribution — see below.

## Publishing to HuggingFace

Once a dataset is fully encoded, `push_to_hub.py` joins each split's images + embeddings + CSV into one table (image, embedding, label, class_name, one column per concept) and pushes it as a HF `DatasetDict`:

```bash
uv run huggingface-cli login   # or export HF_TOKEN=...
cd cub
uv run scripts/push_to_hub.py --repo-id <your-username>/cub-user-study
```

Pushes as a **private** repo by default; pass `--public` once it's ready to share. Run `--help` for all options.

## Repository layout

```
cub/
├── scripts/
│   ├── encode_clip.py            # CLIP encoding script
│   └── push_to_hub.py            # packages + pushes a dataset to HF Hub
├── metadata/cub/                 # class/concept name lists (tracked)
├── notebooks/generate_data.ipynb # exploratory reference notebook
└── data/                          # dataset + generated artifacts (gitignored)
    ├── cub/                       # raw dataset
    ├── clip_embeddings/           # CLIP embeddings
    └── cub_csv/                   # concepts/labels CSVs
emails/                            # second experiment (TBD)
pyproject.toml, uv.lock            # shared environment for all experiments
```
