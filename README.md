# user-study-CBMs

Reproducible codebase for the CBM user study paper, built around [CQA](https://github.com/debryu/CQA) for dataset loading and concept-bottleneck-model tooling.

The repo is organized by experiment (`cub/`, `emails/`, ...), each with its own `data/`, `metadata/`, `scripts/`, `notebooks/` — sharing a single top-level `uv` environment.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and an NVIDIA GPU (CUDA 12.1+ driver).

```bash
uv sync
```

This creates a `.venv` with a CUDA build of PyTorch, CQA (installed straight from its GitHub repo), and OpenAI's CLIP.

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

We don't own the CUB-200-2011 images (Caltech redistributes them for non-commercial research use only — the photos themselves stay copyright of the original photographers), so images and the *official* CUB labels are kept in a separate repo from anything that's genuinely our own contribution, licensed accordingly. Both scripts push a `DatasetDict` with all three splits in one call and share the `image_path` column so the two repos can be joined back together locally.

```bash
uv run huggingface-cli login   # or export HF_TOKEN=...
cd cub

# 1. images + CLIP embeddings + official CUB labels — CUB's non-commercial research-use terms
uv run scripts/push_to_hub.py --repo-id <your-username>/cub-mirror \
    --annotations-repo-id <your-username>/cub-user-study-annotations

# 2. our own annotations only (once they exist, under data/cub_annotations/{split}.csv) — license of our choosing
uv run scripts/push_annotations_to_hub.py --repo-id <your-username>/cub-user-study-annotations \
    --base-repo-id <your-username>/cub-mirror --license cc-by-4.0
```

Both push as **private** repos by default; pass `--public` once ready to share. Each script also writes a dataset card (`README.md` on the Hub) documenting attribution/license and how to join the two repos. Run either with `--help` for all options.

`push_to_hub.py`'s dataset (currently live at [`NWeak/cub-mirror`](https://huggingface.co/datasets/NWeak/cub-mirror)) stores everything needed to run the modeling notebook without any local files: `label` is a HF `ClassLabel` (so `ds[split].features["label"].names` gives the 200 class names), `concepts` is a single ground-truth vector column (same order as the 112 individual concept columns), and `class_names.txt`/`concept_names.txt` are also uploaded as plain files in the repo.

## Modeling notebook

`cub/notebooks/generate_data.ipynb` loads `NWeak/cub-mirror` directly via `load_dataset()` and runs the full concept-bottleneck pipeline for a hard pair (Le Conte vs. Savannah Sparrow): CLIP embedding → concept classifiers → concept → label classifier → end-to-end evaluation → a tidy per-sample CSV → a hand-checkable linear formula for the user study. No local CUB download or CQA needed to run it — just `uv sync` + `hf auth login`.

## Repository layout

```
cub/
├── scripts/
│   ├── encode_clip.py             # CLIP encoding script
│   ├── push_to_hub.py             # pushes images + embeddings + official CUB labels (CUB terms)
│   └── push_annotations_to_hub.py # pushes our own annotations only (our choice of license)
├── metadata/cub/                  # class/concept name lists (tracked)
├── notebooks/generate_data.ipynb  # modeling pipeline, sourced entirely from the HF dataset
└── data/                          # dataset + generated artifacts (gitignored)
    ├── cub/                       # raw dataset
    ├── clip_embeddings/           # CLIP embeddings
    ├── cub_csv/                   # concepts/labels CSVs
    ├── cub_annotations/           # our own annotations, keyed by image_path (once added)
    └── user_study/                # generate_data.ipynb's output CSV
emails/                            # second experiment (TBD)
pyproject.toml, uv.lock            # shared environment for all experiments
```
