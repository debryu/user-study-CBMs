# CUB preprocessing (before CLIP encoding)

Companion to [`emails/emails_preprocessing.md`](../emails/emails_preprocessing.md).
Describes exactly what happens to the CUB-200-2011 data before it's encoded
by CLIP in `cub/scripts/encode_clip.py`.

## Short answer: there is no preprocessing/filtering

Unlike the emails pipeline, **every image in the official split gets
encoded, unmodified in selection** — no filtering, no exclusion, no
subsetting. The only "preprocessing" is CLIP's own standard image
transform, applied identically to every image.

## Data source

- Raw images: `CUB_200_2011.tgz` (Caltech CUB-200-2011).
- Concept/label annotations: three pickle files,
  `<root>/class_attr_data_10/{train,val,test}.pkl`, from the official CBM
  (Concept Bottleneck Models) CodaLab data bundle — these are the standard
  200-class, 312-concept CUB splits used throughout the CBM literature, not
  something re-split or re-curated by this repo.

## Loading (no filtering happens here)

`encode_clip.py` loads each split via CQA's `GenericDataset`
(`from CQA.datasets import GenericDataset`), which for `ds_name="cub"`
resolves to `CQA.datasets.dataset_classes.CUBDataset`. For a given split,
this class loads **the entire corresponding pkl file** in one shot:

```python
self.data.extend(pickle.load(open(self.pkl_file_paths[split], 'rb')))
```

No sampling, slicing, deduplication, or class exclusion happens anywhere in
this path. `__len__` returns the full pkl entry count; `__getitem__` opens
every image (`Image.open(img_path).convert('RGB')`) unconditionally.
`GenericDataset` itself adds nothing beyond this — it's a pass-through that
also computes summary stats (concept/label frequencies, class weights) for
`dataset_info.json`, which are informational only and not used to filter
anything.

## Image transform (CLIP's own preprocessing, not custom)

The transform handed to the dataset loader comes directly from
`clip.load()`:

```python
model, preprocess = clip.load(args.clip_model, device=args.device)
```

which returns:

```python
Compose([
    Resize(n_px, interpolation=BICUBIC),
    CenterCrop(n_px),
    convert_to_rgb,
    ToTensor(),
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
              std=(0.26862954, 0.26130258, 0.27577711)),
])
```

where `n_px = model.visual.input_resolution` (e.g. 224 for `ViT-L/14`).
This is entirely CLIP's own standard preprocessing — the same transform
anyone using OpenAI's CLIP package gets by default — applied identically to
every image, in every split, with no dataset-specific variation.

## Model

`--clip-model` (default `ViT-L/14`), any of `clip.available_models()`:
`RN50`, `RN101`, `RN50x4`, `RN50x16`, `RN50x64`, `ViT-B/32`, `ViT-B/16`,
`ViT-L/14`, `ViT-L/14@336px`.

## Output

For each of `train`/`val`/`test`:

- **Embeddings**: `data/clip_embeddings/cub_<split>_<model_tag>.pt` — a
  single stacked tensor (`model.encode_image(...)`, `.float().cpu()`) over
  every image in the split, in loading order (`/` in the model name becomes
  `%` in the filename, e.g. `ViT-L%14`).
- **CSV**: `data/cub_csv/<split>.csv` — one row per image, in the same
  order as the embeddings tensor: `sample_idx`, `split`, `label`,
  `class_name` (via `metadata/cub/classes.txt`), `image_path` (resolved
  relative to `--data-root`, always present for CUB), plus one column per
  concept (via `metadata/cub/concepts.txt`).

## What happens later is *not* part of this preprocessing

Downstream analyses (e.g. the Le Conte vs. Savannah Sparrow minimal-pair
experiment in `cub/notebooks/generate_data.ipynb`) subset the
*already-encoded* embedding/concept tensors in memory (by class id and
concept index) purely for training a specific downstream classifier. No
image is ever excluded from encoding itself — `encode_clip.py` always
produces embeddings for the full official train/val/test splits, and any
class- or concept-level selection happens strictly afterward, at
analysis/training time.
