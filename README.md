# Are Concept Bottleneck Models Effective as Decision-Support Systems?

Official repository for the paper
**[Are Concept Bottleneck Models Effective as Decision-Support Systems?](https://arxiv.org/abs/2608.25581)**
(arXiv:2608.25581).

Alessandro Bogani, Nicola Debole, Emanuele Marconato, Andrea Pugnana,
Katya Tentori, Andrea Passerini

> We present two large-scale user studies (N participants = 705,
> N observations = 6,959) evaluating how concept-based explanations and user
> interventions on the model's concepts affect the performance of the human-AI
> team in two distinct binary classification tasks.

This repo contains the code for both experiments (**CUB**, bird species
identification, and **emails**, phishing detection) and the links to every
dataset they produced, including the human participant responses.

## Datasets

All five datasets live on the HuggingFace Hub.

| Dataset | What it is | Size | License |
|---|---|---|---|
| [`NWeak/CBM-user-study-cub`](https://huggingface.co/datasets/NWeak/CBM-user-study-cub) | Human participant responses, CUB study | 568 participants | CC-BY-4.0 |
| [`NWeak/CBM-user-study-emails`](https://huggingface.co/datasets/NWeak/CBM-user-study-emails) | Human participant responses, emails study | 417 / 363 / 3,539 rows (3 configs) | CC-BY-4.0 |
| [`NWeak/cub-mirror`](https://huggingface.co/datasets/NWeak/cub-mirror) | CUB images + CLIP embeddings + official CUB labels & concepts | 4,796 / 1,198 / 5,794 | CUB research-use |
| [`NWeak/emails-mirror`](https://huggingface.co/datasets/NWeak/emails-mirror) | Email corpus + sentence embeddings + concept/label ground truth | 1,064 / 266 / 1,000 | CC-BY-4.0 (see below) |
| [`NWeak/emails-user-study`](https://huggingface.co/datasets/NWeak/emails-user-study) | Model concept activations + predictions for the 1,000 study emails | 1,000 | CC-BY-4.0 |

CUB is split across two repos because we don't own the CUB-200-2011 images:
`cub-mirror` carries only CUB-derived content under Caltech's non-commercial
research-use terms (citing Wah et al. 2011), while our own contribution, the
participant responses, sits in a separate CC-BY-4.0 repo, joined back on
`sample_idx`.

Neither email dataset is an original corpus. Both derive from the
**PhishingSpamDataSet** of Toth, Bisztray and Dubniczky, redistributed under
its CC-BY-4.0 licence. See [Acknowledgments](#acknowledgments) for what is
theirs and what is ours.

## Installation

Requires [`uv`](https://docs.astral.sh/uv/). A GPU helps but is not required to
run the tutorials.

```bash
git clone git@github.com:debryu/user-study-CBMs.git
cd user-study-CBMs
uv sync
uv run huggingface-cli login
```

`uv sync` creates a `.venv` with a CUDA build of PyTorch, CQA, and OpenAI's
CLIP. No dataset download is needed, since everything is pulled from the Hub.

## Using the datasets

```python
from datasets import load_dataset

cub = load_dataset("NWeak/CBM-user-study-cub", split="test")

# the emails study ships three curations of the same data
emails       = load_dataset("NWeak/CBM-user-study-emails", "full",       split="test")  # 417
emails_clean = load_dataset("NWeak/CBM-user-study-emails", "clean_wide", split="test")  # 363
emails_long  = load_dataset("NWeak/CBM-user-study-emails", "clean_long", split="test")  # 3,539
```

One row per participant (or, for `clean_long`, one row per
participant-stimulus). For each of 10 stimuli it records the stimulus shown,
the correct answer, the model's answer, the participant's answer and
confidence, per-concept ground truth and model activations, per-concept click
counts, time spent, number of answer changes, and time spent outside the
browser tab. Participant IDs are sequential integers, so no identifying
information is present.

**Condition names differ between the two published studies.**
`CBM-user-study-cub` uses the internal names (`NoSupport`, `BlackBox`,
`FixedCBM`, `InteractiveCBM`); `CBM-user-study-emails` uses the public ones
(`NoSupport`, `LabelOnly`, `NonInteractiveConcepts`, `InteractiveConcepts`).
These are the same four experimental arms in the same order, so map them
before pooling the two studies.

### Joining responses back to stimuli

`CBM-user-study-cub` has a `StimX_TestSampleIdx` column (X = 1..10) giving the
`sample_idx` into `cub-mirror`'s test split, so you can recover the exact
image, concepts, and label behind any stimulus a participant saw. `-1` means no
stimulus at that position.

```python
from datasets import load_dataset

resp = load_dataset("NWeak/CBM-user-study-cub", split="test").to_pandas()
cub  = load_dataset("NWeak/cub-mirror", split="test").to_pandas()

stim1 = resp.merge(cub, left_on="Stim1_TestSampleIdx", right_on="sample_idx")
```

## Running the experiments

The two tutorial notebooks run the full pipeline end to end, sourcing all data
from the Hub, with no local dataset and no preprocessing step:

- [`cub/notebooks/tutorial.ipynb`](cub/notebooks/tutorial.ipynb)
- [`emails/scripts/tutorial.ipynb`](emails/scripts/tutorial.ipynb)

Each trains the concept classifiers and the label predictor from scratch,
evaluates end to end, and verifies that the hand-computable linear formula
reproduces the model's predictions exactly.

| Path | Purpose |
|---|---|
| `cub/notebooks/generate_data.ipynb` | Builds the CUB stimulus set used in the study |
| `cub/notebooks/train_model.ipynb` | CUB model training and evaluation |
| `emails/scripts/preprocessing.ipynb` | Corpus filtering, embedding, 19→6 concept merge |
| `emails/scripts/train_concept_extractor2.ipynb` | Email model training, with regression-guard assertions |
| `{cub,emails}/scripts/report_test_accuracy.py` | Test-set accuracy tables |

Preprocessing is documented in
[`cub/cub_preprocessing.md`](cub/cub_preprocessing.md) and
[`emails/emails_preprocessing.md`](emails/emails_preprocessing.md); dataset
statistics in [`DATASET_STATISTICS.md`](DATASET_STATISTICS.md).

## Cleaning the participant data

The paper's analyses use the **cleaned** sets (342 CUB + 363 emails = 705
participants). A participant is excluded if they were blank/incomplete, failed
either attention check, or left the browser tab more than 3 times in total
across the 10 stimuli (a proxy for consulting an outside tool).

| Script | Does |
|---|---|
| `cub/scripts/prepare_user_study_dataset.py` | Turns the raw CUB export into a Hub-ready CSV: adds `StimX_TestSampleIdx`, relabels the `"None"` baseline arm to `NoSupport`, drops never-started rows |
| `cub/scripts/prepare_analysis_data.py` | Derives the cleaned wide/long CUB tables from scratch. **Superseded** by the official cleaned files, but kept for provenance. Its output was verified identical to them (same 342 participants, same values) |
| `emails/scripts/verify_participant_data.py` | Hard-assertion verification of the emails clean/full CSVs before publishing: confirms the 363 are a strict subset of the 417 and that all 54 exclusions are explained, with 0 unexplained |

For CUB, attention checks pass when `AttentionCheck1` is answered `"Le Conte"`
with confidence `1` and `AttentionCheck2` is answered `"Savannah"` with
confidence `13` (the two extremes of the bipolar 1 to 13 scale).

Full write-ups:
[`emails/PARTICIPANT_DATA_REPORT.md`](emails/PARTICIPANT_DATA_REPORT.md) for
the exclusion counts and verification procedure, and
[`emails/DATASET_NOTES.md`](emails/DATASET_NOTES.md) for two known data-entry
issues in the raw emails export.

## Repository layout

```
cub/                  # experiment 1: bird species identification
├── notebooks/        # tutorial, stimulus generation, training
├── scripts/          # encoding, publishing, cleaning, analysis
├── metadata/cub/     # class/concept name lists (tracked)
├── figures/
└── data/             # datasets + generated artifacts (gitignored; on the Hub)
emails/               # experiment 2: phishing detection, same structure
sosci_templates/      # SoSci Survey exports + stimuli, for re-running the studies
power_analysis/       # pre-registration power analysis (R)
pyproject.toml        # one shared uv environment for everything
```

## Acknowledgments

### Emails experiment

The email corpus is **not ours**. It comes from:

> Rebeka Toth, Tamas Bisztray, Richard A. Dubniczky.
> *Constructing and Benchmarking: a Labeled Email Dataset for Text-Based
> Phishing and Spam Detection Framework.*
> [arXiv:2511.21448](https://arxiv.org/abs/2511.21448),
> [github.com/DataPhish/PhishingSpamDataSet](https://github.com/DataPhish/PhishingSpamDataSet)

Their **PhishingSpamDataSet** (roughly 12,000 emails) supplies the messages
themselves and their original annotations: the `Subject`, `Body`, `Sender` and
`URL(s)` fields, the phishing/spam/legitimate `Type` label, the human vs.
LLM-generated `Created by` flag, the upstream `Source` and `Year`, and the
emotional-appeal and motivation annotations (`LLM detected emotion`,
`LLM detected motivation`). It is released under CC-BY-4.0, and our
redistribution keeps that licence and this attribution.

Our own contribution on top of it is:

- the filtering down to 2,330 usable emails (English, under 400 words,
  human-authored) and the pinned train/val/test split,
- the sentence embeddings (`all-MiniLM-L6-v2`),
- the merge of their 19 emotion and motivation annotations into the 6 concepts
  the study actually measured (`concept_gts_f`),
- the trained concept and label predictors and their outputs
  (`NWeak/emails-user-study`),
- and the human participant responses collected in our user study
  (`NWeak/CBM-user-study-emails`), which are entirely ours.

If you use the email data, please cite Toth et al. alongside this work.

### CUB experiment

The images and the official class and concept annotations come from
CUB-200-2011:

> C. Wah, S. Branson, P. Welinder, P. Perona, S. Belongie.
> *The Caltech-UCSD Birds-200-2011 Dataset.*
> Technical Report CNS-TR-2011-001, California Institute of Technology, 2011.

Redistributed under Caltech's non-commercial research-use terms. Our
contribution is the CLIP encoding, the concept and label predictors, and the
participant responses.

## Citation

```bibtex
@misc{bogani2026cbm,
  title  = {Are Concept Bottleneck Models Effective as Decision-Support Systems?},
  author = {Bogani, Alessandro and Debole, Nicola and Marconato, Emanuele
            and Pugnana, Andrea and Tentori, Katya and Passerini, Andrea},
  year   = {2026},
  eprint = {2608.25581},
  archivePrefix = {arXiv}
}
```
