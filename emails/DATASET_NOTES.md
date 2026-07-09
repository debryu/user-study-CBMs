# Known data-quality issues: `emails` user study

## Root cause

While preparing the real user-study export (`emails/temp/User Study/dataComplete.csv`),
one of the annotators copy-pasted the wrong line for two stimulus entries.
As a result, two emails have incorrect participant-facing data:

| `Original email No.` | Wrong activations (`Feature{j}_Detected`) | Wrong ground truth (`Feature{j}_GT`) |
|---|---|---|
| **268** | Yes | Yes (`Reply` recorded as `0`, should be `1`) |
| **263** | Yes | No — GT is correct |

This was diagnosed empirically in `emails/scripts/train_concept_extractor2.ipynb`'s
sanity-check cells, then confirmed as a data-entry error (not a reproduction bug)
by cross-checking against `emails/temp/archival/user_study_data.parquet`:

- Both emails' own embedding, `concept_gts_f`, and `Body` are byte-identical
  between our reproduction and the archival data.
- Training a fresh concept-extractor SVM directly on the archival `train.parquet`
  embeddings reproduces our model's activations for both emails to within
  0.0003 -- i.e. our pipeline (split, embeddings, training) is correctly
  reproducing the archival model. The mismatch is entirely in what got
  recorded in `dataComplete.csv` for these two stimulus entries.

**Practical impact**: any check that compares our model's output against
`dataComplete.csv` will show:
- 99/24180 ground-truth mismatches, 100% attributable to email 268's bad
  `Reply` GT (it was shown to many different participants, hence 99 not 1).
- A handful of activation-magnitude mismatches for 268 and 263 specifically,
  independent of the GT issue.

These are **not** bugs in `preprocessing.ipynb` / `train_concept_extractor2.ipynb`
and do not need to be "fixed" in the pipeline -- they need to be filtered out
of the published dataset instead (see below).

## Published dataset versions (done — see PARTICIPANT_DATA_REPORT.md)

This section originally sketched a plan; it's now implemented. Full
verification methodology, exact exclusion-criteria counts, and per-config
row counts are in
[`PARTICIPANT_DATA_REPORT.md`](PARTICIPANT_DATA_REPORT.md) — this section
just records the outcome.

Published to **`NWeak/CBM-user-study-emails`** (a separate repo from
`NWeak/emails-user-study`, which holds our model's computed
activations/predictions, not participant responses) via
`scripts/push_participant_responses_to_hub.py`, as three configs:

- **`full`** — all 417 participants, everything as-is (public condition
  labels only relabeled, no filtering).
- **`clean_wide`** — 363 participants (one row each), manually cleaned by
  the researcher: removes blank/incomplete participants, participants who
  failed `AttentionCheck1`/`AttentionCheck2`, and participants who switched
  browser tabs more than 3 times total (a tab-switching risk heuristic,
  e.g. possible use of an external tool/LLM to answer). All 54 removed
  participants are accounted for by these three criteria (0 unexplained).
  Email 268 is excluded (its stimulus slots blanked in place) for
  participants who saw it; email 263's GT is correct and its binary concept
  pattern matched what participants should have seen, so its observations
  were **kept** — resolves the previously-open question of whether 263
  should also be dropped.
- **`clean_long`** — the same 363 participants, melted to one row per
  participant-stimulus (main 10 stimuli only); email 268 rows fully absent
  here (not just blanked).

The `AttentionCheckN_Passed`-must-be-derived note below turned out to be
necessary but not sufficient — attention-check failure alone explained 35 of
the 54 removed participants; the remaining 19 needed the tab-switching
criterion, confirmed against `dataWideClean.csv`'s own `TotalTimesTabLeft`
column.

## Single HF dataset vs. two separate repos

**Recommendation: one dataset repo, multiple named configs**, not separate
repos per version — implemented above (`full`/`clean_wide`/`clean_long` all
in `NWeak/CBM-user-study-emails`).

HuggingFace `datasets` supports multiple named "configs" (a.k.a. subsets) inside
a single dataset repo -- this is exactly the mechanism datasets like `glue` or
`common_voice` use for "same schema, different curation" variants:

```python
from datasets import load_dataset
load_dataset("NWeak/CBM-user-study-emails", "full", split="test")
load_dataset("NWeak/CBM-user-study-emails", "clean_wide", split="test")
load_dataset("NWeak/CBM-user-study-emails", "clean_long", split="test")
```

This keeps one repo (one README/dataset card, one set of stars/discussions/versioning)
while still letting users load exactly the variant they want, and keeps all
versions clearly documented side-by-side in the same card.

Two separate repos would also work and is simpler to browse in isolation,
but duplicates the dataset card/licensing text and splits the
discussion/versioning history for what's fundamentally the same underlying
data with different row filters -- worse for this case than the
multi-config approach.

Implementation note for anyone adding a fourth config later: don't
overwrite the whole `DatasetCard` after multiple `push_to_hub(config_name=...)`
calls -- that auto-manages the `configs:` YAML block (file paths per
config), and a hand-written duplicate risks drifting from the real
uploaded layout. Instead `DatasetCard.load(repo_id)` after all pushes and
only replace `.text` (and any license/tags fields), leaving `.data`'s
`configs` entry alone. See `push_participant_responses_to_hub.py`.
