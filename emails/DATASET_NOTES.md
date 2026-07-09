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

## Planned dataset versions

Two versions of the user-study results are planned:

1. **Full** — everything as-is, including the known-bad email 268/263 entries
   and all participants regardless of attention-check performance. Useful for
   anyone who wants to see/audit the raw data quality issue itself.
2. **Clean** (not yet generated) — email 268 removed, plus participants who
   failed the attention checks filtered out. `dataComplete.csv` has no
   precomputed `AttentionCheckN_Passed` column (unlike `Annotation Study/dataAnnotationStudy.csv`,
   which does) -- passing has to be derived by comparing
   `AttentionCheck{1,2}_ParticipantAnswer` against `AttentionCheck{1,2}_CorrectAnswer`
   per participant.
   - Open question: whether email 263 (activations wrong, GT fine) should
     also be dropped from the clean version, or just flagged/left in since
     its GT is trustworthy. Not yet decided.

## Single HF dataset vs. two separate repos

**Recommendation: one dataset repo, two named configs**, not two separate repos.

HuggingFace `datasets` supports multiple named "configs" (a.k.a. subsets) inside
a single dataset repo -- this is exactly the mechanism datasets like `glue` or
`common_voice` use for "same schema, different curation" variants. Concretely:

```python
# push_user_study_to_hub.py, called twice with different config_name / input:
ds_full.push_to_hub("NWeak/emails-user-study", config_name="full", split="test")
ds_clean.push_to_hub("NWeak/emails-user-study", config_name="clean", split="test")
```

Consumers then pick a version explicitly:

```python
load_dataset("NWeak/emails-user-study", "full", split="test")
load_dataset("NWeak/emails-user-study", "clean", split="test")
```

This keeps one repo (one README/dataset card, one set of stars/discussions/versioning)
while still letting users load exactly the variant they want, and keeps both
versions clearly documented side-by-side in the same card (e.g. "`clean`
excludes email 268 and N participants who failed attention checks").

Two separate repos (e.g. `emails-user-study` and `emails-user-study-clean`)
would also work and is simpler to browse in isolation, but duplicates the
dataset card/licensing text and splits the discussion/versioning history for
what's fundamentally the same underlying data with different row filters --
worse for this case than the multi-config approach.

`push_user_study_to_hub.py` will need a small update to accept a
`--config-name` argument once the clean version is actually generated.
