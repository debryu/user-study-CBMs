# Emails preprocessing (before sentence-transformers encoding)

Companion to [`cub/cub_preprocessing.md`](../cub/cub_preprocessing.md).
Describes exactly what happens to the raw email data before it's encoded
by `sentence-transformers` in `emails/scripts/preprocessing.ipynb`, in the
order the notebook actually applies it.

## Short answer: unlike CUB, emails ARE filtered before encoding

CUB encodes every image in the official split unmodified. Emails go through
a row-level filter first — language, length, and authorship — because the
raw corpus (12,300 emails) contains material that isn't suitable for the
study (non-English text, overly long emails, non-human-authored content).
Only the ~2,330 emails that survive the filter ("usable") get embedded.

## 1. Load the raw corpus

`../data/merged_emails_with_categories.jsonl` (12,300 emails), loaded via
`pd.read_json(..., lines=True)`.

## 2. Annotate language and word count (not filtering yet)

For every row's `Body` text:

```python
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0   # deterministic langdetect results across runs

word_count = len(text.split())     # whitespace split, no tokenizer
language = detect(text)            # langdetect's own ISO 639-1 guess
```

Empty/whitespace-only bodies short-circuit to `{"word_count": 0, "language": "unknown"}`
without calling `detect()`. Results are stored as new `lang`/`words`
columns on the full 12,300-row dataframe, saved to `../data/data.csv`, then
reloaded (the notebook re-reads from disk rather than continuing in-memory —
no behavioral effect, just how the notebook is structured).

## 3. Filter to the "usable" set

The actual row-level filter — all three conditions applied simultaneously
(`&`, not sequential steps), exact expression from the notebook:

```python
usable = original[(original['lang'] == 'en') & (original['words'] < 400) & (original['Created by'] == 'Human')]
```

| Condition | Meaning |
|---|---|
| `lang == 'en'` | Keep only emails `langdetect` classified as English |
| `words < 400` | Keep only emails under 400 words (strict, 400 itself excluded) |
| `Created by == 'Human'` | Keep only human-authored emails (excludes e.g. LLM-generated/synthetic rows) |

This reduces 12,300 emails down to the "usable" set (2,330 emails). No
other row-level filter, and no deduplication, is applied — a
subject-based dedup/rebalancing branch exists further down the notebook
(see step 6) but was abandoned and never used in the final output.

## 4. Encode with sentence-transformers — on the full usable set, before any split

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

bodies = [usable['Body'].iloc[i] for i in range(len(usable))]
embeddings = model.encode(bodies)
```

- Only `Body` is embedded — `Subject` is never passed to the encoder.
- The raw `Body` text is embedded **as-is**: no HTML stripping, no
  lowercasing, no whitespace normalization, no additional truncation beyond
  the `words < 400` filter already applied in step 3.
- This runs once, over the entire usable set (all 2,330 rows) — **no
  train/val/test split exists yet at this point.**

## 5. Concept/label construction (still on the full, unsplit set)

The 19-dim concept ground truth (`concept_gts`) is built from the LLM-labeled
`emotion`/`motivation` fields, and the task `label` (`Valid`→1,
`Phishing`/`Phishing Simulation`→0, `Spam`→2) is assigned — still applied to
every row of the full usable set, before splitting.

## 6. Splitting — applied to the already-embedded data, not before

An earlier stochastic-balancing/splitting attempt (class-balanced sampling
with `random_state=42`, subject-duplicate rebalancing, `train_test_split`)
exists in the notebook but is **not used** — it turned out not to be
reproducible across pandas/numpy versions. The authoritative split instead
selects rows **by ID** from a fixed, pre-recorded list (matching the split
actually used for the real user study):

```python
with open('../metadata/emails/archival_split_ids.json') as f:
    archival_ids = json.load(f)

train = final[final['Original email No.'].isin(archival_ids['train'])]  # 1064 rows
val   = final[final['Original email No.'].isin(archival_ids['val'])]    #  266 rows
test  = final[final['Original email No.'].isin(archival_ids['test'])]   # 1000 rows
```

Because this operates on `final` (the full, already-embedded dataframe from
step 4), **the embeddings are computed once, before the split exists — the
split step only selects which already-embedded rows go into each output
file**, it never re-embeds anything.

## Output

- `data/{train,val,user_study_data}.parquet` — full rows including the
  `embedding` column.
- `data/{balanced_user_set,train}.csv` — same rows, embeddings dropped
  (kept only in the parquet files).
- `data/text_embeddings/{train,val,user_study}_embeddings.pt` — the same
  embeddings from step 4, re-saved standalone as stacked tensors (not
  recomputed).

## Summary (order of operations)

```
12,300 raw emails
  -> annotate lang + word count (langdetect, whitespace split)
  -> filter: lang=='en' AND words<400 AND Created by=='Human'   =>  2,330 "usable" emails
  -> encode Body (raw, unmodified text) with all-MiniLM-L6-v2   =>  embeddings on all 2,330
  -> build concept_gts / label                                  (still all 2,330, unsplit)
  -> split by fixed ID list (archival_split_ids.json)           =>  train / val / test
```
