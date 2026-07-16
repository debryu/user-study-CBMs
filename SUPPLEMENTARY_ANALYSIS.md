# Supplementary analysis: time, decisiveness, trust/confidence correlations

New analyses beyond the primary NoSupport vs. model-output vs.
model+concepts accuracy comparison, covering both experiments (`cub/` and
`emails/`). Three questions, each answered per experiment:

- **A. Time vs. support modality** — does the modality of support change
  how long participants take?
- **B. Decisiveness** — does support make participants change their
  classification answer (or, where possible, edit concept values) more or
  less often?
- **C. Trust / confidence / interaction / time correlations** — how do
  post-task trust, in-task confidence, answer-changing, and time-on-task
  relate to each other?

## How to reproduce

```bash
# Emails (data already cleaned -- see emails/PARTICIPANT_DATA_REPORT.md)
cd emails && uv run scripts/supplementary_analysis.py

# CUB (needs the cleaning step first -- no equivalent existed before this)
cd cub && uv run scripts/prepare_analysis_data.py && uv run scripts/supplementary_analysis.py
```

Figures are saved to `emails/figures/` and `cub/figures/` (tracked in git,
unlike `data/`/`temp/`). Shared plotting/stats logic lives in
`utils/participant_analysis.py`, used identically by both experiments'
`scripts/supplementary_analysis.py`. All condition comparisons use
Kruskal-Wallis (nonparametric — time/count data here are skewed, not
normal); all correlations use Spearman.

## Important caveat: CUB's participant data had no cleaning pipeline before this

Emails already had a manually-cleaned, verified participant dataset
(`dataWideClean.csv`/`dataLongClean.csv`, see `emails/PARTICIPANT_DATA_REPORT.md`).
**CUB had none** — `cub/temp/updated_complete.csv` includes all 568 raw
participants, with no attention-check or tab-switching exclusion applied
anywhere in this repo. New `cub/scripts/prepare_analysis_data.py` builds
the CUB equivalent, but with one assumption worth double-checking:

- **Attention-check pass criterion** — verified directly against the data
  (matches the single largest response pattern, 355/568 participants):
  `AttentionCheck1_ParticipantAnswer == "Le Conte" AND Confidence == 1`
  AND `AttentionCheck2_ParticipantAnswer == "Savannah" AND Confidence == 13`.
- **Tab-switching threshold (`sum(TimeTabWasLeft) > 3`)** — carried over
  **by analogy** from the emails study's confirmed threshold, not
  independently confirmed for CUB. If CUB used a different QC threshold
  (or none), the "clean" CUB numbers below would shift.
- **Net effect**: 226/568 CUB participants excluded (17 blank/incomplete,
  213 failed attention checks, 18 excessive tab-switching, with overlap) —
  a much higher exclusion rate (40%) than emails (54/417, 13%), driven
  almost entirely by the attention-check criterion. Worth sanity-checking
  this large a cut is intended/expected for CUB before treating the
  numbers below as final.

## A. Time vs. support modality

Both experiments show the same pattern: **time-on-task increases with how
much the support modality asks the participant to engage** (Kruskal-Wallis
significant in both, per-stimulus and per-participant-total):

| | NoSupport | Label-only | Non-interactive concepts | Interactive concepts |
|---|---|---|---|---|
| **Emails** mean total time (participant, min) | 4.23 | 4.52 | 4.45 | **5.91** |
| **CUB** mean total time (participant, min) | 2.43 | 2.40 | 3.37 | **4.17** |

Interactive-concept support takes noticeably longer than any other
condition in both experiments — consistent with participants actually
using the ability to edit concepts, not just glancing at extra output.
CUB additionally shows a bigger gap between "sees concepts" (FixedCBM) and
"sees only the label" (BlackBox) than emails does between the analogous
conditions — plausible given CUB concepts require visually checking image
regions, which is a slower judgment than reading text.

## B. Decisiveness

**No significant difference across conditions in either experiment**
(Kruskal-Wallis p=0.35 emails, p=0.60 CUB) — support modality does not
measurably change how often participants revise their classification
answer. This is a genuinely near-zero-inflated behavior in both studies:
the overwhelming majority of stimuli (>97%) see zero answer changes
regardless of condition.

Concept-value editing is (by UI design) only possible in the fully
interactive condition, where it's fairly common: **36.7%** of stimuli saw
at least one concept edit in emails (mean 0.69 edits/stimulus), **44.5%**
in CUB (mean 0.96 edits/stimulus) — participants engage with editable
concepts noticeably more in CUB than emails, consistent with CUB's task
taking visibly longer under interactive support (section A).

## C. Trust / confidence / interaction / time correlations

Both experiments show the same qualitative pattern, and all correlations
are weak (|rho| < 0.2) — trust, confidence, and time-on-task are only
loosely related to each other in this data:

| Pair | Emails rho (p) | CUB rho (p) |
|---|---|---|
| Trust vs. mean confidence | +0.12 (p=0.045) | +0.11 (p=0.075, n.s.) |
| Trust vs. total time | **-0.12 (p=0.050)** | **-0.16 (p=0.010)** |
| Confidence vs. total time | +0.04 (n.s.) | **+0.17 (p=0.001)** |
| Confidence vs. answer changes | +0.03 (n.s.) | +0.12 (p=0.022) |

The one consistent, significant-in-both-experiments finding: **participants
who report higher post-task trust tend to have spent less total time on
the task** (both negative, both p<0.05) — plausibly, participants who
trust the model more feel less need to deliberate. Trust could not be
computed for NoSupport participants (no model output to rate), so those
correlations are necessarily restricted to the three support conditions.

`TrustIndex` for both experiments is the mean of `TrustQuestionnaire_Q1..Q8`
(1-7 scale) with `Q2`/`Q5`/`Q6`/`Q8` reverse-scored (`8 - value`) — this
exact formula was reverse-engineered from and verified against emails'
already-published `TrustIndex` column (0.0 max diff across all 274
non-blank participants), then applied identically to CUB (same 8-item,
1-7-scale questionnaire structure) — not independently re-derived from a
codebook, so worth a spot-check if the questionnaire differed between
studies in some way not visible in the column structure.
