# Authors

Code and materials for **[Are Concept Bottleneck Models Effective as
Decision-Support Systems?](https://arxiv.org/abs/2608.25581)** (arXiv:2608.25581).

Alessandro Bogani, Nicola Debole, Emanuele Marconato, Andrea Pugnana,
Katya Tentori, Andrea Passerini

## Who wrote what

Git history alone does not reflect this, because parts of the repository were
committed in bulk by whoever assembled it rather than by their author.

### Alessandro Bogani ([@aleBogani](https://github.com/aleBogani))

**`sosci_templates/`**: the SoSci Survey implementations of both studies. This
is the experiment as participants actually saw it: question flow, randomisation
into the four support conditions, the interactive concept widget, attention
checks, practice trials, and the trust questionnaire.

- `cub_study/TemplateSoSci.xml`, `email_study/TemplateSoSci_Emails.xml`
- `*/dataBaseContent.xlsx`, the stimulus databases each template reads
- `cub_study/Images/`, the stimuli and instruction figures

**`power_analysis/`**: the pre-registration power analysis that fixed the
target sample size.

- `PowerAnalysis_ExpMail_ThreeWayInteraction.R`
- `Results_SampleSize_WithNone_340.RData`, `dataLongComplete.xlsx`

The R analysis scripts distributed with the participant data on Google Drive
(`Analyses_Final.R`) are also his. They are not tracked here.

### Nicola Debole ([@debryu](https://github.com/debryu))

**`cub/`** and **`emails/`**: the modelling pipelines, the HuggingFace
publishing scripts, and the participant-data preparation and verification
scripts. Repository assembly and packaging.

## Datasets

The published datasets are the joint work of the authors. Licensing, including
which columns are ours and which carry upstream attribution requirements, is
set out in [`LICENSE`](LICENSE) and in the Acknowledgments section of
[`README.md`](README.md).
