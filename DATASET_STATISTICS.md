# Dataset Statistics

Concept-annotation counts and per-class example counts for both experiments, computed directly from the published train/val/test splits (`emails/data/*.parquet`, `cub/data/cub_csv/*.csv`). "Occurrence" = number of rows where that concept's ground-truth value is 1.

## Summary

"#concepts" = total positive concept annotations in the split (sum of occurrences across the 6 modeling concepts each dataset uses), computed over the **whole** split (all rows, all classes). Note this is *not* the same as the concept extractor's actual train/eval subsets for CUB, since CUB's concept extractor is trained on `train` *excluding* the sparrow pair (4,745 samples, 7,223 concepts) and evaluated only on the sparrow-pair rows of `val`/`test` (9/59 samples) -- see the "Concept extractor: concept distribution, per split" tables below for those exact figures.

| Dataset | Train #samples | Train #concepts | Test #samples | Test #concepts | Val #samples | Val #concepts |
|---|---|---|---|---|---|---|
| Emails | 1,064 | 1,929 | 1,000 | 1,832 | 266 | 481 |
| CUB | 4,796 | 7,376 | 5,794 | 8,943 | 1,198 | 1,852 |

## Emails (phishing-email classification)

### Raw corpus and preprocessing funnel

The raw corpus (`merged_emails_with_categories.jsonl`, annotated with `lang`/`words` in `data/data.csv`) is filtered down to the "usable" set before any splitting or modeling happens.

| Stage | Rows |
|---|---|
| Raw corpus | 12,300 |
| ...passing `lang == 'en'` | 11,890 |
| ...passing `words < 400` | 10,059 |
| ...passing `Created by == 'Human'` | 3,073 |
| **Usable (all three, jointly)** | **2,330** |

The three conditions are applied simultaneously (logical AND), so the individual pass-counts above overlap heavily. A clearer view is the marginal cost of each condition -- how many additional rows would be recovered if *only that* condition were dropped, holding the other two fixed:

| Condition relaxed | Additional rows recovered |
|---|---|
| `lang == 'en'` | +94 |
| `words < 400` | +634 |
| `Created by == 'Human'` | +7,366 |

Authorship is by far the dominant filter -- most of the raw corpus is LLM-generated (synthetic) rather than human-authored (`Created by` distribution: `Human`=3,073, plus 25 distinct LLM-model labels such as `deepseek-chat`=3,075, `OpenAI`=3,075, `LLM (GPT-4o)`=340, etc., totaling 9,227 non-human rows).

**Word count** (whitespace-split `Body` length):

| | Raw corpus (n=12,300) | Usable set (n=2,330) |
|---|---|---|
| Mean | 271.6 | 165.8 |
| Std | 351.8 | 99.6 |
| Min | 0 | 3 |
| 25% | 83 | 81 |
| Median | 165.5 | 150.5 |
| 75% | 311 | 248 |
| Max | 4,220 | 399 |

**`Type` (task label source) distribution, raw vs. usable:**

| Type | Raw | Usable |
|---|---|---|
| Phishing | 4,360 | 923 |
| Spam | 4,000 | 641 |
| Valid | 3,800 | 731 |
| Phishing Simulation | 140 | 35 |

`Phishing` and `Phishing Simulation` are merged into task label `0` (Phishing) downstream: $923 + 35 = 958$, matching the train+val+test Phishing counts in the split table below ($366+92+500=958$) -- a useful cross-check that the funnel and the final splits agree.

### Split sizes and class distribution

| Split | Total rows | Phishing (0) | Valid (1) | Spam (2) |
|---|---|---|---|---|
| train | 1064 | 366 | 185 | 513 |
| val | 266 | 92 | 46 | 128 |
| test | 1000 | 500 | 500 | 0 |

Note: `test` (the `user_study_data` split) is class-balanced by construction (500 Phishing / 500 Valid, Spam excluded). The label predictor is trained/evaluated only on the binary Phishing-vs-Valid subset of `train`/`val` (`train_class`: 366+185=551 rows; see `supplementary_materials.tex`).

### Label predictor (task predictor): class distribution, per split

Binary (Phishing/Valid) subset only, Spam excluded (`train_class`/`val_class`/`test_class`). Only `train_class` is used for fitting; `val_class`/`test_class` are the evaluation subsets.

**train (n=551)**

| Class | % class | # class |
|---|---|---|
| Phishing (0) | 66.42% | 366 |
| Valid (1) | 33.58% | 185 |
| **Total** | **100%** | **551** |

**test (n=1,000)**

| Class | % class | # class |
|---|---|---|
| Phishing (0) | 50.00% | 500 |
| Valid (1) | 50.00% | 500 |
| **Total** | **100%** | **1,000** |

**val (n=138)**

| Class | % class | # class |
|---|---|---|
| Phishing (0) | 66.67% | 92 |
| Valid (1) | 33.33% | 46 |
| **Total** | **100%** | **138** |

### Concept extractor: concept distribution, per split

Full split (all rows regardless of task label), using `concept_gts_f`. Concepts are multi-label (not mutually exclusive), so `% concept` is the fraction of that split's rows where the concept is positive -- rows do not sum to 100%.

**train (n=1,064)**

| Concept | % concept | # concept |
|---|---|---|
| Fear+Authority | 40.60% | 432 |
| Urgency | 55.08% | 586 |
| Curiosity | 46.05% | 490 |
| Neutral | 29.14% | 310 |
| Reply | 5.73% | 61 |
| Open attachment | 4.70% | 50 |

**test (n=1,000)**

| Concept | % concept | # concept |
|---|---|---|
| Fear+Authority | 50.10% | 501 |
| Urgency | 55.30% | 553 |
| Curiosity | 26.20% | 262 |
| Neutral | 41.60% | 416 |
| Reply | 3.30% | 33 |
| Open attachment | 6.70% | 67 |

**val (n=266)**

| Concept | % concept | # concept |
|---|---|---|
| Fear+Authority | 42.86% | 114 |
| Urgency | 56.39% | 150 |
| Curiosity | 43.98% | 117 |
| Neutral | 24.81% | 66 |
| Reply | 7.89% | 21 |
| Open attachment | 4.89% | 13 |

### Concept occurrence -- the 6 concepts used for modeling (`concept_gts_f`)

| Concept | train (n=1064) | val (n=266) | test (n=1000) |
|---|---|---|---|
| Fear+Authority | 432 | 114 | 501 |
| Urgency | 586 | 150 | 553 |
| Curiosity | 490 | 117 | 262 |
| Neutral | 310 | 66 | 416 |
| Reply | 61 | 21 | 33 |
| Open attachment | 50 | 13 | 67 |

`Fear+Authority` is the OR-merge of the raw `Fear` and `Authority` concepts (see below).

### Concept occurrence -- full 19-concept LLM annotation (`concept_gts`, appendix)

| Concept | train (n=1064) | val (n=266) | test (n=1000) |
|---|---|---|---|
| Fear | 303 | 74 | 368 |
| Urgency | 586 | 150 | 553 |
| Authority | 403 | 104 | 471 |
| Greed | 280 | 80 | 40 |
| Curiosity | 490 | 117 | 262 |
| Altruism | 25 | 13 | 23 |
| Neutral | 310 | 66 | 416 |
| Humor | 1 | 1 | 0 |
| Promotion/ad | 8 | 2 | 1 |
| Unique | 1 | 0 | 0 |
| Gratitude | 1 | 0 | 0 |
| Blackmail | 1 | 0 | 0 |
| Follow the link | 787 | 197 | 571 |
| Reply | 61 | 21 | 33 |
| Open attachment | 50 | 13 | 67 |
| Financial fraud | 6 | 1 | 3 |
| Unknown | 180 | 40 | 344 |
| Promotion/ad | 28 | 9 | 1 |
| Data theft/credential harvesting | 27 | 11 | 41 |

## CUB-200-2011 (Le Conte's vs. Savannah Sparrow)

### Split sizes (full 200-class CUB-200-2011)

| Split | Total images | Distinct classes |
|---|---|---|
| train | 4796 | 200 |
| val | 1198 | 200 |
| test | 5794 | 200 |

Every image in the official split is used, unmodified -- no filtering or exclusion at the dataset-loading stage (unlike emails). Concepts/labels come from the official CUB-200-2011 annotations (112 binary concepts, 200 classes).

### The actual binary task: Le Conte's (123) vs. Savannah (126) Sparrow

| Split | Le Conte's Sparrow (123) | Savannah Sparrow (126) | Total |
|---|---|---|---|
| train | 25 | 26 | 51 |
| val | 5 | 4 | 9 |
| test | 29 | 30 | 59 |

The **concept extractor** is trained on all 4745 `train` images *except* the sparrow pair (i.e. `train` total 4796 minus the 51 sparrow-pair rows above). The **label predictor** is trained only on the sparrow pair's own 51 `train` rows, and evaluated on the sparrow pair's 59 `test` rows.

### Label predictor (task predictor): class distribution, per split

Sparrow-pair-only rows of each split (`train`/`val`/`test`). Only the `train` rows are used for fitting; `val`/`test` are evaluation subsets (the paper reports the `test` evaluation; `val` is included here for completeness).

**train (n=51)**

| Class | % class | # class |
|---|---|---|
| Le Conte's Sparrow (123) | 49.02% | 25 |
| Savannah Sparrow (126) | 50.98% | 26 |
| **Total** | **100%** | **51** |

**test (n=59)**

| Class | % class | # class |
|---|---|---|
| Le Conte's Sparrow (123) | 49.15% | 29 |
| Savannah Sparrow (126) | 50.85% | 30 |
| **Total** | **100%** | **59** |

**val (n=9)**

| Class | % class | # class |
|---|---|---|
| Le Conte's Sparrow (123) | 55.56% | 5 |
| Savannah Sparrow (126) | 44.44% | 4 |
| **Total** | **100%** | **9** |

### Concept extractor: concept distribution, per split

Trained on `train` *excluding* the sparrow pair (n=4,745, every other class), but -- matching how it's actually evaluated in the paper -- reported here on the sparrow-pair-only rows of `val`/`test` (since that's the subset the concept extractor is actually tested on: does concept knowledge learned on the other 198 classes transfer to the pair?). Concepts are multi-label, so `% concept` is the fraction of that subset's rows where the concept is positive -- rows do not sum to 100%.

**train (n=4,745, sparrow pair excluded)**

| Concept | % concept | # concept |
|---|---|---|
| striped breast | 6.81% | 323 |
| buff breast | 11.53% | 547 |
| white throat | 36.97% | 1,754 |
| buff nape | 8.01% | 380 |
| solid belly | 75.13% | 3,565 |
| brown crown | 13.78% | 654 |

**test (n=59, sparrow pair only)**

| Concept | % concept | # concept |
|---|---|---|
| striped breast | 50.85% | 30 |
| buff breast | 49.15% | 29 |
| white throat | 50.85% | 30 |
| buff nape | 49.15% | 29 |
| solid belly | 49.15% | 29 |
| brown crown | 50.85% | 30 |

**val (n=9, sparrow pair only)**

| Concept | % concept | # concept |
|---|---|---|
| striped breast | 44.44% | 4 |
| buff breast | 55.56% | 5 |
| white throat | 44.44% | 4 |
| buff nape | 55.56% | 5 |
| solid belly | 55.56% | 5 |
| brown crown | 44.44% | 4 |

### Concept occurrence -- the 6 masked concepts used for modeling

#### Whole split (all 200 classes)

| Concept | train (n=4796) | val (n=1198) | test (n=5794) |
|---|---|---|---|
| striped breast | 349 | 100 | 420 |
| buff breast | 572 | 146 | 715 |
| white throat | 1780 | 437 | 2123 |
| buff nape | 405 | 104 | 496 |
| solid belly | 3590 | 876 | 4342 |
| brown crown | 680 | 189 | 847 |

#### Concept extractor's actual training rows only (train, sparrow pair excluded, n=4745)

| Concept | Occurrence |
|---|---|
| striped breast | 323 |
| buff breast | 547 |
| white throat | 1754 |
| buff nape | 380 |
| solid belly | 3565 |
| brown crown | 654 |

#### Sparrow-pair rows only, per split (relevant to the label predictor)

| Concept | train (n=51) | val (n=9) | test (n=59) |
|---|---|---|---|
| striped breast | 26 | 4 | 30 |
| buff breast | 25 | 5 | 29 |
| white throat | 26 | 4 | 30 |
| buff nape | 25 | 5 | 29 |
| solid belly | 25 | 5 | 29 |
| brown crown | 26 | 4 | 30 |

### Concept occurrence -- full 112-concept CUB-200-2011 annotation (appendix)

| Concept | train (n=4796) | val (n=1198) | test (n=5794) |
|---|---|---|---|
| dagger beak | 407 | 102 | 499 |
| hooked seabird beak | 286 | 74 | 338 |
| all-purpose beak | 1947 | 482 | 2382 |
| cone beak | 1165 | 303 | 1408 |
| brown wing | 1174 | 293 | 1432 |
| grey wing | 1075 | 305 | 1337 |
| yellow wing | 305 | 55 | 357 |
| black wing | 1987 | 470 | 2361 |
| white wing | 981 | 247 | 1174 |
| buff wing | 617 | 161 | 763 |
| brown upperparts | 1101 | 276 | 1352 |
| grey upperparts | 1257 | 363 | 1564 |
| yellow upperparts | 398 | 82 | 477 |
| black upperparts | 1973 | 484 | 2350 |
| white upperparts | 843 | 205 | 1018 |
| buff upperparts | 593 | 155 | 735 |
| brown underparts | 281 | 79 | 354 |
| grey underparts | 490 | 110 | 551 |
| yellow underparts | 685 | 155 | 832 |
| black underparts | 733 | 195 | 892 |
| white underparts | 2050 | 526 | 2479 |
| buff underparts | 688 | 180 | 852 |
| solid breast | 3109 | 757 | 3761 |
| striped breast | 349 | 100 | 420 |
| multi-colored breast | 381 | 99 | 477 |
| brown back | 1082 | 266 | 1324 |
| grey back | 1201 | 329 | 1496 |
| yellow back | 252 | 48 | 299 |
| black back | 1310 | 308 | 1535 |
| white back | 579 | 139 | 690 |
| buff back | 520 | 139 | 644 |
| notched tail | 1371 | 368 | 1688 |
| brown upper-tail | 806 | 212 | 996 |
| grey upper-tail | 849 | 231 | 1062 |
| black upper-tail | 1482 | 345 | 1738 |
| white upper-tail | 581 | 138 | 689 |
| buff upper-tail | 399 | 109 | 492 |
| eyebrow head | 239 | 61 | 289 |
| plain head | 1044 | 245 | 1268 |
| brown breast | 308 | 82 | 384 |
| grey breast | 438 | 102 | 490 |
| yellow breast | 710 | 160 | 857 |
| black breast | 786 | 203 | 929 |
| white breast | 1774 | 443 | 2133 |
| buff breast | 572 | 146 | 715 |
| grey throat | 237 | 63 | 286 |
| yellow throat | 512 | 118 | 620 |
| black throat | 957 | 242 | 1141 |
| white throat | 1780 | 437 | 2123 |
| buff throat | 310 | 78 | 376 |
| black eye | 4598 | 1156 | 5596 |
| beak length about the same as head | 1628 | 408 | 1989 |
| beak length shorter than head | 3033 | 745 | 3635 |
| blue forehead | 239 | 61 | 293 |
| brown forehead | 658 | 180 | 818 |
| grey forehead | 544 | 146 | 676 |
| yellow forehead | 368 | 82 | 443 |
| black forehead | 1629 | 407 | 1943 |
| white forehead | 409 | 101 | 482 |
| brown under-tail | 722 | 175 | 887 |
| grey under-tail | 709 | 191 | 880 |
| black under-tail | 1866 | 472 | 2237 |
| white under-tail | 775 | 183 | 911 |
| buff under-tail | 405 | 103 | 492 |
| brown nape | 610 | 169 | 747 |
| grey nape | 783 | 207 | 964 |
| yellow nape | 291 | 69 | 354 |
| black nape | 1122 | 256 | 1293 |
| white nape | 900 | 208 | 1070 |
| buff nape | 405 | 104 | 496 |
| brown belly | 254 | 76 | 313 |
| grey belly | 412 | 98 | 471 |
| yellow belly | 710 | 160 | 862 |
| black belly | 593 | 155 | 716 |
| white belly | 1967 | 490 | 2361 |
| buff belly | 550 | 139 | 676 |
| rounded wings | 2025 | 523 | 2481 |
| pointed wings | 709 | 191 | 874 |
| size small | 3005 | 772 | 3677 |
| size medium | 897 | 213 | 1051 |
| very small size | 447 | 122 | 566 |
| duck-like | 243 | 57 | 286 |
| perching-like | 2849 | 719 | 3496 |
| solid back | 2377 | 590 | 2871 |
| striped back | 548 | 171 | 705 |
| multi-colored back | 337 | 83 | 405 |
| solid tail | 2080 | 498 | 2477 |
| striped tail | 307 | 82 | 388 |
| multi-colored tail | 610 | 170 | 767 |
| solid belly | 3590 | 876 | 4342 |
| brown primary color | 893 | 245 | 1105 |
| grey primary color | 1014 | 276 | 1265 |
| yellow primary color | 613 | 137 | 742 |
| black primary color | 1360 | 317 | 1595 |
| white primary color | 982 | 246 | 1171 |
| buff primary color | 427 | 112 | 536 |
| grey leg | 1087 | 262 | 1250 |
| black leg | 1322 | 327 | 1626 |
| buff leg | 531 | 127 | 635 |
| grey beak | 588 | 161 | 744 |
| black beak | 2365 | 572 | 2860 |
| buff beak | 257 | 73 | 316 |
| blue crown | 239 | 61 | 293 |
| brown crown | 680 | 189 | 847 |
| grey crown | 543 | 147 | 676 |
| yellow crown | 265 | 65 | 323 |
| black crown | 1623 | 383 | 1915 |
| white crown | 361 | 89 | 422 |
| solid wing | 1293 | 325 | 1532 |
| spotted wing | 241 | 58 | 283 |
| striped wing | 953 | 245 | 1164 |
| multi-colored wing | 1051 | 269 | 1295 |

