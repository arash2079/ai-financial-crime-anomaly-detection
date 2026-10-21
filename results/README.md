# First model comparison

Run on October 21, 2026, using the frozen [protocol](PROTOCOL.md) and the
[IBM/Kaggle preparation scope](../data/README.md). These are measured results
on synthetic USD ACH data, not real-bank performance or a full IBM benchmark.
The protocol was committed as `9897468` before training. No hyperparameters,
features, split dates or candidate choices were changed after seeing scores.

## Data and selection

| Split | Rows | Label-1 records |
| --- | ---: | ---: |
| Train | 121,962 | 796 |
| Validation | 38,688 | 301 |
| Test | 40,731 | 322 |

Fit only on train; no refit on validation. **forest_history** was selected by
highest validation Average Precision (AP), before test scoring. Seed: 42.
Prepared feature CSV SHA-256:
`c88ab55da0b9e328073bde6a0a17e6920d89c8ee02306bf4cf4b034dccaaa864`.

## Results

AP is sklearn Average Precision, not trapezoidal PR-AUC. Precision and recall
below use the top **2,037** test rows, ceil(5% of 40,731), with stable row-order
tie-breaking. All six preselected candidates are shown, not just the winner.

| Candidate | Validation AP | Test AP | Test precision at 5% | Test recall at 5% |
| --- | ---: | ---: | ---: | ---: |
| Amount rule | 0.014341 | 0.016579 | 0.44% | 2.80% |
| Logistic amount | 0.014341 | 0.016579 | 0.44% | 2.80% |
| Logistic history | 0.026599 | 0.025389 | 0.64% | 4.04% |
| Forest amount | 0.046434 | 0.054707 | 5.89% | 37.27% |
| Forest history (selected) | 0.102750 | 0.087369 | 7.85% | 49.69% |
| Isolation history | 0.005842 | 0.006225 | 0.44% | 2.80% |

The selected model found **160** of **322** positives, missed **162**, and
raised **1,877 false alerts**. Most reviewed rows (92.15%) were still label-0.
The test positive prevalence was 0.79%; the model's review precision was about
9.94 times that prevalence. This describes retrospective alert yield, not
accuracy, calibrated probability or a guarantee for new transactions.

## What this experiment suggests

Within this subset, historical features improved the forest compared with the
same forest configuration using amount alone: AP rose from 0.054707 to 0.087369,
and recall at the fixed review budget rose from 37.27% to 49.69%. Logistic
history also improved over its amount-only version, but remained weak. The
amount-only logistic model has the same ranking as the amount rule here.

Isolation Forest performed poorly on the laundering labels, despite detecting
unusual feature combinations. Its AP was below test prevalence. An unusual
transaction is not necessarily a laundering transaction. We retain this result
rather than tuning the test set or removing an unsuccessful candidate.

The forest is useful as an educational comparison, not ready for operational
use: half the positives were missed and false-alert volume remained high.
There is one seed and one temporal holdout, with no confidence intervals or
evidence of statistical significance. The synthetic population and recurring
accounts limit generalization.

## Small error analysis

Aggregate feature medians for true alerts, false alerts and missed positives
are included in [metrics.json](metrics.json). The median prior-24h sender
count was zero in all three groups. Many selected transactions have no earlier
outgoing ACH/USD history, limiting the available context. Median current amount
was approximately $12,568 among true alerts, $7,877 among false alerts and
$3,444 among missed positives. These are descriptions of score-defined groups,
not reasons proving why a transaction was classified or evidence of criminality.

Top-k review uses the entire held-out period's scores. It is not an online
decision threshold; the future interactive demo must make this distinction.

## Reproduce

Follow the commands in the [main README](../README.md). The full run uses
Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0 and joblib 1.5.3.
Pinned dependencies are in requirements.txt. Full training is local, not a CI
download-and-train job. Tests use separate artificial fixtures only.

`metrics.json` is a verbatim copy of the generated `artifacts/results.json`.
The referenced model and score paths are local ignored outputs, not downloadable
files in this repository. Raw transactions, account identifiers and row-level
predictions are not redistributed. Rebuilding reproduces those artifacts locally.

## Checks

31 unit/integration tests passed locally, including train-only preprocessing,
test-data mutation independence and fresh-process model reload. An independent
artifact check recomputed all 12 validation/test metric sets, reconciled saved
features and labels to the prepared CSV, and reproduced all six models' scores
after loading. No raw-data retraining is needed for the small CI tests.
