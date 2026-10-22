# Financial Crime Anomaly Detection

A personal ML project asking whether recent account activity improves
suspicious-transaction ranking compared with amount alone. It uses IBM's public
synthetic AML dataset from Kaggle, not employer or customer data.

This repository is a focused rebuild of an earlier exploratory prototype.
Implementation and testing use AI-assisted development.

## Current progress

The project downloads and verifies the IBM file, prepares a focused subset,
computes historical features and compares standard models. See
[data source notes](data/README.md) and the frozen
[comparison protocol](results/PROTOCOL.md). The [web interface](web/README.md)
now includes project notes, interactive browser-local inference and a recorded
review-budget comparison. Its workflow publishes to GitHub Pages after tests;
Pages must be enabled in this repository's settings before the first deployment.

The comparison is an amount-based rule, Logistic Regression, Random Forest
and Isolation Forest. Amount-only versions of the supervised models test whether
history adds value. No new machine-learning algorithm is proposed.

## First measured result

The validation-selected historical Random Forest found **160 of 322** label-1
test transactions while reviewing **2,037 of 40,731** rows (5% budget): 49.69%
recall and 7.85% precision, with **1,877 false alerts**. Its test AP was 0.087369,
compared with 0.054707 for the amount-only forest. These are synthetic-subset
results, not production accuracy. All six candidates, including weak results,
and the limitations are reported in [results](results/README.md).

## Run locally

Use Python 3.11 or newer (tested with Python 3.12). Data preparation uses the
standard library; modeling adds NumPy, SciPy and scikit-learn.
Run commands from the repository root:

```bash
python -m venv .venv
# Activate: source .venv/bin/activate (macOS/Linux)
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.download_data
python -m src.build_dataset --input data/raw/HI-Small_Trans.csv --output data/features.csv
python -m src.train --input data/features.csv --output-dir artifacts
python -m src.export_web
node --test tests/test_web.mjs
```

Node.js 24 is used for browser-inference tests. It is not required to view the
website. To serve the checked-in web assets locally:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory web
```

Open `http://127.0.0.1:8000`. No frontend dependency installation or build is
needed. See the [web notes](web/README.md) for inference, deployment and limits.

The download is approximately 476 MB. Alternatively, download
`HI-Small_Trans.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
and place it in `data/raw/`. The pipeline verifies its checksum. Downloaded data
and generated CSVs stay out of Git. Full training runs locally; CI uses only
small test fixtures and does not download the dataset.

Training writes `artifacts/models.joblib`, `artifacts/scores.npz` and
`artifacts/results.json`. These local artifacts are ignored by Git. Only load
joblib models you generated with trusted code; unpickling untrusted files is
unsafe. The model file contains preprocessing as well as estimators.

## Data and features

The study uses positive-valued ACH transfers with both payment and receiving
currency set to US Dollar, dated September 1-10, 2022. This keeps amount units
consistent and limits the scope. It is not a full IBM benchmark evaluation.
Selection uses transfer type, currency and dates, not labels or model scores.
There is no label-based sampling or evaluation-set balancing.

Splits are fixed before model training:

| Split | Source dates, inclusive |
| --- | --- |
| Train | September 1-6 |
| Validation | September 7-8 |
| Test | September 9-10 |

Account IDs combine bank and account columns. Self-transfers are retained.
Records are sorted by time. Source timestamps have no timezone; UTC is a
processing convention, not a claim about the original geographic timezone.

Features describe the current amount and the sender's preceding 24 hours of
outgoing transactions within the selected subset. Current and future events do not contribute to
historical features, including other transactions at the same timestamp.
Account and transaction identifiers are not numerical model inputs. Amounts are
stored as integer cents. Amount features are in USD.

| Feature | Meaning |
| --- | --- |
| `log_amount` | Natural log of one plus the current amount |
| `sender_count_24h` | Number of earlier outgoing payments in the window |
| `sender_total_24h` | Total value of those payments |
| `sender_mean_24h` | Mean value of those payments |
| `sender_unique_receivers_24h` | Number of distinct earlier receivers |
| `is_new_receiver_24h` | Whether the current receiver is absent from that window |
| `amount_to_prior_mean_ratio` | Current amount divided by the prior mean |

An empty history gives zero count, total, mean and amount ratio; the receiver is
treated as new. Transactions exactly 24 hours earlier are included. No timestamp
group crosses a split boundary. Earlier history carries across split boundaries,
but earlier labels never enter it.
The feature CSV keeps `label` and `split` alongside the features for later
evaluation. Neither column belongs in the model inputs.

## Project structure

- `src/`: download, data preparation, historical features and model comparison.
- `tests/`: hand-checkable fixtures, leakage checks and model round-trip tests.
- `data/`: source notes and ignored local data files.
- `results/`: frozen protocol and aggregate measured results.
- `web/`: static interface, browser inference and the exported forest.

## Limitations

IBM's published dataset contains synthetic data.
No employer, bank or customer data is used. Results on IBM data will not establish
effectiveness on real customer transactions. The project is an educational
demonstration, not a production AML system.
The subset omits other currencies and transfer types, including their possible
contribution to account history. Accounts can repeat across splits, so this does
not test unseen-account generalization. Features track outgoing payments only,
not incoming flows or graph structures. The first day starts without earlier history.

## License

Code: MIT; see [LICENSE](LICENSE). IBM data has its own license; see
[data source notes](data/README.md).
