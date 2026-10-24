# A short project walkthrough

Use this as a guide to exploring and explaining the experiment. It describes
the implemented work, not a statement about who independently authored it.

## 1. Start with the question

Does a sender's recent outgoing activity improve suspicious-transaction ranking
compared with amount alone? This is a comparison of standard algorithms on a
bounded synthetic dataset, with no proposed new ML algorithm.

Open the [live demo](https://arash2079.github.io/ai-financial-crime-anomaly-detection/)
and its About tab. The source is IBM's public Kaggle dataset, version 8,
HI-Small_Trans.csv. The study keeps positive USD-to-USD ACH transfers from
September 1–10, 2022: 201,381 rows, including 1,419 positive labels.
It does not use employer or customer data.

## 2. Explain how time is handled

The first six days train the models, the next two select the model, and the
last two evaluate it. Preprocessing fits on training rows only. Account IDs
index histories but are not numerical predictors. Seven features describe the
current amount and strictly earlier sender activity in the preceding 24 hours.
Transactions sharing a timestamp are scored before any of them update history.
Earlier activity can carry across split boundaries; labels cannot.

Amount-only and historical versions of Logistic Regression and Random Forest
test the value of context. The amount rule and unsupervised Isolation Forest
provide additional comparisons. Fixed settings and seed 42 keep the experiment
small and reproducible. The historical forest wins by validation AP.

## 3. Interpret the result carefully

On the untouched test period, the selected forest's AP is 0.087369, versus
0.054707 for the same forest using amount alone. At the fixed 5% review budget,
it reviews 2,037 rows: 160 true alerts, 1,877 false alerts, and 162 missed positives.
Recall is 49.69%; precision is 7.85%. These are different questions: how many
positives were found, and how many reviewed rows were positive?

Accuracy is not the headline for this imbalanced task. AP summarizes score
rankings across thresholds; precision/recall at 5% describe one review capacity.
Neither makes the model's individual score a calibrated probability.
The weak Isolation Forest result remains in the report.

## 4. Run the interactive example

In Try the model, choose each handcrafted scenario and press Run model.
Approximate scores with the published model are:

| Scenario | Ranking score |
| --- | ---: |
| No recent history | 0.4709 |
| Repeated outgoing activity | 0.0091 |
| Larger new payment | 0.0000 |

These examples have no laundering labels. A larger amount does not necessarily
produce a larger score: the forest learns feature splits from this dataset,
not a general rule that large payments are criminal. Zero is not proof of
safety. Hypothetical inputs can be outside the training distribution.

Expand the derived inputs to connect the form to the seven features. Change
an input to see the old score cleared. An invalid amount should give a clear
error. Changing the review budget explores saved test rankings, independently
of the hypothetical payment. Only 5% was the primary comparison; other budgets
are descriptive views, not subsequent model selection.

## 5. Know the limits and the code

Results apply to one short synthetic USD/ACH subset. Accounts can recur across
splits; unseen-account performance was not tested. Incoming flows, other payment
types and graph structure are omitted. False-alert volume remains high.
No repeated-seed or temporal uncertainty study establishes significance.

Read `src/data.py` for positional CSV parsing and exact cents, `src/features.py`
for causal histories, `src/train.py` for fitting and selection, and
`src/export_web.py` plus `web/model.mjs` for browser export and inference.
The browser follows the same trees, including sklearn's float32 input cast.
Synthetic boundary fixtures compare its scores with Python.

The [main README](../README.md) gives reproduction commands. Before presenting
the project, be able to explain these choices, the failed baseline, the false
alerts and the actual development assistance you used.
