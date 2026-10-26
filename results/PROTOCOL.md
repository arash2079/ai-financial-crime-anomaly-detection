# Model comparison protocol

Written before running the model comparison. The earlier data preparation
commit fixes the source file, subset, seven features and chronological splits.
This is a small personal benchmark, not a new algorithm or a production system.

## Fixed candidates

All random seeds are 42. No parameter search, resampling or test-driven changes.

| Candidate | Inputs | Fixed configuration |
| --- | --- | --- |
| Amount rule | `log_amount` | Higher amount ranks first. No fitting. |
| Logistic amount | `log_amount` | Balanced class weights, C=1, lbfgs, max_iter=2000 |
| Logistic history | All seven features | Same logistic configuration |
| Forest amount | `log_amount` | 100 trees, max_depth=12, min_samples_leaf=10, balanced class weights |
| Forest history | All seven features | Same forest configuration |
| Isolation history | All seven features | 100 trees, max_samples=256, contamination=auto |

For logistic models only, apply log1p to nonnegative history counts, totals,
means, unique-receiver counts and the amount ratio. Do not log the existing
`log_amount` again or the new-receiver indicator. Fit StandardScaler on training
rows only. Tree models use the original seven numerical features. Fit Isolation
Forest on all training rows without labels, including their small unknown-to-the-
algorithm laundering fraction. Negate score_samples so higher means more anomalous.
Supervised models use label-1 predict_proba scores as ranking scores, not validated
real-world risk probabilities. Class weights do not balance evaluation data.

## Selection and evaluation

Fit using September 1-6 only. Compare validation average precision (AP) on
September 7-8. Select the highest validation AP, resolving exact ties by candidate
name alphabetically, before computing test scores. Do not refit on validation.
Report all six frozen candidates on September 9-10, identifying the validation-
selected model even if another model happens to score better on test.

AP uses sklearn.metrics.average_precision_score: it is not trapezoidal PR-AUC.
Also report precision and recall at a preselected 5% review budget, using exactly
ceil(0.05 * split rows) alerts. Equal scores at the budget boundary are resolved
by original prepared-row order (chronological with source-row ties). This choice
is deterministic but arbitrary. It can affect fixed-budget metrics, particularly
for constant-score groups. AP handles score ties as groups.

Top-k is a retrospective ranking exercise within each entire evaluation period.
It is not a learned score threshold, calibrated risk estimate or an online
guarantee that exactly 5% of an incoming stream will be reviewed. The web demo
explains that distinction.

Keep labels, IDs, timestamps and split names out of model inputs. Earlier
transaction history can cross split boundaries, but labels cannot. Report
positive prevalence and true/false alert counts, not accuracy as the headline.
For the selected model, summarize feature medians among true alerts, false
alerts and missed positives. These are descriptions, not causal explanations.

## Reproducibility and limitations

Save fitted models and per-row scores locally under ignored artifacts/. Publish
aggregate JSON results and a short interpretation, without raw transactions or
account identifiers. Record the feature CSV hash, source hash, package versions
and configuration. Only load joblib files produced locally by trusted code.

Keep weak or negative findings. No superiority, statistical significance or
real-bank effectiveness claim follows from one ten-day synthetic subset and
one seed. Accounts may recur across splits. This does not establish unseen-
account generalization. No repeated experiments will be used to select a better
test score. Any later change to the method must be reported as a new experiment.

Implementation references (version 1.8, matching requirements.txt):
[Logistic Regression](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html),
[Random Forest](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html),
[Isolation Forest](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.IsolationForest.html),
[Average Precision](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.average_precision_score.html).
