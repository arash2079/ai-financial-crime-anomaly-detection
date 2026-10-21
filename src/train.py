"""Fixed sklearn baselines: fit train, select by validation AP, then score test."""

import argparse
import csv
import json
import math
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import SCOPE_START, TRAIN_END, VALIDATION_END, SCOPE_END, SOURCE_SHA256, sha256_file
from .features import FEATURE_NAMES

SEED = 42
LOG_COLUMNS = (1, 2, 3, 4, 6)
LR_PARAMS = dict(C=1, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=SEED)
RF_PARAMS = dict(n_estimators=100, max_depth=12, min_samples_leaf=10,
                 class_weight="balanced", random_state=SEED, n_jobs=2)
IF_PARAMS = dict(n_estimators=100, max_samples=256, contamination="auto", random_state=SEED, n_jobs=2)


def load_dataset(path):
    """Read only the seven named numerical predictors; metadata stays separate."""
    ids, timestamps, labels, splits, features = [], [], [], [], []
    seen, previous = set(), None
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["id", "timestamp", "label", "split", *FEATURE_NAMES]:
            raise ValueError("feature CSV header must match the explicit seven-feature schema")
        for line, row in enumerate(reader, 2):
            try:
                if len(row) != 4 + len(FEATURE_NAMES):
                    raise ValueError("wrong number of columns")
                identifier = row[0]
                if not identifier.strip() or identifier in seen:
                    raise ValueError("transaction IDs must be non-empty and unique")
                timestamp = datetime.fromisoformat(row[1].replace("Z", "+00:00"))
                if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                    raise ValueError("timestamps must be timezone-aware")
                timestamp = timestamp.astimezone(timezone.utc)
                if previous is not None and timestamp < previous:
                    raise ValueError("rows must be sorted chronologically")
                if not SCOPE_START <= timestamp < SCOPE_END:
                    raise ValueError("timestamp is outside the frozen ten-day scope")
                expected = "train" if timestamp < TRAIN_END else "validation" if timestamp < VALIDATION_END else "test"
                if row[3] != expected:
                    raise ValueError("split does not match the fixed calendar boundaries")
                if row[2] not in ("0", "1"):
                    raise ValueError("label must be 0 or 1")
                values = [float(value) for value in row[4:]]
                if not all(math.isfinite(value) and value >= 0 for value in values):
                    raise ValueError("all predictors must be finite and nonnegative")
                ids.append(identifier)
                timestamps.append(timestamp)
                labels.append(int(row[2]))
                splits.append(row[3])
                features.append(values)
                seen.add(identifier)
                previous = timestamp
            except ValueError as error:
                raise ValueError(f"CSV line {line}: {error}") from error
    X = np.asarray(features, dtype=float).reshape(-1, len(FEATURE_NAMES))
    y, split_array = np.asarray(labels, dtype=int), np.asarray(splits, dtype=str)
    for split in ("train", "validation", "test"):
        if set(y[split_array == split].tolist()) != {0, 1}:
            raise ValueError(f"{split} must contain both classes")
    return {"ids": ids, "timestamps": timestamps, "X": X, "y": y, "splits": split_array}


def make_models():
    """Construct six fixed candidates. All fitted estimators receive seven columns."""
    def amount_column():
        return ColumnTransformer([("amount", "passthrough", [0])])

    def history_transform():
        # One transformer per column preserves the original seven-column order.
        return ColumnTransformer([
            (f"column_{i}", FunctionTransformer(np.log1p) if i in LOG_COLUMNS else "passthrough", [i])
            for i in range(len(FEATURE_NAMES))
        ])

    def logistic(transform):
        return Pipeline([("transform", transform),
                         ("scale", StandardScaler()), ("model", LogisticRegression(**LR_PARAMS))])

    return {
        "amount_rule": None,
        "logistic_amount": logistic(amount_column()),
        "logistic_history": logistic(history_transform()),
        "forest_amount": Pipeline([("select", amount_column()),
                                   ("model", RandomForestClassifier(**RF_PARAMS))]),
        "forest_history": RandomForestClassifier(**RF_PARAMS),
        "isolation_history": IsolationForest(**IF_PARAMS),
    }


def score_model(name, model, X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or not np.isfinite(X).all() or (X < 0).any():
        raise ValueError("scoring requires finite nonnegative seven-column input")
    if name == "amount_rule":
        scores = X[:, 0]
    elif name == "isolation_history":
        scores = -model.score_samples(X)
    elif name in ("logistic_amount", "logistic_history", "forest_amount", "forest_history"):
        scores = model.predict_proba(X)[:, list(model.classes_).index(1)]
    else:
        raise ValueError(f"unknown candidate: {name}")
    if not np.isfinite(scores).all():
        raise ValueError(f"{name} produced non-finite scores")
    return np.asarray(scores, dtype=float)


def ranking_metrics(y, scores, budget=.05):
    """Rank descending, keeping original chronological order when scores tie."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not len(y):
        raise ValueError("labels and scores must be equally sized non-empty 1D arrays")
    if set(y.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("metrics require both binary classes and finite scores")
    if not math.isfinite(budget) or not 0 < budget <= 1:
        raise ValueError("review budget must be in (0, 1]")
    k = math.ceil(len(y) * budget)
    order = np.argsort(-scores, kind="stable")
    tp = int(np.sum(y[order[:k]]))
    positives = int(np.sum(y))
    return {"average_precision": float(average_precision_score(y, scores)),
            "review_count": k, "true_positives": tp, "false_positives": k - tp,
            "false_negatives": positives - tp, "precision": tp / k, "recall": tp / positives,
            "budget": budget, "rows": len(y), "positives": positives}


def select_winner(validation_metrics):
    return min(validation_metrics, key=lambda name: (-validation_metrics[name]["average_precision"], name))


def _error_summary(X, y, scores, budget=.05):
    reviewed = np.zeros(len(y), dtype=bool)
    reviewed[np.argsort(-scores, kind="stable")[:math.ceil(len(y) * budget)]] = True
    groups = {"reviewed": reviewed, "true_positives": reviewed & (y == 1),
              "false_positives": reviewed & (y == 0), "missed_positives": ~reviewed & (y == 1)}
    summary = {}
    for name, mask in groups.items():
        medians = np.median(X[mask], axis=0) if mask.any() else [None] * len(FEATURE_NAMES)
        summary[name] = {"count": int(mask.sum()),
                         "feature_medians": {key: None if value is None else float(value)
                                             for key, value in zip(FEATURE_NAMES, medians)}}
    return summary


def run_experiment(input_path="data/features.csv", output_dir="artifacts"):
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    paths = {"models": output_dir / "models.joblib", "scores": output_dir / "scores.npz",
             "results": output_dir / "results.json"}
    if output_dir.resolve() == input_path.resolve() or input_path.resolve() in [path.resolve() for path in paths.values()]:
        raise ValueError("artifact paths must not overwrite the input CSV")
    dataset = load_dataset(input_path)
    X, y, splits = dataset["X"], dataset["y"], dataset["splits"]
    masks = {split: splits == split for split in ("train", "validation", "test")}
    models, validation, test, arrays = make_models(), {}, {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for name, model in models.items():
            if model is not None:
                if name == "isolation_history":
                    model.fit(X[masks["train"]])  # Unsupervised fit: no labels passed.
                else:
                    model.fit(X[masks["train"]], y[masks["train"]])
            scores = score_model(name, model, X[masks["validation"]])
            arrays[f"validation_{name}"] = scores
            validation[name] = ranking_metrics(y[masks["validation"]], scores)

        # Freeze the winner before any candidate is scored on the test split.
        winner = select_winner(validation)
        for name, model in models.items():
            scores = score_model(name, model, X[masks["test"]])
            arrays[f"test_{name}"] = scores
            test[name] = ranking_metrics(y[masks["test"]], scores)
    for split in ("validation", "test"):
        arrays[f"X_{split}"] = X[masks[split]]
        arrays[f"y_{split}"] = y[masks[split]]
    report = {
        "selected_model": winner, "selection": "highest validation average precision; alphabetical tie-break; no tuning",
        "validation": validation, "test": test,
        "counts": {split: {"rows": int(mask.sum()), "positives": int(y[mask].sum())} for split, mask in masks.items()},
        "feature_names": list(FEATURE_NAMES), "seed": SEED,
        "configs": {"amount_rule": "descending log_amount", "logistic": LR_PARAMS, "forest": RF_PARAMS,
                    "isolation": IF_PARAMS, "logistic_history_log1p_columns": [FEATURE_NAMES[i] for i in LOG_COLUMNS],
                    "amount_candidates_columns": [FEATURE_NAMES[0]], "history_candidates_columns": list(FEATURE_NAMES),
                    "standardization": "logistic only; fitted on train only", "review_budget": .05},
        "input_sha256": sha256_file(input_path), "source_sha256": SOURCE_SHA256,
        "source_provenance": "source_sha256 is the expected pinned raw hash, verified by preparation; this trainer hashes only its actual input CSV",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
        "error_summary": _error_summary(X[masks["test"]], y[masks["test"]], arrays[f"test_{winner}"]),
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, paths["models"])
    np.savez_compressed(paths["scores"], **arrays)
    paths["results"].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/features.csv")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    try:
        report = run_experiment(args.input, args.output_dir)
    except (ValueError, OSError, ConvergenceWarning) as error:
        parser.error(str(error))
    print(f"Selected by validation AP: {report['selected_model']}; results: {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
