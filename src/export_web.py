"""Export the saved forest and aggregate results for browser inference; no fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .data import sha256_file
from .features import FEATURE_NAMES
from .train import ranking_metrics


def export_forest(model, source_model_sha256, training_input_sha256):
    if not isinstance(model, RandomForestClassifier) or not hasattr(model, "estimators_") or model.n_features_in_ != len(FEATURE_NAMES):
        raise ValueError("expected a fitted seven-feature RandomForestClassifier")
    if list(model.classes_) != [0, 1]:
        raise ValueError("expected binary classes [0, 1]")
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :]
        positive = values[:, 1] / values.sum(axis=1)
        trees.append({"left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
                      "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
                      "value": positive.tolist()})
    return {"schema_version": 1, "model_name": "forest_history", "feature_names": list(FEATURE_NAMES),
            "source_model_sha256": source_model_sha256, "training_input_sha256": training_input_sha256,
            "trees": trees}


def make_parity(model):
    """Build 80 synthetic vectors from model thresholds, never source records."""
    nodes = [(int(feature), float(threshold)) for estimator in model.estimators_
             for feature, threshold in zip(estimator.tree_.feature, estimator.tree_.threshold) if feature >= 0]
    limits = np.ones(len(FEATURE_NAMES))
    for feature, threshold in nodes:
        limits[feature] = max(limits[feature], 2 * threshold)
    limits = np.minimum(limits, np.finfo(np.float32).max / 1000)
    limits[5] = 1  # is_new_receiver is binary.
    vectors = [np.zeros(7), limits.copy()]
    for feature in range(7):
        vector = np.zeros(7)
        vector[feature] = limits[feature]
        vectors.append(vector)
    rng = np.random.default_rng(42)
    while len(vectors) < 20:
        vector = np.expm1(rng.uniform(0, np.log1p(limits)))
        vector[[1, 4]] = np.floor(vector[[1, 4]])
        vector[5] = rng.integers(0, 2)
        vectors.append(vector)
    if nodes:
        # Exact float32 neighbors and a float64 value just above a threshold
        # exercise sklearn's input cast before its <= split comparison.
        for index in range(20):
            feature, threshold = nodes[index % len(nodes)]
            center = np.float32(threshold)
            below = center if float(center) <= threshold else np.nextafter(center, np.float32(-np.inf))
            above = center if float(center) > threshold else np.nextafter(center, np.float32(np.inf))
            for value in (below, above, np.nextafter(threshold, np.inf)):
                vector = limits / 4
                vector[5] = 1
                vector[feature] = max(0, float(value))
                vectors.append(vector)
    else:  # Constant fixture forests have no split boundaries.
        vectors.extend(vector.copy() for vector in (vectors * 3)[:60])
    scores = model.predict_proba(np.asarray(vectors))[:, 1]
    return {"schema_version": 1,
            "description": "Synthetic feature vectors from model threshold ranges, extremes and float32 boundary neighbors; not transaction rows.",
            "cases": [{"features": vector.tolist(), "expected": float(score)} for vector, score in zip(vectors, scores)]}


def _json_bytes(payload):
    return (json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def export_web(model_path="artifacts/models.joblib", metrics_path="results/metrics.json",
               scores_path="artifacts/scores.npz", output_dir="web/data",
               parity_path="tests/fixtures/web_parity.json"):
    models = joblib.load(model_path)  # Only load the project's own trusted artifact.
    report = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    if report["selected_model"] != "forest_history" or report["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("aggregate report does not select the expected seven-feature forest")
    model = models["forest_history"]
    model_hash = sha256_file(model_path)
    payload = export_forest(model, model_hash, report["input_sha256"])
    forest_bytes = _json_bytes(payload)
    forest_hash = hashlib.sha256(forest_bytes).hexdigest()
    with np.load(scores_path, allow_pickle=False) as arrays:
        X, y, scores = arrays["X_test"], arrays["y_test"], arrays["test_forest_history"]
        expected = model.predict_proba(X)[:, 1]
        if not np.allclose(scores, expected, rtol=0, atol=1e-12):
            raise ValueError("saved test scores do not match the saved forest")
        if len(y) != report["counts"]["test"]["rows"] or int(y.sum()) != report["counts"]["test"]["positives"]:
            raise ValueError("saved test counts do not match the aggregate report")
        report["selected_review_budgets"] = [ranking_metrics(y, scores, budget)
                                              for budget in (.01, .02, .05, .1, .2)]
    report.pop("artifacts", None)  # Local paths are not public result data.
    report["web_export"] = {"schema_version": 1, "source_model_sha256": model_hash,
                            "forest_json_sha256": forest_hash, "training_input_sha256": report["input_sha256"],
                            "inference": "No preprocessing; cast each feature to float32, use <= left branches, average positive leaf fractions."}
    parity = make_parity(model)
    parity["source_model_sha256"] = model_hash
    parity["forest_json_sha256"] = forest_hash
    paths = {"forest": Path(output_dir) / "forest.json", "report": Path(output_dir) / "report.json",
             "parity": Path(parity_path)}
    inputs = {Path(path).resolve() for path in (model_path, metrics_path, scores_path)}
    if len({path.resolve() for path in paths.values()}) != 3 or any(path.resolve() in inputs for path in paths.values()):
        raise ValueError("export files must be distinct and must not overwrite inputs")
    for path, data in ((paths["forest"], forest_bytes), (paths["report"], _json_bytes(report)),
                       (paths["parity"], _json_bytes(parity))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return {"forest_bytes": len(forest_bytes), "forest_sha256": forest_hash,
            "source_model_sha256": model_hash, "parity_cases": len(parity["cases"]),
            "paths": {name: str(path) for name, path in paths.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/models.joblib")
    parser.add_argument("--metrics", default="results/metrics.json")
    parser.add_argument("--scores", default="artifacts/scores.npz")
    parser.add_argument("--output-dir", default="web/data")
    parser.add_argument("--parity", default="tests/fixtures/web_parity.json")
    args = parser.parse_args()
    try:
        summary = export_web(args.model, args.metrics, args.scores, args.output_dir, args.parity)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
