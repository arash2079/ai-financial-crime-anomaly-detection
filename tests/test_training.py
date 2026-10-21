"""Training checks on small artificial fixtures, never research observations."""

import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.features import FEATURE_NAMES
from src.train import (
    load_dataset, make_models, ranking_metrics, run_experiment, score_model, select_winner,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {
    "amount_rule", "logistic_amount", "logistic_history", "forest_amount",
    "forest_history", "isolation_history",
}
HEADER = ["id", "timestamp", "label", "split", *FEATURE_NAMES]


def fixture_rows():
    """Balanced test-only data with visibly different train/validation/test ranges."""
    rows = []
    for split, day, offset in (("train", 1, 0), ("validation", 7, 10), ("test", 9, 1000)):
        for index in range(20):
            label = index % 2
            signal = 1 + 3 * label + (index % 3) / 10
            timestamp = datetime(2022, 9, day, tzinfo=timezone.utc) + timedelta(seconds=index)
            values = [np.log1p(signal), index % 4, signal * 2, signal,
                      index % 3, index % 2, signal / 2]
            rows.append([f"{split}-{index}", timestamp.isoformat().replace("+00:00", "Z"),
                         str(label), split, *(float(value + offset) for value in values)])
    return rows


class MetricTests(unittest.TestCase):
    def test_average_precision_groups_ties_but_review_order_is_stable(self):
        scores = np.array([0.8, 0.8, 0.4, 0.1])
        result = ranking_metrics(np.array([1, 0, 1, 0]), scores)
        # AP = (1/2)*(1/2) + (1/2)*(2/3), not row-wise tie interpolation.
        self.assertAlmostEqual(result["average_precision"], 7 / 12)
        self.assertEqual(result["review_count"], 1)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 0)
        self.assertEqual(result["false_negatives"], 1)
        self.assertEqual(result["precision"], 1)
        self.assertEqual(result["recall"], 0.5)
        reversed_labels = ranking_metrics(np.array([0, 1, 1, 0]), scores)
        self.assertAlmostEqual(reversed_labels["average_precision"], 7 / 12)
        self.assertEqual(reversed_labels["true_positives"], 0)

    def test_review_budget_rounds_up_and_confusion_counts_reconcile(self):
        labels = np.array([1] + [0] * 20)
        scores = np.array([0.9, 0.9] + [0.1] * 19)
        result = ranking_metrics(labels, scores, budget=0.05)
        self.assertEqual(result["review_count"], 2)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 0)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1)

    def test_invalid_budgets_are_rejected(self):
        for budget in (0, -0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                ranking_metrics(np.array([0, 1]), np.array([0.1, 0.9]), budget)

    def test_invalid_metric_arrays_are_rejected(self):
        invalid = [([], []), ([0, 1], [0.1]), ([[0, 1]], [[0.1, 0.9]]),
                   ([0, 0], [0.1, 0.2]), ([1, 1], [0.1, 0.2]),
                   ([0, 2], [0.1, 0.2]), ([0, 1], [float("nan"), 0.2]),
                   ([0, 1], [0.1, float("inf")])]
        for labels, scores in invalid:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                ranking_metrics(np.array(labels), np.array(scores))

    def test_winner_uses_validation_ap_and_alphabetical_tie_break(self):
        reports = {name: {"average_precision": 0.1} for name in CANDIDATES}
        reports["logistic_amount"]["average_precision"] = 0.7
        reports["forest_amount"]["average_precision"] = 0.7
        self.assertEqual(select_winner(reports), "forest_amount")


class TrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.input_path = self.directory / "features.csv"
        self.enterContext(warnings.catch_warnings())
        warnings.filterwarnings("ignore", message=r"max_samples \(256\) is greater than the total number of samples.*",
                                category=UserWarning)

    def write_rows(self, rows=None, header=None):
        with self.input_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER if header is None else header)
            writer.writerows(fixture_rows() if rows is None else rows)
        return self.input_path

    def test_loader_has_exact_seven_feature_matrix_and_preserves_row_metadata(self):
        rows = fixture_rows()
        data = load_dataset(self.write_rows(rows))
        self.assertEqual(data["X"].shape, (60, 7))
        np.testing.assert_array_equal(data["X"], np.asarray([row[4:] for row in rows]))
        np.testing.assert_array_equal(data["y"], np.asarray([int(row[2]) for row in rows]))
        self.assertEqual(data["ids"], [row[0] for row in rows])
        self.assertTrue(all(timestamp.utcoffset() == timedelta(0) for timestamp in data["timestamps"]))

    def test_loader_rejects_missing_duplicate_extra_or_reordered_columns(self):
        cases = [HEADER[:-1], HEADER + ["label_copy"], HEADER[:-1] + [FEATURE_NAMES[0]],
                 HEADER[:4] + list(reversed(FEATURE_NAMES))]
        for header in cases:
            with self.subTest(header=header), self.assertRaises(ValueError):
                load_dataset(self.write_rows(header=header))

    def test_loader_rejects_nonfinite_negative_and_missing_feature_values(self):
        for value in ("nan", "inf", "-0.01", "", "not-a-number"):
            rows = fixture_rows()
            rows[0][4] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_dataset(self.write_rows(rows))

    def test_loader_rejects_bad_ids_labels_dates_splits_and_single_class_partitions(self):
        mutations = [
            lambda rows: rows[1].__setitem__(0, rows[0][0]),
            lambda rows: rows[0].__setitem__(0, ""),
            lambda rows: rows[0].__setitem__(2, "2"),
            lambda rows: rows[0].__setitem__(1, "2022-09-01T00:00:00"),
            lambda rows: rows[0].__setitem__(1, "not-a-date"),
            lambda rows: rows[0].__setitem__(3, "test"),
            lambda rows: rows[-1].__setitem__(1, "2022-09-11T00:00:00Z"),
            lambda rows: rows.reverse(),
            lambda rows: [row.__setitem__(2, "0") for row in rows if row[3] == "validation"],
        ]
        for index, mutate in enumerate(mutations):
            rows = fixture_rows()
            mutate(rows)
            with self.subTest(case=index), self.assertRaises(ValueError):
                load_dataset(self.write_rows(rows))

    def test_candidates_and_isolation_forest_score_orientation(self):
        models = make_models()
        self.assertEqual(set(models), CANDIDATES)
        self.assertIsNone(models["amount_rule"])
        data = load_dataset(self.write_rows())
        train = data["X"][np.asarray(data["splits"]) == "train"]
        model = models["isolation_history"]
        model.fit(train)
        np.testing.assert_allclose(score_model("isolation_history", model, train), -model.score_samples(train))
        np.testing.assert_array_equal(score_model("amount_rule", None, train), train[:, 0])

    def test_scoring_rejects_wrong_matrix_shapes_nonfinite_values_and_unknown_candidates(self):
        invalid = [np.ones(7), np.ones((2, 6)), np.full((2, 7), np.nan),
                   np.full((2, 7), np.inf), -np.ones((2, 7))]
        for matrix in invalid:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                score_model("amount_rule", None, matrix)
        with self.assertRaises(ValueError):
            score_model("not-a-candidate", None, np.ones((2, 7)))

    def test_cli_artifacts_reload_and_score_in_a_fresh_python_process(self):
        self.write_rows()
        output = self.directory / "cli"
        environment = os.environ | {"PYTHONDONTWRITEBYTECODE": "1"}
        trained = subprocess.run(
            [sys.executable, "-m", "src.train", "--input", str(self.input_path), "--output-dir", str(output)],
            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(trained.returncode, 0, trained.stderr)
        check = """
import json, sys
from pathlib import Path
import joblib
import numpy as np
from src.train import score_model
directory = Path(sys.argv[1])
models = joblib.load(directory / 'models.joblib')
report = json.loads((directory / 'results.json').read_text())
with np.load(directory / 'scores.npz') as arrays:
    for name, model in models.items():
        score = score_model(name, model, arrays['X_test'])
        np.testing.assert_allclose(score, arrays['test_' + name], rtol=0, atol=1e-12)
assert report['selected_model'] in models
print('fresh-process inference matches saved scores')
"""
        loaded = subprocess.run(
            [sys.executable, "-c", check, str(output)], cwd=ROOT, env=environment,
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(loaded.returncode, 0, loaded.stderr)
        self.assertIn("fresh-process inference matches saved scores", loaded.stdout)

    def test_small_runs_are_train_only_reproducible_and_independent_of_test_labels(self):
        self.write_rows()
        original = load_dataset(self.input_path)
        train_mask = np.asarray(original["splits"]) == "train"
        train_x = original["X"][train_mask]
        isolation_labels = []
        actual_fit = IsolationForest.fit

        def record_fit(estimator, matrix, y=None, *args, **kwargs):
            isolation_labels.append(y)
            self.assertEqual(len(matrix), len(train_x))
            return actual_fit(estimator, matrix, y, *args, **kwargs)

        first_dir = self.directory / "first"
        with patch.object(IsolationForest, "fit", record_fit):
            first = run_experiment(self.input_path, first_dir)
        self.assertTrue(isolation_labels)
        self.assertTrue(all(labels is None for labels in isolation_labels))
        first_models = joblib.load(first_dir / "models.joblib")
        self.assertEqual(set(first_models), CANDIDATES)
        for name in ("logistic_amount", "logistic_history"):
            model = first_models[name]
            transformed = model.named_steps["transform"].transform(train_x)
            scaler = model.named_steps["scale"]
            self.assertEqual(scaler.n_samples_seen_, len(train_x))
            np.testing.assert_allclose(scaler.mean_, transformed.mean(axis=0), rtol=0, atol=1e-12)
            full_mean = model.named_steps["transform"].transform(original["X"]).mean(axis=0)
            self.assertFalse(np.allclose(scaler.mean_, full_mean))

        second_dir = self.directory / "repeat"
        second = run_experiment(self.input_path, second_dir)
        self.assertEqual(first["selected_model"], second["selected_model"])
        self.assertEqual(first["validation"], second["validation"])
        self.assertEqual(first["test"], second["test"])

        changed = fixture_rows()
        for row in changed:
            if row[3] == "test":
                row[2] = str(1 - int(row[2]))
                row[4:] = [value + 5000 for value in row[4:]]
        self.write_rows(changed)
        third_dir = self.directory / "different-test"
        third = run_experiment(self.input_path, third_dir)
        self.assertEqual(first["selected_model"], third["selected_model"])
        self.assertEqual(first["validation"], third["validation"])
        common_x = original["X"]
        for directory in (second_dir, third_dir):
            models = joblib.load(directory / "models.joblib")
            for name in CANDIDATES:
                with self.subTest(run=directory.name, candidate=name):
                    np.testing.assert_allclose(score_model(name, first_models[name], common_x),
                                               score_model(name, models[name], common_x), rtol=0, atol=1e-12)
        self.assertEqual(json.loads((first_dir / "results.json").read_text()), first)
        with np.load(first_dir / "scores.npz") as scores:
            for split in ("validation", "test"):
                self.assertEqual(scores["X_" + split].shape, (20, 7))
                self.assertEqual(scores["y_" + split].shape, (20,))
                for name in CANDIDATES:
                    self.assertEqual(scores[split + "_" + name].shape, (20,))


if __name__ == "__main__":
    unittest.main()
