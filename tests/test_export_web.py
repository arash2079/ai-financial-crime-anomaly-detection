"""Exporter checks using a tiny artificial forest, not research-data fitting."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.export_web import export_forest, export_web, make_parity
from src.features import FEATURE_NAMES
from src.train import ranking_metrics


def exported_score(payload, features):
    """Read the public JSON format independently of the exporter."""
    values = np.asarray(features, dtype=np.float32)
    scores = []
    for tree in payload["trees"]:
        node = 0
        while tree["left"][node] != -1:
            feature = tree["feature"][node]
            node = tree["left"][node] if values[feature] <= tree["threshold"][node] else tree["right"][node]
        scores.append(tree["value"][node])
    return sum(scores) / len(scores)


class WebExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = np.arange(168, dtype=float).reshape(24, 7) / 10
        cls.y = np.asarray([index % 2 for index in range(24)])
        cls.model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42).fit(cls.X, cls.y)

    def test_json_tree_format_preserves_sklearn_predictions_and_provenance(self):
        payload = export_forest(self.model, "a" * 64, "b" * 64)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["model_name"], "forest_history")
        self.assertEqual(payload["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(payload["source_model_sha256"], "a" * 64)
        self.assertEqual(payload["training_input_sha256"], "b" * 64)
        self.assertEqual(len(payload["trees"]), 3)
        payload = json.loads(json.dumps(payload, allow_nan=False))
        expected = self.model.predict_proba(self.X)[:, 1]
        actual = [exported_score(payload, vector) for vector in self.X]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_export_rejects_unfitted_wrong_dimension_and_nonbinary_estimators(self):
        wrong_dimension = RandomForestClassifier(n_estimators=1, random_state=42).fit(self.X[:, :6], self.y)
        single_class = RandomForestClassifier(n_estimators=1, random_state=42).fit(self.X, np.zeros(len(self.X)))
        for invalid in (None, RandomForestClassifier(), wrong_dimension, single_class):
            with self.subTest(estimator=invalid), self.assertRaises(ValueError):
                export_forest(invalid, "a" * 64, "b" * 64)

    def test_eighty_synthetic_reference_vectors_match_sklearn(self):
        fixture = make_parity(self.model)
        self.assertEqual(len(fixture["cases"]), 80)
        self.assertIn("synthetic", fixture["description"].lower())
        features = np.asarray([case["features"] for case in fixture["cases"]])
        self.assertEqual(features.shape, (80, 7))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue((features >= 0).all())
        expected = self.model.predict_proba(features)[:, 1]
        np.testing.assert_allclose([case["expected"] for case in fixture["cases"]], expected, rtol=0, atol=1e-12)

    def test_web_package_hashes_budgets_and_public_report_are_consistent(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model_path, metrics_path, scores_path = [directory / name for name in ("models.joblib", "metrics.json", "scores.npz")]
            scores = self.model.predict_proba(self.X)[:, 1]
            joblib.dump({"forest_history": self.model}, model_path)
            report = {"selected_model": "forest_history", "feature_names": list(FEATURE_NAMES),
                      "input_sha256": "b" * 64, "counts": {"test": {"rows": len(self.y), "positives": int(self.y.sum())}},
                      "test": {"forest_history": ranking_metrics(self.y, scores)},
                      "artifacts": {"models": "/private/local/models.joblib"}}
            metrics_path.write_text(json.dumps(report), encoding="utf-8")
            np.savez(scores_path, X_test=self.X, y_test=self.y, test_forest_history=scores)
            output, parity_path = directory / "public", directory / "parity.json"
            summary = export_web(model_path, metrics_path, scores_path, output, parity_path)
            forest_bytes = (output / "forest.json").read_bytes()
            forest_hash = hashlib.sha256(forest_bytes).hexdigest()
            public = json.loads((output / "report.json").read_text())
            parity = json.loads(parity_path.read_text())
            self.assertEqual(summary["forest_sha256"], forest_hash)
            self.assertEqual(public["web_export"]["forest_json_sha256"], forest_hash)
            self.assertEqual(parity["forest_json_sha256"], forest_hash)
            self.assertNotIn("artifacts", public)
            self.assertNotIn("/private/local", json.dumps(public))
            self.assertEqual([entry["budget"] for entry in public["selected_review_budgets"]], [.01, .02, .05, .1, .2])
            self.assertEqual(public["selected_review_budgets"][2], report["test"]["forest_history"])


if __name__ == "__main__":
    unittest.main()
