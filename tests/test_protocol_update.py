"""Synthetic protocol, reporting and deferred cleanup checks; no MNIST runs."""

from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.artifacts import load_json, save_json
from src.checkpoints import matching_manifests, retire_pilot_checkpoints
from src.config import DEFAULT_SEED, BackendConfig, DataConfig, PoisonConfig
from src.data.dataset import ImageDataset, make_cv_splits
from src.data.poisoning import apply_poisoning
from src.evaluation.analysis import fold_summaries
from src.experiments import run_clean_vqc
from test_notebook_stages import fixtures


class PilotCleanupTests(unittest.TestCase):
    data = DataConfig(train_per_class=140, test_per_class=None, split_strategy="stratified_kfold")
    backend = BackendConfig(shots=None, method="statevector")

    def fake_run(self, root, name, *, compatible=False, stage="supervised_models", status="complete"):
        path = Path(root).resolve() / "results/runs" / stage / name
        config = {"seed": DEFAULT_SEED, "data": asdict(self.data if compatible else replace(
            self.data, train_per_class=100))}
        if stage != "classical_cv":
            config["backend"] = asdict(self.backend if compatible else replace(self.backend, shots=2048))
        manifest = {"status": status, "identity": {"config": config}, "created_utc": "2026-11-15"}
        save_json(path / "manifest.json", manifest)
        save_json(path / "metrics.json", {"pilot_accuracy": 0.5})
        save_json(path / "checkpoints/model.json", {"best_step": 1})
        save_json(path / "checkpoints/model.resume.json", {"step": 1})
        for filename in ("weights.npz", "checkpoints/model.npz", "classical_models.joblib",
                         "fold_0_models.joblib", "predictions.npz", "history.csv", "source.zip"):
            (path / filename).write_bytes(b"synthetic artifact")
        return path

    def clean(self, root):
        return retire_pilot_checkpoints(root, data_config=self.data,
                                       backend_config=self.backend, seed=DEFAULT_SEED)

    def test_removes_only_old_model_files_and_never_new_runs_on_rerun(self):
        with tempfile.TemporaryDirectory() as root:
            old = self.fake_run(root, "pilot")
            current = self.fake_run(root, "new", compatible=True)
            classical = self.fake_run(root, "classical_pilot", stage="classical_cv")
            save_json(old.parent / "active_checkpoints.json", {
                "old": {"run": old.name}, "current": {"run": current.name}})
            original = load_json(old / "manifest.json")
            journal = self.clean(root)
            self.assertEqual(len(journal["targets"]), 2)
            self.assertEqual(journal["status"], "complete")
            self.assertIn(original, [item["original_manifest"] for item in journal["targets"]])
            for run in (old, classical):
                self.assertFalse((run / "weights.npz").exists())
                self.assertFalse((run / "checkpoints/model.resume.json").exists())
                self.assertFalse((run / "fold_0_models.joblib").exists())
                self.assertEqual(load_json(run / "manifest.json")["status"], "retired")
                for kept in ("metrics.json", "checkpoints/model.json", "predictions.npz", "history.csv", "source.zip"):
                    self.assertTrue((run / kept).exists())
            self.assertEqual(matching_manifests(old.parent, original["identity"]), [])
            self.assertTrue((current / "weights.npz").exists())
            self.assertEqual(set(load_json(old.parent / "active_checkpoints.json")), {"current"})
            # Even an incompatible model created later is outside the frozen one-time plan.
            later = self.fake_run(root, "created_after_cleanup")
            self.assertEqual(self.clean(root), journal)
            self.assertTrue((later / "weights.npz").exists())

    def test_interrupted_cleanup_resumes_frozen_plan(self):
        with tempfile.TemporaryDirectory() as root:
            old = self.fake_run(root, "pilot", status="failed")
            unlink = Path.unlink
            def interrupt(file, *args, **kwargs):
                if file.name == "model.resume.json":
                    raise OSError("Simulated interrupted deletion")
                return unlink(file, *args, **kwargs)
            with patch.object(Path, "unlink", interrupt):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    self.clean(root)
            marker = Path(root) / "results/maintenance/mnist420_exact_v1.json"
            self.assertEqual(load_json(marker)["status"], "pending")
            new = self.fake_run(root, "new_after_interruption")
            result = self.clean(root)
            self.assertEqual(result["status"], "complete")
            self.assertFalse((old / "weights.npz").exists())
            self.assertTrue((new / "weights.npz").exists())

    def test_unsafe_paths_and_junctions_are_rejected_before_deletion(self):
        with tempfile.TemporaryDirectory() as root:
            old = self.fake_run(root, "pilot")
            junction = old / "checkpoints"
            with patch.object(Path, "is_junction", lambda path: path == junction):
                with self.assertRaisesRegex(ValueError, "links"):
                    self.clean(root)
            self.assertTrue((old / "weights.npz").exists())
            self.assertEqual(load_json(old / "manifest.json")["status"], "complete")
            marker = Path(root) / "results/maintenance/mnist420_exact_v1.json"
            data = {k: asdict(self.data)[k] for k in ("train_per_class", "test_per_class", "split_strategy",
                    "n_folds", "validation_fraction", "seed")}
            backend = {k: asdict(self.backend)[k] for k in ("shots", "method", "use_ibm_hardware",
                       "one_qubit_error", "two_qubit_error", "readout_error")}
            save_json(marker, {"status": "pending", "protocol": {"seed": DEFAULT_SEED, "data": data,
                "backend": backend}, "targets": [{"run": "../outside", "files": ["weights.npz"]}]})
            with self.assertRaisesRegex(ValueError, "Invalid run path"):
                self.clean(root)
            self.assertTrue((old / "weights.npz").exists())


class UpdatedProtocolTests(unittest.TestCase):
    def test_420_pool_has_fixed_disjoint_folds_and_nested_poison_rates(self):
        dataset = ImageDataset(np.zeros((420, 28, 28), dtype=np.uint8),
            np.repeat([0, 1, 2], 140), np.array([f"synthetic_{i}" for i in range(420)]))
        folds = make_cv_splits(dataset)
        test_ids = []
        for fold in folds:
            self.assertEqual(tuple(map(len, (fold.train, fold.validation, fold.test))), (268, 68, 84))
            self.assertEqual(np.bincount(fold.test.labels).tolist(), [28, 28, 28])
            self.assertFalse(set(fold.train.sample_ids) & set(fold.test.sample_ids))
            self.assertFalse(set(fold.train.sample_ids) & set(fold.validation.sample_ids))
            test_ids.extend(fold.test.sample_ids)
            for kind in ("pixel_patch", "label_flip"):
                previous = set()
                eligible = np.sum(fold.train.labels == (2 if kind == "pixel_patch" else 0))
                for rate in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
                    variant = apply_poisoning(fold.train, PoisonConfig(kind=kind, rate=rate))
                    selected = set(variant.observed.sample_ids[variant.is_poisoned])
                    self.assertEqual(len(selected), int(eligible * rate))
                    self.assertTrue(previous <= selected)
                    previous = selected
        self.assertEqual(len(set(test_ids)), 420)

    def test_clean_gold_supports_exact_expectations_and_cache_reuse(self):
        splits, config = fixtures()
        with tempfile.TemporaryDirectory() as root:
            args = dict(data_config=config.data, circuit_config=config.circuit,
                        backend_config=replace(config.backend, method="statevector"),
                        training_config=config.training, root=root)
            result = run_clean_vqc(splits, allow_training=True, **args)
            resources = load_json(result["path"] / "resources.json")
            self.assertEqual(resources["expectation_mode"], "exact")
            self.assertEqual(resources["resources"]["shots"], 0)
            with patch("src.experiments.fit_vqc", side_effect=AssertionError("Must reuse exact checkpoint")):
                cached = run_clean_vqc(splits, allow_training=True, **args)
            np.testing.assert_array_equal(result["weights"]["clean"], cached["weights"]["clean"])

    def test_fold_statistics_use_per_fold_values_and_detect_missing_or_duplicate_folds(self):
        common = {"seed": DEFAULT_SEED, "attack": "label_flip", "poison_rate": 0.5, "model": "poisoned_guided"}
        tables = {
            "metrics": pd.DataFrame([{**common, "fold": i, "retained_accuracy": value}
                                      for i, value in enumerate([0.5, 1.0])]),
            "per_class": pd.DataFrame([{**common, "fold": i, "digit": 0, "recall": value}
                                        for i, value in enumerate([0.25, 0.75])]),
            "retraining": pd.DataFrame([{**common, "fold": i, "full_kl_mean": value}
                                         for i, value in enumerate([0.1, 0.3])]),
            "effects": pd.DataFrame([{**common, "fold": i, "method": "guided", "metric": "retained_accuracy",
                                       "additional_change_under_poisoning": value}
                                      for i, value in enumerate([-0.2, 0.1])]),
        }
        summary = fold_summaries(tables, expected_folds=range(5))
        scores = summary["metrics"].set_index("metric")
        self.assertAlmostEqual(scores.loc["retained_accuracy", "mean"], 0.75)
        self.assertAlmostEqual(scores.loc["retained_accuracy", "fold_sd"], np.std([0.5, 1.0], ddof=1))
        self.assertAlmostEqual(scores.loc["recall_digit_0", "mean"], 0.5)
        self.assertAlmostEqual(scores.loc["full_kl_mean", "mean"], 0.2)
        self.assertFalse(scores["complete_folds"].any())
        self.assertAlmostEqual(summary["effects"]["mean"].iloc[0], -0.05)
        complete = fold_summaries(tables, expected_folds=range(2))
        self.assertTrue(complete["metrics"]["complete_folds"].all())
        tables["metrics"] = pd.concat([tables["metrics"], tables["metrics"].iloc[:1]])
        with self.assertRaisesRegex(ValueError, "Duplicate fold"):
            fold_summaries(tables, expected_folds=range(5))


if __name__ == "__main__":
    unittest.main()
