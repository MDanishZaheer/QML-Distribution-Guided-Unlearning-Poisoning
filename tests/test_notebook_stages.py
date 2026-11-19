"""Synthetic checks for notebook stage boundaries and exact-record references."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from src.artifacts import load_arrays, load_run
from src.config import (DEFAULT_SEED, BackendConfig, CircuitConfig, DataConfig,
                        ExperimentConfig, PoisonConfig, TrainingConfig, UnlearningConfig)
from src.data.dataset import DatasetSplits, ImageDataset
from src.data.poisoning import apply_poisoning
from src.data.preprocessing import ImagePreprocessor
from src.evaluation.analysis import phase1_tables, accuracy_interval
from src.experiment_stages import (train_condition, unlearn_condition,
                                   restore_stage_model, poison_feature_effect)
from src.experiments import run_clean_vqc
from src.training import fit_vqc


def fixtures():
    random = np.random.default_rng(DEFAULT_SEED)
    def dataset(name, count):
        labels = np.repeat([0, 1, 2], count)
        images = random.integers(0, 256, (len(labels), 28, 28), dtype=np.uint8)
        return ImageDataset(images, labels, np.array([f"{name}_{i}" for i in range(len(labels))]))
    splits = DatasetSplits(dataset("train", 6), dataset("validation", 3), dataset("test", 3))
    training = TrainingConfig(steps=1, batch_size=2, evaluate_every=1)
    config = ExperimentConfig(
        data=DataConfig(split_strategy="stratified_kfold", train_per_class=6),
        circuit=CircuitConfig(reuploads=1, ansatz_reps=1),
        backend=BackendConfig(shots=None, device="CPU", max_parallel_threads=1),
        training=training, unlearning=UnlearningConfig(training=training),
        poison=PoisonConfig(kind="label_flip", rate=0.5), classical_baselines=False,
    )
    return splits, config


class NotebookStageTests(unittest.TestCase):
    def test_disabled_stages_do_not_train_or_create_results(self):
        splits, config = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            with patch("src.experiment_stages.fit_vqc", side_effect=AssertionError("Training disabled")):
                result = train_condition(config, splits, root=temporary)
                self.assertTrue(all(value is None for value in result["models"].values()))
                result = unlearn_condition(result, splits, root=temporary)
                self.assertFalse(result["unlearned"])
                self.assertFalse((Path(temporary) / "results").exists())

    def test_training_then_unlearning_matches_ids_and_keeps_parents(self):
        splits, config = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            trained = train_condition(config, splits, allow_training=True, root=temporary)
            self.assertFalse(trained["unlearned"])
            parent_weights = {role: run["weights"]["model"].copy() for role, run in trained["models"].items()}
            variant = apply_poisoning(splits.train, config.poison)
            forget = variant.observed.labels == 2
            retained = load_arrays(trained["models"]["retrained_poisoned"]["path"] / "features.npz")
            np.testing.assert_array_equal(retained["train_ids"], splits.train.sample_ids[~forget])
            np.testing.assert_array_equal(retained["train_labels"], splits.train.labels[~forget])
            processor = joblib.load(trained["models"]["retrained_poisoned"]["path"] / "preprocessing.joblib")
            np.testing.assert_array_equal(processor.fit_sample_ids_, retained["train_ids"])
            with patch("src.experiment_stages.fit_vqc", side_effect=AssertionError("No supervised refit")):
                result = unlearn_condition(trained, splits, allow_execution=True, root=temporary)
                cached = unlearn_condition(trained, splits, root=temporary)
            for role, run in result["unlearned"].items():
                self.assertEqual(run["path"], cached["unlearned"][role]["path"])
                case = role.split("_")[0]
                refs = load_arrays(run["path"] / "references.npz")
                np.testing.assert_array_equal(refs["original_weights"], parent_weights[case])
                self.assertFalse(set(refs["anchor_ids"]) & set(refs["forget_ids"]))
                self.assertTrue(set(refs["calibration_ids"]) <= set(refs["forget_ids"]))
                self.assertFalse(set(refs["anchor_ids"]) & set(splits.test.sample_ids))
                self.assertEqual(run["predictions"]["model__test"].shape, (9, 3))
            for role, run in trained["models"].items():
                np.testing.assert_array_equal(load_run(run["path"])["weights"]["model"], parent_weights[role])
            model, processor, weights = restore_stage_model(result["unlearned"]["poisoned_guided"])
            np.testing.assert_allclose(model.predict_proba(processor.transform(splits.test.images), weights),
                                       result["unlearned"]["poisoned_guided"]["predictions"]["model__test"], atol=1e-12)
            tables = phase1_tables({"label_flip_50pct": result})
            self.assertEqual(len(tables["targets"]), 4)
            self.assertEqual(len(tables["effects"]), 6)
            self.assertEqual(len(tables["metrics"]), 8)
            self.assertTrue((tables["metrics"].accuracy_ci95_low <= tables["metrics"].accuracy + 1e-12).all())
            # A changed forget policy needs a different retained-data reference.
            incompatible = replace(config, unlearning=replace(config.unlearning, forget_definition="original"))
            with self.assertRaises(AssertionError):
                unlearn_condition(trained, splits, config=incompatible, root=temporary)

    def test_cache_reuses_clean_and_identical_retraining_across_rates(self):
        splits, config = fixtures()
        config = replace(config, poison=PoisonConfig(kind="pixel_patch", rate=0.5))
        with tempfile.TemporaryDirectory() as temporary:
            first = train_condition(config, splits, allow_training=True, root=temporary)
            second = train_condition(replace(config, poison=replace(config.poison, rate=1.0)), splits,
                                     allow_training=True, root=temporary)
            for role in ("clean", "retrained_clean", "retrained_poisoned"):
                self.assertEqual(first["models"][role]["path"], second["models"][role]["path"])
            self.assertEqual(first["models"]["retrained_clean"]["path"], first["models"]["retrained_poisoned"]["path"])
            self.assertNotEqual(first["models"]["poisoned"]["path"], second["models"]["poisoned"]["path"])
            first_unlearned = unlearn_condition(first, splits, allow_execution=True, root=temporary)
            second_unlearned = unlearn_condition(second, splits, root=temporary)
            self.assertEqual(first_unlearned["unlearned"]["clean_guided"]["path"],
                             second_unlearned["unlearned"]["clean_guided"]["path"])
            self.assertIsNone(second_unlearned["unlearned"]["poisoned_guided"])

    def test_import_existing_gold_and_reject_mismatched_settings(self):
        splits, config = fixtures()
        config = replace(config, backend=replace(config.backend, shots=64), poison=PoisonConfig())
        with tempfile.TemporaryDirectory() as temporary:
            gold = run_clean_vqc(splits, data_config=config.data, circuit_config=config.circuit,
                backend_config=config.backend, training_config=config.training, seed=config.seed,
                allow_training=True, root=temporary)
            with patch("src.experiment_stages.fit_vqc", wraps=fit_vqc) as fit:
                imported = train_condition(config, splits, gold_run=gold, allow_training=True, root=temporary)
                # Only the retained-data reference is fitted; the clean gold is copied.
                self.assertEqual(fit.call_count, 1)
            np.testing.assert_array_equal(imported["models"]["clean"]["weights"]["model"], gold["weights"]["clean"])
            np.testing.assert_array_equal(imported["models"]["clean"]["predictions"]["model__test"], gold["predictions"]["clean__test"])
            self.assertEqual(imported["models"]["clean"]["path"], imported["models"]["poisoned"]["path"])
            with self.assertRaisesRegex(ValueError, "incompatible training"):
                train_condition(replace(config, training=replace(config.training, steps=2)), splits,
                                gold_run=gold, root=temporary)

    def test_feature_diagnostic_is_label_invariant_and_bounded(self):
        splits, _ = fixtures()
        processor = ImagePreprocessor().fit(splits.train.images)
        variants = {kind: apply_poisoning(splits.train, PoisonConfig(kind=kind, rate=1.0))
                    for kind in ("pixel_patch", "label_flip")}
        before = splits.fingerprint()
        table = poison_feature_effect(variants, processor).set_index("condition")
        self.assertEqual(table.loc["label_flip", "mean_angle_l2"], 0)
        self.assertEqual(table.loc["label_flip", "unchanged_angle_records"], 6)
        self.assertGreater(table.loc["pixel_patch", "mean_pixel_l2"], 0)
        self.assertTrue(0 <= table.loc["pixel_patch", "mean_projected_energy_fraction"] <= 1)
        self.assertEqual(splits.fingerprint(), before)
        self.assertEqual(accuracy_interval(0, 0), (np.nan, np.nan))

    def test_classical_and_shuffle_control_are_saved_separately(self):
        splits, config = fixtures()
        config = replace(config, poison=PoisonConfig(), classical_baselines=True, label_shuffle_control=True)
        with tempfile.TemporaryDirectory() as temporary:
            result = train_condition(config, splits, allow_training=True, root=temporary)
            original = result["models"]["clean"]
            self.assertEqual(set(original["metrics"]["models"]),
                             {"model", "majority", "logistic", "linear_svm", "rbf_svm"})
            self.assertTrue((original["path"] / "classical_models.joblib").is_file())
            shuffled = result["models"]["label_shuffle"]
            arrays = load_arrays(shuffled["path"] / "features.npz")
            np.testing.assert_array_equal(arrays["train_ids"], splits.train.sample_ids)
            np.testing.assert_array_equal(np.sort(arrays["train_labels"]), np.sort(splits.train.labels))
            np.testing.assert_array_equal(arrays["validation_labels"], splits.validation.labels)
            self.assertNotEqual(original["path"], shuffled["path"])


if __name__ == "__main__":
    unittest.main()
