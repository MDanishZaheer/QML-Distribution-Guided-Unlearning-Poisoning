"""Synthetic protocol and recovery checks; no MNIST experiments are executed."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile
import hashlib
import unittest

import matplotlib
matplotlib.use("Agg")  # Headless figures; GUI objects must not be collected by fit workers.
import numpy as np

from src import artifacts, experiment_stages as stages
from src.ablations import ablation_execution, benchmark_execution, plan_table, run_study
from src.checkpoint_storage import checkpoint_status
from src.config import DEFAULT_SEED, CircuitConfig, DataConfig, TrainingConfig, UnlearningConfig
from src.data.dataset import DatasetSplits, ImageDataset
from src.data.feature_randomization import (FeaturePoisonConfig, SignedFeatureDataset, UnitInputPreprocessor,
                                           experiment_config, feature_variant, unit_inputs)
from src.evaluation.ablation_analysis import study_tables as ablation_tables
from src.evaluation.fresh_analysis import (comparison_figures, download_bundle, export_study,
                                          input_diagnostics, plot_feature_examples, study_tables, detailed_figures)
from src.fresh_study import (FreshStudy, base_config, benchmark_problem, condition_configs,
                              fresh_execution, run_fresh_study, study_plan)
from src.models.parallel_aer import AerExecution, AerBatchService
from src.noise_study import load_noise_inputs
from src.noisy_training import (load_training_problems, run_noisy_training, noisy_training_execution,
                                training_plan)
from src.models.noise import NoiseSetting


def fixture():
    rng = np.random.default_rng(DEFAULT_SEED)
    parts = {}
    for name, count in (("train", 8), ("validation", 2), ("test", 2)):
        labels = np.repeat([0, 1, 2], count)
        parts[name] = ImageDataset(rng.integers(0, 256, (len(labels), 28, 28), dtype=np.uint8), labels,
                                  np.array([f"synthetic_{name}_{i}" for i in range(len(labels))]))
    training = TrainingConfig(steps=2, batch_size=2, evaluate_every=1)
    base = base_config()
    config = replace(base, data=DataConfig(train_per_class=8, split_strategy="stratified_kfold"),
                     circuit=CircuitConfig(reuploads=1, ansatz_reps=1),
                     backend=replace(base.backend, device="CPU", max_parallel_threads=1),
                     classical_baselines=False,
                     training=training, unlearning=UnlearningConfig(training=training))
    return DatasetSplits(**parts), config


class FeatureRandomizationTests(unittest.TestCase):
    def setUp(self):
        self.splits, self.config = fixture()
        self.execution = AerExecution(device="CPU", workers=2, threads=1, circuit_batch_size=2048)

    def test_real_normalized_replacement_nested_rates_and_record_streams(self):
        original = self.splits.train
        before = original.fingerprint()
        small = feature_variant(original, FeaturePoisonConfig(rate=.25))
        large = feature_variant(original, FeaturePoisonConfig(rate=.5))
        self.assertEqual(small.is_poisoned.sum(), 2)
        self.assertEqual(large.is_poisoned.sum(), 4)
        np.testing.assert_array_equal(small.observed.labels, original.labels)
        np.testing.assert_array_equal(small.observed.sample_ids, original.sample_ids)
        np.testing.assert_array_equal(small.observed.images[~small.is_poisoned], original.images[~small.is_poisoned])
        self.assertTrue(np.all(original.labels[large.is_poisoned] == 2))
        self.assertTrue(np.all(large.is_poisoned[small.is_poisoned]))
        np.testing.assert_array_equal(small.observed.images[small.is_poisoned], large.observed.images[small.is_poisoned])
        vectors = large.observed.images[large.is_poisoned].reshape(4, -1) / 255
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-14)
        self.assertLess(vectors.min(), 0)
        # Independent reproduction of a recorded stream and normalized Gaussian law.
        index = np.flatnonzero(large.is_poisoned)[0]
        token = hashlib.sha256(str(original.sample_ids[index]).encode()).digest()
        rng = np.random.default_rng(np.random.SeedSequence([DEFAULT_SEED, 6304,
            int.from_bytes(token[:4], "little"), int.from_bytes(token[4:8], "little")]))
        expected = rng.normal(0, .1, 784)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(vectors[0], expected, atol=1e-16)
        self.assertEqual(before, original.fingerprint())
        with self.assertRaises(ValueError):
            ImageDataset(large.observed.images, original.labels, original.sample_ids)
        self.assertIsInstance(large.observed.subset(large.is_poisoned), SignedFeatureDataset)

    def test_pca_uses_only_unit_training_vectors_and_config_roundtrip(self):
        config = replace(self.config, poison=FeaturePoisonConfig(rate=.5))
        variant = feature_variant(self.splits.train, config.poison)
        restored = experiment_config(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())
        processor = UnitInputPreprocessor().fit(variant.observed.images, variant.observed.sample_ids)
        np.testing.assert_allclose(processor.pca.mean_, unit_inputs(variant.observed.images).mean(0), atol=1e-14)
        before = processor.pca.components_.copy()
        features = processor.transform(self.splits.test.images)
        self.assertEqual(features.shape, (6, 4))
        self.assertTrue(np.all((features >= 0) & (features <= np.pi)))
        np.testing.assert_array_equal(processor.pca.components_, before)
        self.assertFalse(np.intersect1d(processor.fit_sample_ids_, self.splits.test.sample_ids).size)
        np.testing.assert_array_equal(unit_inputs(np.zeros((1, 28, 28))), np.zeros((1, 784)))

    def test_planned_five_fold_counts_and_probe_are_not_trained_parents(self):
        plan = study_plan()
        self.assertEqual(len(plan), 265)
        self.assertEqual((plan.stage == "supervised").sum(), 115)
        self.assertEqual((plan.stage == "unlearning").sum(), 150)
        self.assertEqual(plan.condition.nunique(), 15)
        self.assertEqual(set(plan.fold), set(range(5)))
        problem = benchmark_problem(self.config, self.splits)
        self.assertIn("untrained_probe", str(problem.parent["path"]))
        with TemporaryDirectory() as directory:
            selected, rows, _ = benchmark_execution(problem, directory, candidates=[self.execution],
                require_gpu_device=False, probe_steps=1, minimum_workers=2)
            self.assertEqual(selected, self.execution)
            self.assertEqual(rows.iloc[0].status, "passed")
            self.assertFalse((Path(directory) / "results/runs").exists())

    def test_fresh_all_parents_cached_ablation_noise_inputs_and_portable_results(self):
        import matplotlib.pyplot as plt
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            study, index = run_fresh_study(root, {0: self.splits}, self.execution, base=self.config, rates=(.25, .5))
            self.assertEqual(len(study.runs), 18)
            self.assertFalse((root / "results/runs/supervised_models").exists())
            problems = study.ablation_problems()
            self.assertEqual(len(plan_table(problems)), 60)
            for problem in problems:
                retained = artifacts.load_arrays(problem.retrained["path"] / "features.npz")
                np.testing.assert_array_equal(retained["train_ids"], np.delete(problem.arrays["train_ids"], problem.references.forget_indices))
                if problem.config.poison.kind == "feature_randomization":
                    self.assertEqual(len(problem.references.forget_indices), 8)
                    clean = next(p for p in problems if p.name == "clean")
                    self.assertEqual(problem.retrained["path"], clean.retrained["path"])
            digest = {p: artifacts._file_digest(p) for path in study.runs for p in path.rglob("*") if p.is_file()}
            with patch.object(stages, "fit_vqc", side_effect=AssertionError("Completed models must not fit")):
                again, same_index = run_fresh_study(root, {0: self.splits}, self.execution, base=self.config, rates=(.25, .5))
            self.assertEqual(index, same_index)
            self.assertEqual(digest, {p: artifacts._file_digest(p) for p in digest})
            models, bindings = load_noise_inputs(again, folds=(0,))
            self.assertEqual(len(models), 18)
            self.assertEqual(len(bindings), 20)
            noisy, _, _ = load_training_problems(again, folds=(0,))
            self.assertEqual(len(training_plan(noisy)), 18)
            chosen = next(p for p in noisy if p.config.poison.kind == "feature_randomization")
            selected_bindings = bindings[(bindings.condition == chosen.name) & (bindings.fold == chosen.fold)]
            selected_models = {key: models[key] for key in selected_bindings.run_id}
            with noisy_training_execution(root):
                noisy_index = run_noisy_training(study, [chosen], selected_models, selected_bindings,
                    self.execution, setting=NoiseSetting("combined", .01), allow_training=True)
            self.assertTrue(artifacts.load_json(noisy_index)["complete"])
            with ablation_execution(root):
                ablation_index = run_study(study, [problems[0]], self.execution, baseline_checks=True)
            _, tables = ablation_tables(ablation_index, root)
            self.assertTrue(tables["coverage"].completed.all())
            diagnostics, _ = input_diagnostics(condition_configs(self.config, (0,), (.25, .5)), {0: self.splits})
            self.assertTrue((diagnostics[diagnostics.attack == "feature_randomization"].mean_angle_l2 > 0).all())
            self.assertTrue((diagnostics[diagnostics.attack == "label_flip"].mean_angle_l2 == 0).all())
            tables = study_tables(study)
            figures = comparison_figures(tables)
            figures.update(detailed_figures(study, 0))
            export_study(study, tables, figures, diagnostics)
            example = plot_feature_examples(feature_variant(self.splits.train, FeaturePoisonConfig(rate=.5)))
            figures["examples"] = example
            for figure in figures.values():
                plt.close(figure)
            for name in ("README.md", "requirements.txt", "main_project.ipynb"):
                (root / name).write_text("synthetic export placeholder")
            bundle = download_bundle(study, [ablation_index])
            with ZipFile(bundle) as archive:
                self.assertIsNone(archive.testzip())
                names = archive.namelist()
                self.assertTrue(any(n.startswith("results/runs/fresh_supervised/") for n in names))
                self.assertTrue(any(n.startswith("results/runs/ablations/") for n in names))
                self.assertTrue(any(n.startswith("data/annotated/feature_randomization") for n in names))

    def test_new_classical_comparisons_corrupt_cache_and_changed_settings(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(self.config, classical_baselines=True)
            study, index = run_fresh_study(root, {0: self.splits}, self.execution, base=config, rates=(.5,))
            self.assertTrue(any(name.startswith("clean_logistic") for name in study_tables(study)["metrics"].model))
            original_paths = set(study.runs)
            changed = replace(config, training=replace(config.training, learning_rate=.03))
            revised, other_index = run_fresh_study(root, {0: self.splits}, self.execution, base=changed, rates=(.5,))
            self.assertNotEqual(index, other_index)
            self.assertFalse(original_paths & set(revised.runs))
            run = next(iter(study.runs))
            (run / "weights.npz").write_bytes(b"synthetic corrupt completed checkpoint")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                FreshStudy(root, index)

    def test_interruption_preserves_completed_models_and_resumes_optimizer(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            def stop(event):
                if event.get("condition") == "label_flip_25pct" and event.get("role") == "original" and event.get("step") == 1:
                    raise KeyboardInterrupt("synthetic runtime session expiry")
            with self.assertRaises(KeyboardInterrupt):
                run_fresh_study(root, {0: self.splits}, replace(self.execution, workers=1), base=self.config,
                                rates=(.25,), progress=stop)
            health = checkpoint_status(root)
            completed = [row for row in health if row["status"] == "complete"]
            unfinished = [row for row in health if row["status"] == "failed"]
            self.assertTrue(completed)
            self.assertTrue(unfinished)
            self.assertTrue(any(row["saved_step"] == 1 for row in unfinished))
            files = list((root / "results/runs/fresh_supervised").glob("*/manifest.json"))
            immutable = {file.parent / "weights.npz": artifacts._file_digest(file.parent / "weights.npz") for file in files
                         if artifacts.load_json(file)["status"] == "complete"}
            study, _ = run_fresh_study(root, {0: self.splits}, replace(self.execution, workers=1),
                                      base=self.config, rates=(.25,))
            self.assertEqual(immutable, {p: artifacts._file_digest(p) for p in immutable})
            resumed = [artifacts.load_json(p / "manifest.json") for p in study.runs]
            self.assertTrue(any(row.get("resumed") for row in resumed))
            with TemporaryDirectory() as other:
                full, _ = run_fresh_study(Path(other), {0: self.splits}, replace(self.execution, workers=1),
                                         base=self.config, rates=(.25,))
                by_train = {r["manifest"]["identity"].get("train_fingerprint"): r for r in full.runs.values()
                            if r["manifest"]["identity"]["stage"] == "supervised_model"}
                for run in study.runs.values():
                    identity = run["manifest"]["identity"]
                    if identity["stage"] == "supervised_model":
                        np.testing.assert_allclose(run["weights"]["model"], by_train[identity["train_fingerprint"]]["weights"]["model"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
