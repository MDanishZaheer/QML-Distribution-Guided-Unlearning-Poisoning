"""New exact-25% inputs must use matched records, isolated paths and resumable fits."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile
import unittest

import numpy as np

from src import experiment_stages as stages
from src.ablation_conditions import quarter_configs, quarter_plan, prepare_quarter_problems, quarter_execution
from src.ablations import ablation_execution, plan_table, run_study
from src.artifacts import load_json, load_arrays
from src.checkpoints import CheckpointSession
from src.config import (DEFAULT_SEED, BackendConfig, DataConfig, ExperimentConfig,
                        TrainingConfig, UnlearningConfig)
from src.data.dataset import ImageDataset, DatasetSplits
from src.data.poisoning import apply_poisoning
from src.evaluation.ablation_analysis import study_tables, export_study, download_bundle
from src.evaluation.saved_results import protect_review
from src.execution_resources import parallel_worker_counts, aer_cpu_threads, execution_refinements
from src.models.parallel_aer import AerExecution, AerBatchService


class QuarterAblationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.execution = AerExecution(device="CPU", workers=2, threads=1, circuit_batch_size=2048)
        self.splits, self.conditions = {}, {}
        rng = np.random.default_rng(DEFAULT_SEED)
        for fold in (0, 1):
            parts = {}
            for name, per_class in (("train", 4), ("validation", 2), ("test", 2)):
                labels = np.repeat([0, 1, 2], per_class)
                images = rng.integers(0, 256, (len(labels), 28, 28), dtype=np.uint8)
                parts[name] = ImageDataset(images, labels,
                    np.array([f"synthetic_{fold}_{name}_{i}" for i in range(len(labels))]))
            splits = DatasetSplits(**parts)
            self.splits[fold] = splits
            training = TrainingConfig(steps=2, batch_size=2, evaluate_every=1)
            config = ExperimentConfig(data=DataConfig(fold=fold, n_folds=5, split_strategy="stratified_kfold", train_per_class=4),
                backend=BackendConfig(device="CPU", shots=None, method="statevector", max_parallel_threads=1),
                training=training, unlearning=UnlearningConfig(training=training), classical_baselines=False)
            self.conditions[str(fold)] = {"config": config.to_dict(), "split_fingerprint": splits.fingerprint()}
        self.review = SimpleNamespace(root=self.root, report_path=self.root / "original_report",
                                      source_fingerprint="historical", conditions=self.conditions,
                                      split_paths={fold: None for fold in self.splits})

    def test_exact_quarter_membership_and_all_five_fold_plan(self):
        config = quarter_configs(self.review, (0,))[0]
        observed = apply_poisoning(self.splits[0].train, config.poison)
        self.assertEqual(observed.is_poisoned.sum(), 1)
        np.testing.assert_array_equal(observed.original.images, observed.observed.images)
        self.assertTrue(np.all(observed.original.labels[observed.is_poisoned] == 0))
        self.assertTrue(np.all(observed.observed.labels[observed.is_poisoned] == 2))
        for fold in (2, 3, 4):
            copy = dict(self.conditions["0"])
            config = replace(ExperimentConfig.from_dict(copy["config"]), data=replace(config.data, fold=fold))
            copy["config"] = config.to_dict()
            self.review.conditions[str(fold)] = copy
            self.review.split_paths[fold] = None
        plan = quarter_plan(self.review, range(5))
        self.assertEqual(len(plan), 20)
        self.assertEqual(set(plan.fold), set(range(5)))
        self.assertTrue((plan.seed == DEFAULT_SEED).all())

    def test_new_inputs_cache_fixed_teachers_matched_retraining_and_portable_sweep(self):
        problems, index_file = prepare_quarter_problems(self.review, self.splits, self.execution, folds=(0, 1))
        self.assertEqual(len(problems), 2)
        for problem in problems:
            self.assertEqual(problem.baseline_origin, "prepared_25pct")
            self.assertEqual(len(problem.references.forget_indices), 5)
            arrays = load_arrays(problem.retrained["path"] / "features.npz")
            np.testing.assert_array_equal(arrays["train_ids"], np.delete(problem.arrays["train_ids"], problem.references.forget_indices))
            self.assertEqual(len(problem.references.calibration_indices), 5)
            self.assertFalse(np.intersect1d(problem.references.anchor_indices, problem.references.forget_indices).size)
        paths = [file for stage in ("ablation_inputs_supervised", "ablation_inputs_unlearning")
                 for file in (self.root / "results/runs" / stage).glob("*/manifest.json")]
        self.assertEqual(len(paths), 8)
        checksums = {path: sha256(path.read_bytes()).hexdigest() for path in paths}
        again, same_index = prepare_quarter_problems(self.review, self.splits, self.execution, folds=(0, 1))
        self.assertEqual(index_file, same_index)
        self.assertEqual([p.parent["path"] for p in problems], [p.parent["path"] for p in again])
        self.assertEqual(checksums, {path: sha256(path.read_bytes()).hexdigest() for path in paths})
        self.assertFalse((self.root / "results/runs/supervised_models").exists())
        with ablation_execution(self.root):
            study = run_study(self.review, problems, self.execution, baseline_checks=True, support_index=index_file)
        _, tables = study_tables(study, self.root)
        self.assertEqual(len(tables["coverage"]), 24)
        self.assertTrue(tables["coverage"].completed.all())
        self.assertEqual(set(tables["summary"].condition), {"label_flip_25pct"})
        self.assertIn("prepared_25pct", set(tables["summary"].status))
        export_study(study, self.root)
        # The synthetic temporary project needs only the files requested by the portable export.
        (self.root / "src/evaluation").mkdir(parents=True)
        (self.root / "src/models").mkdir()
        for name in ("src/ablations.py", "src/models/parallel_aer.py", "src/evaluation/ablation_analysis.py",
                     "src/checkpoint_storage.py", "main_project.ipynb", "README.md"):
            (self.root / name).write_text("synthetic export placeholder")
        bundle = download_bundle(study, self.root)
        with ZipFile(bundle) as archive:
            self.assertIsNone(archive.testzip())
            self.assertTrue(any(name.startswith("results/runs/ablation_inputs_supervised/") for name in archive.namelist()))
            self.assertTrue(any(name.startswith("data/annotated/") for name in archive.namelist()))

    def test_interrupted_supervised_fit_resumes_and_review_guard_is_restored(self):
        original = stages.fit_vqc
        # Interrupt both workers after the intended committed update. Otherwise
        # the faster worker can cancel its sibling at step 0, making this
        # test's two step-1 checkpoints depend on thread scheduling.
        committed = Barrier(self.execution.workers)
        def interrupted(*args, **kwargs):
            callback = kwargs.get("callback")
            def stop(row):
                if callback is not None:
                    callback(row)
                if row["step"] == 1:
                    committed.wait(timeout=30)
                    raise KeyboardInterrupt("synthetic interruption")
            kwargs["callback"] = stop
            return original(*args, **kwargs)
        with patch.object(stages, "fit_vqc", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                prepare_quarter_problems(self.review, self.splits, self.execution, folds=(0,))
        resumes = list((self.root / "results/runs/ablation_inputs_supervised").glob("*/checkpoints/model.resume.json"))
        self.assertTrue(resumes)
        self.assertTrue(all(load_json(path)["step"] == 1 for path in resumes))
        with protect_review() as guard:
            problems, _ = prepare_quarter_problems(self.review, self.splits, self.execution, folds=(0,), review_guard=guard)
            self.assertEqual(len(problems), 1)
            with self.assertRaisesRegex(RuntimeError, "Saved-results review blocks"):
                stages.fit_vqc()
        self.assertEqual(len(list((self.root / "results/runs/ablation_inputs_supervised").glob("*/manifest.json"))), 2)

    def test_scope_rejects_old_paths_and_expanded_execution_candidates_are_bounded(self):
        config = quarter_configs(self.review, (0,))[0]
        with AerBatchService(config.circuit, config.backend, self.execution) as service:
            with quarter_execution(self.root, service):
                with self.assertRaisesRegex(RuntimeError, "Only the separate 25%"):
                    CheckpointSession().open(self.root / "results/runs/supervised_models", {}, allow_execution=True)
        self.assertIn(16, parallel_worker_counts())
        self.assertIn(24, parallel_worker_counts())
        with patch("src.execution_resources.os.cpu_count", return_value=8):
            self.assertEqual(aer_cpu_threads(), 8)
            variants = execution_refinements(replace(self.execution, threads=8))
            self.assertEqual({v.threads for v in variants}, {1, 4, 8})
            self.assertTrue(all(v.workers == self.execution.workers for v in variants))

    def test_pixel_quarter_reuses_exact_historical_reference_without_mutating_it(self):
        config = replace(ExperimentConfig.from_dict(self.conditions["0"]["config"]), classical_baselines=True)
        splits = self.splits[0]
        reference = stages._supervised(config, splits, splits.train.subset(splits.train.labels != 2), True,
                                      root=self.root / "historical", allow_training=True, role="retrained_clean")
        before = {file: sha256(file.read_bytes()).hexdigest() for file in reference["path"].rglob("*") if file.is_file()}
        self.conditions["0"]["models"] = {"retrained_clean": reference}
        self.review.runs = {reference["path"].resolve(): reference}
        self.review._read_path = lambda path: path
        plan = quarter_plan(self.review, (0,), "pixel_patch")
        self.assertEqual(len(plan), 3)
        problems, index = prepare_quarter_problems(self.review, self.splits, self.execution, folds=(0,), attack="pixel_patch")
        problem = problems[0]
        self.assertEqual(problem.name, "pixel_patch_25pct")
        self.assertEqual(problem.retrained["path"], reference["path"])
        self.assertEqual(len(problem.references.forget_indices), 4)
        self.assertTrue(load_json(index)["conditions"]["0"]["retained_reference_reused"])
        self.assertEqual(len(list((self.root / "results/runs/ablation_inputs_supervised").glob("*/manifest.json"))), 1)
        self.assertEqual(before, {file: sha256(file.read_bytes()).hexdigest() for file in before})


if __name__ == "__main__":
    unittest.main()
