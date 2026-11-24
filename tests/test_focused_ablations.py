"""Synthetic checks of shared Aer batching, isolated RNG/checkpoints and OFAT reporting."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ablations import (AblationProblem, DEFAULTS, SWEEPS, _run_variant, ablation_execution,
                           parameter_settings, plan_table, validate_execution)
from src.ablations import benchmark_execution, run_study
from src.artifacts import (create_run, finish_run, load_json, load_run, save_arrays, save_json,
                           save_training_result)
from src.checkpoints import CheckpointSession
from src.checkpoint_storage import CheckpointMirror
from src.config import (DEFAULT_SEED, BackendConfig, CircuitConfig, ExperimentConfig, TrainingConfig,
                        UnlearningConfig)
from src.evaluation.ablation_analysis import comparison_figures, study_tables, export_study, download_bundle
from src.evaluation.metrics import classification_metrics, compare_to_retraining, forgetting_changes
from src.evaluation.saved_results import protect_review
from src.models.parallel_aer import AerBatchService, AerExecution
from src.models.vqc import VQC
from src.unlearning import methods
from src.unlearning.targets import prepare_references


def fixture(root):
    rng = np.random.default_rng(DEFAULT_SEED)
    features, labels = rng.uniform(0, np.pi, (9, 4)), np.repeat([0, 1, 2], 3)
    ids = np.array([f"synthetic_{i}" for i in range(len(features))])
    backend = BackendConfig(shots=None, method="statevector", device="CPU", max_parallel_threads=1)
    training = TrainingConfig(steps=4, batch_size=2, evaluate_every=1)
    config = ExperimentConfig(backend=backend, unlearning=UnlearningConfig(training=training))
    model = VQC(config.circuit, backend)
    weights = rng.normal(0, 0.2, 72)
    refs = prepare_references(model, features, labels, labels == 2, weights, config.unlearning)
    probabilities = model.predict_proba(features, weights)
    parent_identity = {"stage": "synthetic_parent", "config": config.to_dict(),
                       "code_fingerprint": "synthetic", "environment": {}}
    path, _ = create_run(root / "parents", parent_identity)
    save_json(path / "config.json", config.to_dict())
    save_arrays(path / "weights.npz", model=weights)
    save_arrays(path / "features.npz", train=features, test=features, train_labels=labels, train_ids=ids)
    save_arrays(path / "predictions.npz", model__test=probabilities, test_labels=labels, test_sample_ids=ids)
    metrics = classification_metrics(labels, probabilities)
    saved_metrics = {"models": {"model": metrics}, "target": [0.5, 0.5, 0],
                     "reference_audit": {"n_forget": 3}, "changes": forgetting_changes(metrics, metrics),
                     "retraining_comparison": compare_to_retraining(labels, probabilities, probabilities)}
    save_json(path / "metrics.json", saved_metrics)
    save_json(path / "preprocessing.json", {"synthetic": True})
    (path / "preprocessing.joblib").write_bytes(b"never unpickled")
    pd.DataFrame({"step": [0]}).to_csv(path / "history.csv", index=False)
    finish_run(path, "complete")
    parent = load_run(path)
    # A minimal historical numeric result, used only as a synthetic fixture.
    baseline = {"path": path, "manifest": parent["manifest"], "weights": {"model": weights},
                "predictions": parent["predictions"], "config": config.to_dict(),
                "metrics": saved_metrics}
    return AblationProblem("clean", 0, config, parent, parent, baseline, baseline,
                           {"train": features, "test": features, "train_labels": labels, "train_ids": ids}, refs)


class FocusedAblationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.problem = fixture(self.root)
        self.execution = AerExecution(device="CPU", workers=2, threads=1, circuit_batch_size=4096, gather_ms=5)

    def test_plan_contains_120_unique_changes_and_no_repeated_default(self):
        config = replace(self.problem.config, unlearning=UnlearningConfig())
        problems = [replace(self.problem, config=config, fold=fold, name=name)
                    for fold in range(5) for name in ("clean", "label_flip_50pct")]
        table = plan_table(problems)
        self.assertEqual(len(table), 120)
        self.assertFalse(table.duplicated(["problem", "parameter", "value"]).any())
        self.assertTrue((table.updates == 100).all())
        for parameter, _, setting in parameter_settings(UnlearningConfig()):
            self.assertEqual(sum(getattr(setting, name) != value for name, value in DEFAULTS.items()), 1)
            self.assertEqual(setting.methods, ("guided",))

    def test_shared_requests_and_all_gradients_equal_serial_aer(self):
        model = VQC(self.problem.config.circuit, self.problem.config.backend)
        x, w = self.problem.arrays["train"][:2], self.problem.references.original_weights
        targets = np.eye(3)[[0, 1]]
        expected = model.loss_and_gradient(x, targets, w)
        with AerBatchService(self.problem.config.circuit, self.problem.config.backend, self.execution) as service:
            with ThreadPoolExecutor(max_workers=2) as workers:
                actual = list(workers.map(lambda _: service.model().loss_and_gradient(x, targets, w), range(2)))
            for loss, gradient in actual:
                np.testing.assert_allclose(loss, expected[0], atol=1e-12, rtol=0)
                np.testing.assert_allclose(gradient, expected[1], atol=1e-12, rtol=0)
            self.assertEqual(service.requests, 2)
            self.assertLessEqual(service.batches, 2)

    def test_concurrent_optimizers_match_independent_serial_trajectories(self):
        settings = [replace(self.problem.config.unlearning, alpha=value) for value in (0.0, 2.0)]
        expected = [methods.unlearn(VQC(self.problem.config.circuit, self.problem.config.backend),
                    self.problem.arrays["train"], self.problem.references, setting,
                    seed=DEFAULT_SEED + 2000)[0] for setting in settings]
        with AerBatchService(self.problem.config.circuit, self.problem.config.backend, self.execution) as service:
            with ThreadPoolExecutor(max_workers=2) as workers:
                jobs = [workers.submit(methods.unlearn, service.model(), self.problem.arrays["train"],
                                       self.problem.references, setting, "guided", DEFAULT_SEED + 2000)
                        for setting in settings]
                actual = [job.result()[0] for job in jobs]
        for first, second in zip(expected, actual):
            np.testing.assert_allclose(first.last_weights, second.last_weights, atol=1e-12, rtol=0)
            np.testing.assert_allclose(first.history.objective, second.history.objective, atol=1e-12, rtol=0)

    def test_interrupted_shared_fit_resumes_exact_adam_and_rng_state(self):
        expected = methods.unlearn(VQC(self.problem.config.circuit, self.problem.config.backend),
                                  self.problem.arrays["train"], self.problem.references,
                                  self.problem.config.unlearning, seed=DEFAULT_SEED + 2000)[0]
        checkpoint = self.root / "resume.json"
        def interrupt(row):
            if row["step"] == 2:
                raise KeyboardInterrupt("synthetic interruption")
        with AerBatchService(self.problem.config.circuit, self.problem.config.backend, self.execution) as service:
            with self.assertRaises(KeyboardInterrupt):
                methods.unlearn(service.model(), self.problem.arrays["train"], self.problem.references,
                                self.problem.config.unlearning, seed=DEFAULT_SEED + 2000,
                                callback=interrupt, checkpoint_path=checkpoint)
            self.assertEqual(load_json(checkpoint)["step"], 2)
            resumed = methods.unlearn(service.model(), self.problem.arrays["train"], self.problem.references,
                                      self.problem.config.unlearning, seed=DEFAULT_SEED + 2000,
                                      checkpoint_path=checkpoint)[0]
        np.testing.assert_allclose(resumed.last_weights, expected.last_weights, atol=1e-12, rtol=0)

    def test_review_guard_permits_only_separate_unlearning_directories(self):
        guard = protect_review()
        try:
            with ablation_execution(self.root, guard):
                with self.assertRaisesRegex(RuntimeError, "separate run"):
                    CheckpointSession().open(self.root / "results/runs/supervised_models", {}, allow_execution=True)
                import src.training
                with self.assertRaisesRegex(RuntimeError, "forbids"):
                    src.training.fit_vqc()
            with self.assertRaisesRegex(RuntimeError, "blocks fitting"):
                methods.unlearn()
        finally:
            guard.close()

    def test_fresh_cached_forced_and_changed_settings_are_isolated(self):
        settings = replace(self.problem.config.unlearning, beta=2.0, methods=("guided",))
        session, lock = CheckpointSession(), Lock()
        with AerBatchService(self.problem.config.circuit, self.problem.config.backend, self.execution) as service:
            arguments = (self.problem, settings, service, session, lock, self.root, {"synthetic": True}, "synthetic")
            first, status = _run_variant(*arguments)
            self.assertEqual(status, "trained")
            second, status = _run_variant(*arguments)
            self.assertEqual(status, "loaded")
            self.assertEqual(first["path"], second["path"])
            forced, _ = _run_variant(self.problem, settings, service, CheckpointSession(force_retrain=True), lock,
                                      self.root, {"synthetic": True}, "synthetic")
            self.assertNotEqual(first["path"], forced["path"])
            changed, _ = _run_variant(self.problem, replace(settings, beta=4.0), service, session, lock,
                                       self.root, {"synthetic": True}, "synthetic")
            self.assertNotEqual(first["path"], changed["path"])
            self.assertTrue((first["path"] / "checkpoints/model.resume.json").exists())
            load_run(first["path"])

    def test_cpu_performance_probe_is_verified_and_cached(self):
        selected, rows, destination = benchmark_execution(self.problem, self.root,
                                      candidates=[self.execution], require_gpu_device=False, probe_steps=1)
        self.assertEqual(selected, self.execution)
        self.assertEqual(rows.iloc[0].status, "passed")
        with patch("src.ablations.AerBatchService", side_effect=AssertionError("benchmark should be cached")):
            second, _, _ = benchmark_execution(self.problem, self.root,
                            candidates=[self.execution], require_gpu_device=False, probe_steps=1)
        self.assertEqual(second, selected)
        self.assertTrue((destination / "benchmark.json").exists())

    def test_study_persists_portable_rows_and_computes_paired_differences(self):
        review = SimpleNamespace(root=self.root, report_path=self.root / "original_report",
                                 source_fingerprint="synthetic")
        reduced = {"alpha": (0.0, 1.0, 2.0), "regularization": (0.01,), "beta": (1.0,)}
        with patch("src.ablations.SWEEPS", reduced), patch("src.evaluation.ablation_analysis.SWEEPS", reduced):
            with ablation_execution(self.root):
                index_path = run_study(review, [self.problem], self.execution, baseline_checks=False)
            index, tables = study_tables(index_path, self.root)
            self.assertTrue(index["complete"])
            self.assertEqual(len(tables["summary"]), 4)
            self.assertTrue(tables["coverage"].completed.all())
            self.assertEqual(len(tables["paired_changes"]), 2)
            for _, row in tables["paired_changes"].iterrows():
                candidate = tables["summary"].loc[tables["summary"].run_id == row.run_id].iloc[0]
                baseline = tables["summary"].loc[tables["summary"].parameter == "baseline"].iloc[0]
                self.assertAlmostEqual(row.delta_retained_accuracy, candidate.retained_accuracy - baseline.retained_accuracy)
            figures = comparison_figures(tables)
            self.assertEqual(len(figures), 4)
            for figure in figures.values():
                plt.close(figure)
            destination = export_study(index_path, self.root)
            self.assertTrue((destination / "summary.csv").exists())
            self.assertEqual(load_json(destination / "provenance.json")["completed_new_runs"], 2)
            for name in ("src/ablations.py", "src/models/parallel_aer.py", "src/evaluation/ablation_analysis.py",
                         "src/checkpoint_storage.py", "main_project.ipynb", "README.md"):
                file = self.root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("synthetic bundle fixture")
            bundle = download_bundle(index_path, self.root)
            with ZipFile(bundle) as archive:
                self.assertEqual(archive.testzip(), None)
                self.assertEqual(sum(name.endswith("checkpoints/model.resume.json") for name in archive.namelist()), 2)
                self.assertFalse(any(name.startswith("parents/") for name in archive.namelist()))

    def test_drive_backup_flushes_interruption_and_preserves_newer_local_progress(self):
        durable, local = self.root / "drive/model.resume.json", self.root / "runtime/model.resume.json"
        mirror = CheckpointMirror(durable)
        mirror.enabled, mirror.local = True, local
        with self.assertRaises(KeyboardInterrupt):
            with mirror as progress_file:
                save_json(progress_file, {"identity": "test", "step": 2})
                raise KeyboardInterrupt("synthetic runtime interrupt")
        self.assertEqual(load_json(durable)["step"], 2)
        save_json(local, {"identity": "test", "step": 3})
        resumed = CheckpointMirror(durable)
        resumed.enabled, resumed.local = True, local
        with resumed as progress_file:
            self.assertEqual(load_json(progress_file)["step"], 3)
            save_json(progress_file, {"identity": "test", "step": 4})
        self.assertEqual(load_json(durable)["step"], 4)


if __name__ == "__main__":
    unittest.main()
