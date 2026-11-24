"""Synthetic checks only: noisy gradients, parallel fits and durable continuation."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace, asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event
from unittest.mock import patch
from zipfile import ZipFile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from test_noise_study import fixture as inference_fixture
from src import training
from src.artifacts import finish_run, load_json, load_run, save_arrays, save_json
from src.checkpoint_storage import CheckpointMirror, _read_checkpoint, checkpoint_status, verify_checkpoint_storage
from src.config import DEFAULT_SEED, ExperimentConfig, TrainingConfig, UnlearningConfig
from src.evaluation.saved_results import protect_review
from src.evaluation.noisy_training_analysis import noisy_training_tables, noisy_training_figures, export_noisy_training
from src.execution_resources import memory_budget_mb, parallel_worker_counts, benchmark_policy
from src.models.noise import DensityReadoutExecutor, NoiseExecution, NoiseSetting, class_probabilities
from src.models.noisy_training import NoisyBatchService
from src.models.parallel_aer import AerExecution
from src.models.vqc import VQC
from src.noisy_training import (load_training_problems, training_plan, noisy_training_execution,
    benchmark_noisy_training, run_noisy_training, noisy_training_bundle, _interleave, _parallel)
from src.unlearning.targets import make_forget_target, prepare_references


def fixture(root):
    review = inference_fixture(root)
    condition = review.conditions["clean"]
    config = replace(ExperimentConfig.from_dict(condition["config"]),
        training=TrainingConfig(steps=2, batch_size=2, evaluate_every=1),
        unlearning=UnlearningConfig(training=TrainingConfig(steps=2, batch_size=2, evaluate_every=1)))
    original = condition["models"]["clean"]
    x = np.load(original["path"] / "features.npz")["test"]
    labels, ids = original["predictions"]["test_labels"], original["predictions"]["test_sample_ids"]
    refs = prepare_references(VQC(config.circuit, config.backend), x, labels, labels == 2,
                              original["weights"]["model"], config.unlearning)
    for role, run in {**condition["models"], **condition["unlearned"]}.items():
        retained = role == "retrained_clean"
        mask = labels != 2 if retained else np.ones(len(labels), bool)
        save_arrays(run["path"] / "features.npz", train=x[mask], train_labels=labels[mask], train_ids=ids[mask],
                    validation=x[mask], validation_labels=labels[mask], validation_ids=ids[mask], test=x)
        save_json(run["path"] / "config.json", config.to_dict())
        save_json(run["path"] / "preprocessing.json", {"synthetic": True})
        (run["path"] / "preprocessing.joblib").write_bytes(b"must never be unpickled")
        if role in condition["unlearned"]:
            save_json(run["path"] / "training_references.json", {"references": asdict(refs)})
            metrics = run["metrics"]
            metrics["target"] = make_forget_target(refs.calibration_probabilities, method=role.split("_")[-1]).tolist()
            save_json(run["path"] / "metrics.json", metrics)
        finish_run(run["path"], "complete")
    condition["config"] = config.to_dict()
    condition["models"] = {role: load_run(run["path"]) for role, run in condition["models"].items()}
    condition["unlearned"] = {role: load_run(run["path"]) for role, run in condition["unlearned"].items()}
    return review


class NoisyTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.review = fixture(self.root)
        self.problems, self.historical, self.bindings = load_training_problems(self.review, folds=(0,))
        self.problem = self.problems[0]
        self.execution = AerExecution(device="CPU", workers=2, threads=1, circuit_batch_size=1024)
        self.setting = NoiseSetting("combined", 0.01)

    def test_large_gpu_memory_scales_and_round_robin_starts_different_folds(self):
        self.assertEqual(memory_budget_mb({"memory.total": 81920, "memory.free": 80000}), 65536)
        self.assertEqual(memory_budget_mb({"memory.total": 40960, "memory.free": 40000}), 32768)
        self.assertIn(5, parallel_worker_counts()); self.assertIn(12, parallel_worker_counts())
        self.assertEqual(benchmark_policy([self.execution], automatic=True),
                         benchmark_policy([replace(self.execution, memory_mb=32768)], automatic=True))
        items = [(fold, i) for fold in range(5) for i in range(2)]
        ordered = _interleave(items, lambda item: item[0])
        self.assertEqual([x[0] for x in ordered[:5]], list(range(5)))
        barrier, results = Barrier(5), []
        def task(item):
            barrier.wait(timeout=10)
            return item[0], "passed"
        _parallel(ordered[:5], task, replace(self.execution, workers=5), Event(),
                  lambda item, result, status: results.append(result))
        self.assertEqual(set(results), set(range(5)))

    def test_noisy_expectations_and_all_gradients_match_independent_joint_readout(self):
        x, w = self.problem.arrays["train"][:2], self.problem.original["weights"]["model"]
        executor = DensityReadoutExecutor(self.problem.config.circuit, self.setting, NoiseExecution("CPU", 128, 1))
        joint = executor.run(x, np.broadcast_to(w, (len(x), len(w))))
        expected, _ = class_probabilities(joint, readout_error=self.setting.readout_error)
        with NoisyBatchService(self.problem.config.circuit, self.problem.config.backend, self.execution, self.setting) as service:
            model = service.model()
            np.testing.assert_allclose(model.predict_proba(x, w), expected, atol=1e-12, rtol=0)
            targets = np.eye(3)[[0, 1]]
            _, gradient = model.loss_and_gradient(x, targets, w)
            numerical = []
            for i in range(len(w)):
                delta = np.zeros_like(w); delta[i] = 1e-6
                numerical.append((model.loss(x, targets, w+delta)-model.loss(x, targets, w-delta))/2e-6)
            np.testing.assert_allclose(gradient, numerical, atol=1e-9, rtol=0)

    def test_parallel_noisy_optimization_and_interruption_match_serial(self):
        a = self.problem.arrays
        configs = [self.problem.config.training, replace(self.problem.config.training, learning_rate=0.03)]
        with NoisyBatchService(self.problem.config.circuit, self.problem.config.backend, self.execution, self.setting) as service:
            def fit(config, checkpoint=None, callback=None):
                return training.fit_vqc(service.model(), a["train"], a["train_labels"], a["validation"], a["validation_labels"],
                                        config, checkpoint_path=checkpoint, callback=callback)
            serial = [fit(config) for config in configs]
            with ThreadPoolExecutor(max_workers=2) as pool:
                parallel = list(pool.map(fit, configs))
            for actual, expected in zip(parallel, serial):
                np.testing.assert_allclose(actual.last_weights, expected.last_weights, atol=1e-12, rtol=0)
            file = self.root / "interrupted.json"
            def interrupt(row):
                if row["step"] == 1: raise KeyboardInterrupt("synthetic interruption")
            with self.assertRaises(KeyboardInterrupt): fit(configs[0], file, interrupt)
            resumed = fit(configs[0], file)
            np.testing.assert_allclose(resumed.last_weights, serial[0].last_weights, atol=1e-12, rtol=0)

    def test_scope_restricts_original_fitting_and_restores_review_guard(self):
        guard = protect_review()
        try:
            with noisy_training_execution(self.root, guard):
                with self.assertRaisesRegex(RuntimeError, "separate noisy-model"):
                    training.fit_vqc(VQC(self.problem.config.circuit, self.problem.config.backend))
                from src.checkpoints import CheckpointSession
                with self.assertRaisesRegex(RuntimeError, "separate noisy-model"):
                    CheckpointSession().open(self.root / "results/runs/supervised_models", {}, allow_execution=True)
            with self.assertRaisesRegex(RuntimeError, "blocks fitting"): training.fit_vqc()
        finally:
            guard.close()

    def test_noisy_training_benchmark_checks_and_caches_parallel_candidate(self):
        with noisy_training_execution(self.root):
            selected, table, _ = benchmark_noisy_training(self.problem, self.setting, self.root,
                candidates=[self.execution], minimum_workers=2, require_gpu_device=False, probe_steps=1)
        self.assertEqual(selected, self.execution); self.assertEqual(table.iloc[0].status, "passed")
        with patch("src.noisy_training.NoisyBatchService", side_effect=AssertionError("must load benchmark")):
            cached, _, _ = benchmark_noisy_training(self.problem, self.setting, self.root,
                candidates=[self.execution], minimum_workers=2, require_gpu_device=False, probe_steps=1)
        self.assertEqual(cached, selected)

    def test_complete_noisy_pipeline_caches_teachers_and_exports_matched_comparisons(self):
        before = {p: p.read_bytes() for item in self.historical.values() for p in item.run["path"].rglob("*") if p.is_file()}
        self.assertEqual(len(training_plan(self.problems)), 4)
        with noisy_training_execution(self.root):
            file = run_noisy_training(self.review, self.problems, self.historical, self.bindings, self.execution,
                                      setting=self.setting, allow_training=True)
        index, tables = noisy_training_tables(file, self.root)
        self.assertTrue(index["complete"]); self.assertTrue(tables["coverage"].completed.all())
        self.assertEqual(len(tables["summary"]), 16)
        np.testing.assert_array_equal(tables["summary"].query("role == 'retrained'").full_kl_mean, 0)
        teacher = load_json(next((file.parent / "teachers").glob("*.json")))
        new_parent = load_run(self.root / index["supervised"][self.problem.original["path"].name]["run"])
        np.testing.assert_array_equal(teacher["references"]["original_weights"], new_parent["weights"]["model"])
        np.testing.assert_array_equal(tables["targets"].target_2, 0)
        for p, content in before.items(): self.assertEqual(p.read_bytes(), content)
        with noisy_training_execution(self.root), patch("src.training.fit_vqc", side_effect=AssertionError("must load fits")), \
             patch("src.unlearning.methods.unlearn", side_effect=AssertionError("must load unlearning")):
            same = run_noisy_training(self.review, self.problems, self.historical, self.bindings, self.execution,
                                      setting=self.setting, allow_training=True)
        self.assertEqual(file, same)
        self.assertTrue(all(r["status"] == "loaded" for g in ("supervised", "unlearned") for r in load_json(same)[g].values()))
        figures = noisy_training_figures(tables)
        self.assertEqual(len(figures), 2)
        output = export_noisy_training(file, self.root, figures)
        for figure in figures.values(): plt.close(figure)
        self.assertTrue((output / "summary.csv").exists())
        names = ("src/noisy_training.py", "src/models/noisy_training.py", "src/evaluation/noisy_training_analysis.py",
                 "src/checkpoint_storage.py", "src/execution_resources.py", "src/models/parallel_aer.py", "main_project.ipynb", "README.md")
        for name in names:
            p = self.root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("synthetic bundle fixture")
        bundle = noisy_training_bundle(file, self.root)
        with ZipFile(bundle) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(sum(n.endswith("checkpoints/model.resume.json") for n in archive.namelist()), 4)

    def test_drive_generations_verify_receipts_and_recover_after_vm_local_loss(self):
        durable, local = self.root / "drive/model.resume.json", self.root / "runtime/model.resume.json"
        mirror = CheckpointMirror(durable); mirror.enabled, mirror.local = True, local
        with mirror:
            save_json(local, {"identity": "synthetic", "step": 1, "weights": [1]}); mirror._flush()
            save_json(local, {"identity": "synthetic", "step": 2, "weights": [2]}); mirror._flush()
        self.assertEqual(_read_checkpoint(durable)[0]["step"], 2)
        local.unlink(); durable.write_bytes(b"synthetic interrupted/corrupt mounted write")
        recovered = CheckpointMirror(durable); recovered.enabled, recovered.local = True, local
        with recovered as progress:
            self.assertEqual(load_json(progress)["step"], 1)
        self.assertEqual(_read_checkpoint(durable)[0]["step"], 1)
        report = verify_checkpoint_storage(self.root)
        self.assertTrue(report["verified"]); self.assertFalse(report["runtime_mirror"])
        self.assertEqual(checkpoint_status(self.root), [])
        with self.assertRaises(FileNotFoundError):
            verify_checkpoint_storage(self.root / "missing")

    def test_interrupted_stage_resumes_its_saved_models_and_forced_session_is_separate(self):
        original_fit = training.fit_vqc
        def interrupted_fit(*args, **kwargs):
            callback = kwargs.get("callback")
            def interrupt(row):
                if callback: callback(row)
                if row["step"] == 1: raise KeyboardInterrupt("synthetic stage interruption")
            kwargs["callback"] = interrupt
            return original_fit(*args, **kwargs)
        arguments = (self.review, self.problems, self.historical, self.bindings, self.execution)
        with noisy_training_execution(self.root), patch("src.training.fit_vqc", interrupted_fit):
            with self.assertRaises(KeyboardInterrupt):
                run_noisy_training(*arguments, setting=self.setting, allow_training=True)
        files = list((self.root / "results/runs/noisy_supervised").glob("*/checkpoints/model.resume.json"))
        self.assertTrue(files); self.assertTrue(all(load_json(file)["step"] == 1 for file in files))
        with noisy_training_execution(self.root):
            resumed = run_noisy_training(*arguments, setting=self.setting, allow_training=True)
        index = load_json(resumed)
        self.assertTrue(all(row["status"] == "resumed" for row in index["supervised"].values()))
        preserved = resumed.read_bytes()
        with noisy_training_execution(self.root):
            forced = run_noisy_training(*arguments, setting=self.setting, allow_training=True, force_retrain=True)
        self.assertNotEqual(resumed, forced); self.assertEqual(resumed.read_bytes(), preserved)
        self.assertTrue(all(row["status"] == "trained" for row in load_json(forced)["supervised"].values()))


if __name__ == "__main__":
    unittest.main()
