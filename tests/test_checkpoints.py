"""Checkpoint continuation tests use tiny synthetic data, never MNIST."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.artifacts import load_json, load_run
from src.checkpoints import CheckpointSession
from src.config import DEFAULT_SEED, PoisonConfig, TrainingConfig
from src.experiment_stages import train_condition, unlearn_condition
from src.experiments import run_clean_vqc
from src.models.vqc import VQC
from src.training import fit_vqc, optimize
from test_notebook_stages import fixtures


class CheckpointTests(unittest.TestCase):
    def test_finite_shot_resume_restores_adam_best_weights_and_both_rngs(self):
        _, config = fixtures()
        backend = replace(config.backend, shots=64)
        training = replace(config.training, steps=4, evaluate_every=2)
        random = np.random.default_rng(DEFAULT_SEED)
        features, labels = random.uniform(0, np.pi, (9, 4)), np.tile([0, 1, 2], 3)

        def fit(model, **kwargs):
            return fit_vqc(model, features[:6], labels[:6], features[6:], labels[6:],
                           training, **kwargs)

        def interrupt(row):
            if row["step"] == 2:
                raise KeyboardInterrupt("Simulated notebook interruption")

        uninterrupted_model = VQC(config.circuit, backend)
        expected = fit(uninterrupted_model)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "progress.json"
            with self.assertRaises(KeyboardInterrupt):
                fit(VQC(config.circuit, backend), checkpoint_path=checkpoint, callback=interrupt)
            self.assertEqual(load_json(checkpoint)["step"], 2)
            resumed_model = VQC(config.circuit, backend)
            events = []
            actual = fit(resumed_model, checkpoint_path=checkpoint, callback=events.append)
            self.assertEqual([event["step"] for event in events], [4])
            np.testing.assert_array_equal(actual.last_weights, expected.last_weights)
            np.testing.assert_array_equal(actual.weights, expected.weights)
            self.assertEqual(actual.best_step, expected.best_step)
            self.assertEqual(actual.best_score, expected.best_score)
            for column in actual.history.columns.difference(["elapsed_seconds"]):
                np.testing.assert_array_equal(actual.history[column], expected.history[column])
            self.assertEqual(resumed_model.executor.stats, uninterrupted_model.executor.stats)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                fit_vqc(VQC(config.circuit, backend), features[:6], labels[:6], features[6:], labels[6:],
                        replace(training, learning_rate=training.learning_rate / 2), checkpoint_path=checkpoint)

    def test_checkpoint_between_evaluations_and_early_stop_survive_interruptions(self):
        config = TrainingConfig(steps=5, evaluate_every=2)
        def objective(weights):
            return float(np.sum(weights**2)), 2 * weights
        def evaluate(weights):
            return {"loss": float(np.sum(weights**2))}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resume.json"
            calls = 0
            def failing_objective(weights):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise RuntimeError("Mid-update failure")
                return objective(weights)
            with self.assertRaises(RuntimeError):
                optimize([1.0], failing_objective, evaluate, config, "loss", checkpoint_path=path)
            self.assertEqual(load_json(path)["step"], 3)  # No validation evaluation at step 3.
            resumed = optimize([1.0], objective, evaluate, config, "loss", checkpoint_path=path)
            expected = optimize([1.0], objective, evaluate, config, "loss")
            np.testing.assert_array_equal(resumed.last_weights, expected.last_weights)
            stopped_path = Path(temporary) / "stopped.json"
            def interrupt(row):
                if row["step"] == 1:
                    raise KeyboardInterrupt()
            stopping = replace(config, patience=1, evaluate_every=1)
            with self.assertRaises(KeyboardInterrupt):
                optimize([1.0], objective, lambda weights: {"loss": 1.0}, stopping, "loss",
                         interrupt, checkpoint_path=stopped_path)
            result = optimize([1.0], lambda weights: self.fail("Already early-stopped"),
                              lambda weights: self.fail("No extra evaluation"), stopping, "loss",
                              checkpoint_path=stopped_path)
            self.assertEqual(result.history.step.tolist(), [0, 1])

    def test_completed_model_is_loaded_and_second_model_resumes_its_own_run(self):
        splits, config = fixtures()
        config = replace(config, training=replace(config.training, steps=2))
        def interrupt(event):
            if event["model"] == "poisoned" and event.get("step") == 1:
                raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(KeyboardInterrupt):
                train_condition(config, splits, allow_training=True, root=temporary, callback=interrupt)
            directory = Path(temporary).resolve() / "results/runs/supervised_models"
            manifests = {path.parent: load_json(path) for path in directory.glob("*/manifest.json")}
            completed = next(path for path, meta in manifests.items() if meta["status"] == "complete")
            interrupted = next(path for path, meta in manifests.items() if meta["status"] == "failed")
            events = []
            result = train_condition(config, splits, allow_training=True, root=temporary, callback=events.append)
            self.assertEqual(result["models"]["clean"]["path"], completed)
            self.assertEqual(result["models"]["poisoned"]["path"], interrupted)
            self.assertEqual([row["step"] for row in events if row["model"] == "poisoned" and "step" in row], [2])
            self.assertFalse(any(row["stage"] == "training_progress" and row["model"] == "clean" for row in events))
            # A newer forced attempt takes priority over an older completed model.
            def stop_new_attempt(event):
                if event["model"] == "clean" and event.get("step") == 0:
                    raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                train_condition(config, splits, allow_training=True, root=temporary,
                                checkpoints=CheckpointSession(force_retrain=True), callback=stop_new_attempt)
            newest = max(directory.glob("*/manifest.json"), key=lambda p: load_json(p)["created_utc"]).parent
            continued = train_condition(config, splits, allow_training=True, root=temporary)
            self.assertEqual(continued["models"]["clean"]["path"], newest)
            self.assertNotEqual(newest, completed)
            # Training settings change: fresh initialization, even with normal reuse enabled.
            changed = replace(config, training=replace(config.training, steps=3))
            new_events = []
            changed_result = train_condition(changed, splits, allow_training=True, root=temporary,
                                             callback=new_events.append)
            self.assertNotEqual(changed_result["models"]["clean"]["path"], completed)
            self.assertTrue(any(row["model"] == "clean" and row.get("step") == 0 for row in new_events))

    def test_force_replaces_active_link_once_per_model_and_preserves_history(self):
        splits, config = fixtures()
        config = replace(config, poison=PoisonConfig(kind="pixel_patch", rate=0.5))
        with tempfile.TemporaryDirectory() as temporary:
            old = train_condition(config, splits, allow_training=True, root=temporary)
            session = CheckpointSession(force_retrain=True)
            forced = train_condition(config, splits, allow_training=True, root=temporary, checkpoints=session)
            old_clean, new_clean = old["models"]["clean"], forced["models"]["clean"]
            self.assertNotEqual(old_clean["path"], new_clean["path"])
            np.testing.assert_array_equal(load_run(old_clean["path"])["weights"]["model"], old_clean["weights"]["model"])
            changed = replace(config, poison=replace(config.poison, rate=1.0))
            other = train_condition(changed, splits, allow_training=True, root=temporary, checkpoints=session)
            for role in ("clean", "retrained_clean", "retrained_poisoned"):
                self.assertEqual(other["models"][role]["path"], forced["models"][role]["path"])
            cached = train_condition(config, splits, root=temporary)
            self.assertEqual(cached["models"]["clean"]["path"], new_clean["path"])
            active = load_json(new_clean["path"].parent / "active_checkpoints.json")
            active_clean = [item for item in active.values() if item["slot"]["role"] == "clean"]
            self.assertEqual(len(active_clean), 1)
            self.assertEqual(active_clean[0]["run"], new_clean["path"].name)
            disabled = train_condition(config, splits, root=temporary,
                                       checkpoints=CheckpointSession(force_retrain=True))
            self.assertTrue(all(run is None for run in disabled["models"].values()))

    def test_unlearning_resume_uses_saved_teacher_and_identical_sampling(self):
        splits, config = fixtures()
        config = replace(config, poison=PoisonConfig(), backend=replace(config.backend, shots=64),
                         unlearning=replace(config.unlearning, methods=("guided",),
                                             training=replace(config.training, steps=2)))
        def interrupt(event):
            if event.get("step") == 1:
                raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as other:
            trained = train_condition(config, splits, allow_training=True, root=temporary)
            expected = unlearn_condition(trained, splits, allow_execution=True, root=temporary)
            # Same hash-verified parents; a different output root gives an independent attempt.
            with self.assertRaises(KeyboardInterrupt):
                unlearn_condition(trained, splits, allow_execution=True, root=other, callback=interrupt)
            with patch("src.experiment_stages.prepare_references", side_effect=AssertionError("Teacher must stay fixed")):
                # Avoid condition export across roots; numerical stage behavior is the target here.
                with patch("src.experiment_stages._save_condition"):
                    resumed = unlearn_condition(trained, splits, allow_execution=True, root=other)
            first, second = expected["unlearned"]["clean_guided"], resumed["unlearned"]["clean_guided"]
            np.testing.assert_array_equal(first["weights"]["model"], second["weights"]["model"])
            np.testing.assert_array_equal(first["predictions"]["model__test"], second["predictions"]["model__test"])
            self.assertEqual(first["metrics"], second["metrics"])

    def test_gold_resume_and_forced_session_gold_import(self):
        splits, config = fixtures()
        config = replace(config, poison=PoisonConfig(), backend=replace(config.backend, shots=64),
                         training=replace(config.training, steps=2))
        def gold(root, **kwargs):
            return run_clean_vqc(splits, data_config=config.data, circuit_config=config.circuit,
                                 backend_config=config.backend, training_config=config.training,
                                 allow_training=True, root=root, **kwargs)
        def interrupt(row):
            if row["step"] == 1:
                raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(KeyboardInterrupt):
                gold(temporary, callback=interrupt)
            original = gold(temporary)
            session = CheckpointSession(force_retrain=True)
            fresh_gold = gold(temporary, checkpoints=session)
            self.assertNotEqual(original["path"], fresh_gold["path"])
            imported = train_condition(config, splits, gold_run=fresh_gold, allow_training=True,
                                       root=temporary, checkpoints=session)
            self.assertEqual(imported["models"]["clean"]["manifest"]["identity"]["imported_gold"], fresh_gold["path"].name)
            fresh = train_condition(config, splits, gold_run=original, allow_training=True,
                                    root=temporary, checkpoints=CheckpointSession(force_retrain=True))
            self.assertIsNone(fresh["models"]["clean"]["manifest"]["identity"]["imported_gold"])


if __name__ == "__main__":
    unittest.main()
