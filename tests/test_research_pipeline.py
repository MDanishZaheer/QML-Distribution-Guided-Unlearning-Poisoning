"""Scientific correctness checks; fixtures are synthetic and are not research results."""

from dataclasses import replace
from pathlib import Path
from importlib.util import find_spec
import hashlib
import io
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from src.artifacts import load_arrays, load_json, save_poisoned_dataset
from src.config import (BackendConfig, CircuitConfig, DataConfig, ExperimentConfig,
                        PoisonConfig, TrainingConfig, UnlearningConfig)
from src.data.dataset import DatasetSplits, ImageDataset, download_mnist, load_mnist, make_cv_splits
from src.data.poisoning import apply_poisoning, forget_mask
from src.data.preprocessing import ImagePreprocessor
from src.evaluation.metrics import classification_metrics, compare_to_retraining, summarize_runs
from src.evaluation.plots import (draw_circuit, plot_class_balance, plot_confusion_comparison,
                                  plot_examples, plot_forget_confidence, plot_metric_summary,
                                  plot_pca_variance, plot_poisoning_curve, plot_training_history,
                                  save_figure)
from src.experiments import evaluate_noise_sweep, results_table, restore_vqc, run_experiment, run_study
from src.models.backends import counts_to_expectations
from src.models.circuits import build_malik_circuit, parameter_groups
from src.models.vqc import VQC, one_hot
from src.training import fit_vqc
from src.unlearning.methods import unlearning_loss_and_gradient
from src.unlearning.targets import make_forget_target, prepare_references


def synthetic_dataset(prefix, per_class=6, seed=0):
    random = np.random.default_rng(seed)
    labels = np.repeat(np.arange(3), per_class)
    images = random.integers(0, 40, (len(labels), 28, 28), dtype=np.uint8)
    for index, label in enumerate(labels):
        images[index, 3 + label * 7:7 + label * 7, 3:22] = random.integers(150, 255)
    return ImageDataset(images, labels, np.array([f"{prefix}_{i}" for i in range(len(labels))]))


def tiny_splits():
    return DatasetSplits(synthetic_dataset("train", 6, 0),
                         synthetic_dataset("validation", 2, 1),
                         synthetic_dataset("test", 3, 2))


EXACT = BackendConfig(shots=None, device="CPU", max_parallel_threads=1)
SMALL_CIRCUIT = CircuitConfig(reuploads=1, ansatz_reps=1)


class DataTests(unittest.TestCase):
    def test_downloader_verifies_archive_and_preserves_official_pool_ids(self):
        buffer = io.BytesIO()
        images = np.zeros((4, 28, 28), dtype=np.uint8)
        labels = np.array([0, 1, 2, 3])
        np.savez(buffer, x_train=images, y_train=labels, x_test=images, y_test=labels)
        payload = buffer.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            with patch("src.data.dataset.MNIST_SHA256", digest):
                with patch("src.data.dataset.urllib.request.urlopen", return_value=io.BytesIO(payload)):
                    path = download_mnist(temporary)
                train, test = load_mnist(temporary)
                self.assertEqual(len(train), 3)
                self.assertTrue(all(value.startswith("mnist_train_") for value in train.sample_ids))
                self.assertFalse(set(train.sample_ids) & set(test.sample_ids))
                path.write_bytes(b"corrupt archive")
                with self.assertRaises(ValueError):
                    load_mnist(temporary)

    def test_label_poisoning_preserves_truth_and_exact_forget_ids(self):
        original = synthetic_dataset("training", 10)
        images, labels = original.images.copy(), original.labels.copy()
        result = apply_poisoning(original, PoisonConfig(kind="label_flip", rate=0.2, seed=8))
        self.assertEqual(result.is_poisoned.sum(), 2)
        self.assertTrue((result.original.labels[result.is_poisoned] == 0).all())
        self.assertTrue((result.observed.labels[result.is_poisoned] == 2).all())
        self.assertEqual(forget_mask(result, definition="observed").sum(), 12)
        self.assertEqual(forget_mask(result, definition="original").sum(), 10)
        np.testing.assert_array_equal(original.images, images)
        np.testing.assert_array_equal(original.labels, labels)
        retained = original.sample_ids[~forget_mask(result)]
        self.assertFalse(set(retained) & set(original.sample_ids[result.is_poisoned]))

    def test_pixel_patch_and_denominator(self):
        original = synthetic_dataset("training", 10)
        config = PoisonConfig(kind="pixel_patch", rate=0.1, denominator="all_training", patch_size=2)
        result = apply_poisoning(original, config)
        self.assertEqual(result.is_poisoned.sum(), 3)
        self.assertTrue((result.observed.images[result.is_poisoned, -2:, -2:] == 255).all())
        np.testing.assert_array_equal(result.observed.labels, original.labels)
        np.testing.assert_array_equal(result.observed.images[~result.is_poisoned], original.images[~result.is_poisoned])
        with self.assertRaises(ValueError):
            apply_poisoning(original, replace(config, rate=0.9))

    def test_pca_does_not_learn_from_validation(self):
        splits = tiny_splits()
        processor = ImagePreprocessor().fit(splits.train.images, splits.train.sample_ids)
        components, mean = processor.pca.components_.copy(), processor.pca.mean_.copy()
        values = processor.transform(np.full((2, 28, 28), 255, dtype=np.uint8))
        self.assertEqual(values.shape, (2, 4))
        self.assertTrue(((values >= 0) & (values <= np.pi)).all())
        np.testing.assert_array_equal(components, processor.pca.components_)
        np.testing.assert_array_equal(mean, processor.pca.mean_)
        self.assertFalse(set(processor.fit_sample_ids_) & set(splits.validation.sample_ids))

    def test_cv_folds_are_disjoint_reproducible_and_cover_data(self):
        data = synthetic_dataset("pool", 15)
        folds = make_cv_splits(data, n_folds=3)
        repeated = make_cv_splits(data, n_folds=3)
        test_ids = []
        for first, second in zip(folds, repeated):
            self.assertEqual(first.fingerprint(), second.fingerprint())
            test_ids.extend(first.test.sample_ids)
        self.assertEqual(len(set(test_ids)), len(data))
        self.assertEqual(len(test_ids), len(data))
        with self.assertRaises(ValueError):
            DatasetSplits(data, data, data)

    def test_saved_annotations_match_images_and_are_reusable(self):
        data = apply_poisoning(synthetic_dataset("training"), PoisonConfig(kind="label_flip", rate=0.5))
        with tempfile.TemporaryDirectory() as temporary:
            path = save_poisoned_dataset(data, Path(temporary) / "annotated")
            saved = load_arrays(path / "images.npz")
            np.testing.assert_array_equal(saved["sample_ids"], data.original.sample_ids)
            np.testing.assert_array_equal(saved["observed_labels"], data.observed.labels)
            self.assertEqual(load_json(path / "metadata.json")["poisoned_count"], 3)
            self.assertEqual(save_poisoned_dataset(data, Path(temporary) / "annotated"), path)


class QuantumTests(unittest.TestCase):
    def test_paper_circuit_counts_and_endianness(self):
        circuit = build_malik_circuit()
        inputs, weights = parameter_groups(circuit)
        self.assertEqual((len(inputs), len(weights)), (4, 72))
        self.assertEqual(dict(circuit.count_ops()), {"ry": 44, "rz": 42, "cx": 48})
        np.testing.assert_array_equal(counts_to_expectations({"001": 10}), [-1, 1, 1])
        np.testing.assert_array_equal(counts_to_expectations({"100": 10}), [1, 1, -1])
        random = np.random.default_rng(25)
        x, theta = random.uniform(0, np.pi, (1, 4)), random.normal(0, 0.5, 72)
        mapping = dict(zip(inputs, x[0])) | dict(zip(weights, theta))
        state = Statevector.from_instruction(circuit.assign_parameters(mapping))
        expected = [state.expectation_value(SparsePauliOp(text)).real for text in ("IIZIII", "IZIIII", "ZIIIII")]
        model = VQC(backend_config=EXACT)
        np.testing.assert_allclose(model.expectations(x, theta)[0], expected, atol=1e-12)

    def test_all_72_loss_gradients_match_finite_differences(self):
        model = VQC(backend_config=EXACT)
        random = np.random.default_rng(14)
        x, theta = random.uniform(0, np.pi, (1, 4)), random.normal(0, 0.5, 72)
        target = np.array([[0.7, 0.3, 0.0]])
        _, analytic = model.loss_and_gradient(x, target, theta)
        numerical = []
        for index in range(72):
            plus, minus = theta.copy(), theta.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            numerical.append((model.loss(x, target, plus) - model.loss(x, target, minus)) / 2e-5)
        np.testing.assert_allclose(analytic, numerical, atol=2e-8, rtol=1e-5)
        np.testing.assert_allclose(analytic[-11::2], 0, atol=1e-12)

    def test_sampled_and_noisy_evaluation_are_normalized_and_repeatable(self):
        random = np.random.default_rng(2)
        x, theta = random.uniform(0, np.pi, (3, 4)), random.normal(0, 0.5, 12)
        backend = BackendConfig(shots=4096, device="CPU", circuit_batch_size=2,
                                max_parallel_threads=1, one_qubit_error=0.01,
                                two_qubit_error=0.02, readout_error=0.01)
        first, second = VQC(SMALL_CIRCUIT, backend, 5), VQC(SMALL_CIRCUIT, backend, 5)
        probabilities = first.predict_proba(x, theta)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(probabilities, second.predict_proba(x, theta))
        self.assertEqual(first.executor.stats["shots"], 3 * 4096)
        exact_noisy = VQC(SMALL_CIRCUIT, replace(backend, shots=None, readout_error=0), 5)
        sampled_noisy = VQC(SMALL_CIRCUIT, replace(backend, readout_error=0), 5)
        np.testing.assert_allclose(exact_noisy.expectations(x, theta), sampled_noisy.expectations(x, theta), atol=0.06)

    def test_gpu_fallback_is_explicit_and_hardware_requires_configuration(self):
        with patch.object(AerSimulator, "available_devices", return_value=("CPU",)):
            with self.assertWarns(UserWarning):
                model = VQC(SMALL_CIRCUIT, replace(EXACT, device="GPU"))
            self.assertEqual(model.executor.actual_device, "CPU")
            with self.assertRaises(RuntimeError):
                VQC(SMALL_CIRCUIT, replace(EXACT, device="GPU", allow_cpu_fallback=False))
        with self.assertRaises(ValueError):
            BackendConfig(use_ibm_hardware=True)

    @unittest.skipUnless(find_spec("qiskit_ibm_runtime"), "Optional IBM Runtime package is absent")
    def test_ibm_adapter_with_offline_local_backend(self):
        # Exercise the runtime PUB/BitArray interface without contacting IBM.
        with patch("qiskit_ibm_runtime.QiskitRuntimeService") as service:
            service.return_value.backend.return_value = AerSimulator(max_parallel_threads=1)
            backend = BackendConfig(shots=128, use_ibm_hardware=True, ibm_backend_name="offline_test")
            model = VQC(SMALL_CIRCUIT, backend)
            probabilities = model.predict_proba(np.zeros((2, 4)), np.zeros(12))
            np.testing.assert_allclose(probabilities, np.full((2, 3), 1 / 3))
            self.assertEqual(len(model.executor.job_ids), 1)

    def test_training_selects_validation_checkpoint(self):
        random = np.random.default_rng(3)
        features = random.uniform(0, np.pi, (6, 4))
        model = VQC(SMALL_CIRCUIT, EXACT)
        result = fit_vqc(model, features, np.array([0, 0, 1, 1, 2, 2]), features[:3],
                         np.array([0, 1, 2]), TrainingConfig(steps=2, batch_size=2, evaluate_every=1))
        self.assertEqual(result.best_step, int(result.history.loc[result.history.validation_loss.idxmin(), "step"]))
        self.assertEqual(result.weights.shape, (12,))


class UnlearningAndMetricTests(unittest.TestCase):
    def test_study_protocol_ids_separate_code_and_ablations_but_pair_seeds(self):
        import copy
        base = {"path": Path("run_a"), "config": ExperimentConfig().to_dict(),
                "manifest": {"identity": {"code_fingerprint": "version_a", "environment": {"python": "3.12"}}},
                "metrics": {"models": {"clean": {"retained_accuracy": 0.8}}, "retraining_comparisons": {}}}
        other_seed = copy.deepcopy(base)
        other_seed["path"] = Path("run_b")
        other_seed["config"]["seed"] = 1
        other_code = copy.deepcopy(base)
        other_code["path"] = Path("run_c")
        other_code["manifest"]["identity"]["code_fingerprint"] = "version_b"
        ablation = copy.deepcopy(base)
        ablation["path"] = Path("run_d")
        ablation["config"]["unlearning"]["alpha"] = 0.0
        table = results_table([base, other_seed, other_code, ablation])
        self.assertEqual(table.protocol_id.iloc[0], table.protocol_id.iloc[1])
        self.assertNotEqual(table.protocol_id.iloc[0], table.protocol_id.iloc[2])
        self.assertNotEqual(table.protocol_id.iloc[0], table.protocol_id.iloc[3])

    def test_saved_configuration_roundtrip(self):
        original = ExperimentConfig(poison=PoisonConfig(kind="pixel_patch", rate=0.1))
        self.assertEqual(original, ExperimentConfig.from_dict(original.to_dict()))
        self.assertEqual(ExperimentConfig(), ExperimentConfig.from_dict({}))

    def test_summary_keeps_poisoning_rates_separate_and_averages_folds_first(self):
        import pandas as pd
        table = pd.DataFrame({"condition": ["label_flip"] * 6, "model": ["poisoned"] * 6,
                              "poison_rate": [0.1, 0.1, 0.1, 0.1, 0.2, 0.2],
                              "seed": [0, 0, 1, 1, 0, 1], "fold": [0, 1, 0, 1, 0, 0],
                              "retained_accuracy": [0.4, 0.6, 0.8, 1.0, 0.2, 0.4]})
        summary = summarize_runs(table, "retained_accuracy")
        self.assertEqual(len(summary), 2)
        self.assertAlmostEqual(summary.loc[summary.poison_rate == 0.1, "mean"].iloc[0], 0.7)
        self.assertTrue((summary.n_seeds == 2).all())

    def test_global_guided_target_and_uniform_target(self):
        probabilities = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
        target = make_forget_target(probabilities)
        np.testing.assert_allclose(target, [0.3 / 0.65, 0.35 / 0.65, 0])
        np.testing.assert_allclose(make_forget_target(probabilities, method="uniform"), [0.5, 0.5, 0])

    def test_anchor_reference_and_full_unlearning_gradient(self):
        random = np.random.default_rng(9)
        x, original = random.uniform(0, np.pi, (6, 4)), random.normal(0, 0.3, 12)
        labels = np.array([0, 0, 1, 1, 2, 2])
        model = VQC(SMALL_CIRCUIT, EXACT)
        refs = prepare_references(model, x, labels, labels == 2, original, UnlearningConfig())
        self.assertTrue((labels[refs.calibration_indices] == 2).all())
        self.assertFalse(set(refs.anchor_indices) & set(refs.forget_indices))
        theta = original + random.normal(0, 0.1, 12)
        target = make_forget_target(refs.calibration_probabilities)
        arguments = (model, x[refs.forget_indices], x[refs.anchor_indices], target, refs.anchor_probabilities)
        _, gradient = unlearning_loss_and_gradient(*arguments, theta, original, 0.8, 0.2)
        numerical = []
        # Independent scalar objective uses loss(), not the implementation's gradient.
        def objective(weights):
            return (model.loss(x[refs.forget_indices], np.tile(target, (2, 1)), weights)
                    + 0.8 * model.loss(x[refs.anchor_indices], refs.anchor_probabilities, weights)
                    + 0.2 * np.sum((weights - original)**2))
        for index in range(12):
            plus, minus = theta.copy(), theta.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            numerical.append((objective(plus) - objective(minus)) / 2e-5)
        np.testing.assert_allclose(gradient, numerical, atol=2e-8)
        np.testing.assert_allclose(refs.original_weights, original)

    def test_retained_accuracy_and_conditional_kl_have_explicit_meaning(self):
        labels = np.array([0, 1, 2])
        probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1]])
        metrics = classification_metrics(labels, probabilities)
        self.assertEqual(metrics["retained_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["forget_recall"], 0.0)
        reference = np.array([[0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.3, 0.6, 0.1]])
        candidate = np.array([[0.4, 0.2, 0.4], [0.2, 0.4, 0.4], [0.2, 0.4, 0.4]])
        comparison = compare_to_retraining(labels, reference, candidate)
        self.assertAlmostEqual(comparison["conditional_kl_mean"], 0)
        self.assertGreater(comparison["full_kl_mean"], 0)
        self.assertGreater(comparison["candidate_forgotten_mass"], comparison["reference_forgotten_mass"])


class IntegrationTests(unittest.TestCase):
    def test_complete_pipeline_saving_reloading_and_visuals(self):
        small_training = TrainingConfig(steps=1, batch_size=2, evaluate_every=1)
        config = ExperimentConfig(
            circuit=SMALL_CIRCUIT, backend=EXACT, training=small_training,
            poison=PoisonConfig(kind="label_flip", rate=0.5),
            unlearning=UnlearningConfig(training=small_training), label_shuffle_control=True,
        )
        splits = tiny_splits()
        with tempfile.TemporaryDirectory() as temporary:
            result = run_experiment(config, root=temporary, splits=splits)
            path = result["path"]
            self.assertEqual(result["manifest"]["status"], "complete")
            self.assertEqual(result["predictions"]["poisoned_guided__test"].shape, (len(splits.test), 3))
            self.assertEqual(run_experiment(config, root=temporary, splits=splits)["path"], path)
            restored, processor, weights = restore_vqc(path, "poisoned_guided")
            np.testing.assert_allclose(restored.predict_proba(processor.transform(splits.test.images), weights),
                                       result["predictions"]["poisoned_guided__test"], atol=1e-12)
            retraining_ids = load_arrays(path / "retraining_ids.npz")
            forget_ids = result["predictions"]["train_sample_ids"][result["predictions"]["poisoned_forget_mask"]]
            self.assertFalse(set(retraining_ids["poisoned"]) & set(forget_ids))
            import joblib
            transformers = joblib.load(path / "preprocessing.joblib")  # Trusted test-generated file.
            self.assertEqual(set(transformers["retrained_poisoned"].fit_sample_ids_), set(retraining_ids["poisoned"]))
            self.assertFalse(set(transformers["poisoned"].fit_sample_ids_) & set(splits.test.sample_ids))
            self.assertIn("poisoned_rbf_svm", result["metrics"]["models"])
            table = results_table([result])
            study = run_study([config], root=temporary, splits=splits)
            self.assertEqual(len(table), len(study))
            summary = summarize_runs(table, "retained_accuracy")
            poisoned = apply_poisoning(splits.train, config.poison)
            figures = [
                plot_examples(splits.train, poisoned.observed),
                plot_class_balance({"train": splits.train.labels, "test": splits.test.labels}),
                plot_pca_variance(transformers["clean"]),
                plot_training_history(result["history"].query("model == 'clean'")),
                plot_confusion_comparison(result["metrics"]["models"]["poisoned"]["confusion_matrix"],
                                          result["metrics"]["models"]["poisoned_guided"]["confusion_matrix"]),
                plot_forget_confidence(splits.test.labels, result["predictions"]["poisoned__test"],
                                      result["predictions"]["poisoned_guided__test"]),
                plot_metric_summary(summary), draw_circuit(build_malik_circuit(SMALL_CIRCUIT)),
            ]
            curve_summary = summarize_runs(table, "retained_accuracy", ("poison_rate", "model"))
            figures.append(plot_poisoning_curve(curve_summary))
            exports = save_figure(figures[4], Path(temporary) / "figures/confusion",
                                  source_run_ids=[path.name], description="Synthetic integration test")
            self.assertTrue(all(file.exists() for file in exports))
            for figure in figures:
                plt.close(figure)
            sweep = evaluate_noise_sweep(SMALL_CIRCUIT, replace(EXACT, shots=128),
                                         transformers["clean"].transform(splits.test.images),
                                         splits.test.labels, result["weights"]["clean"],
                                         noise_levels=(0, 0.01), seeds=(0,))
            self.assertEqual(len(sweep), 2)
            self.assertTrue((sweep.expectation_mode == "sampled").all())
            # Corrupted/mixed outputs must not silently become a cached result.
            (path / "metrics.json").write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                run_experiment(config, root=temporary, splits=splits)

    def test_sampled_full_paper_circuit_and_shared_retraining_reference(self):
        training = TrainingConfig(steps=1, batch_size=1, evaluate_every=1)
        config = ExperimentConfig(
            backend=BackendConfig(shots=128, device="CPU", max_parallel_threads=1),
            poison=PoisonConfig(kind="pixel_patch", rate=0.5), training=training,
            unlearning=UnlearningConfig(training=training), classical_baselines=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = run_experiment(config, root=temporary, splits=tiny_splits())
            self.assertEqual(result["weights"]["clean"].shape, (72,))
            np.testing.assert_array_equal(result["weights"]["retrained_clean"], result["weights"]["retrained_poisoned"])
            resources = load_json(result["path"] / "resources.json")
            self.assertEqual(resources["clean"]["expectation_mode"], "sampled")
            self.assertGreater(resources["clean"]["resources"]["shots"], 0)
            self.assertEqual(resources["retrained_poisoned"]["additional_circuit_evaluations"], 0)


if __name__ == "__main__":
    unittest.main()
