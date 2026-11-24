"""Synthetic channel, matched-reference, cache and portable noise-report checks."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile
import unittest

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator

from src.artifacts import (create_run, finish_run, load_json, load_run, save_arrays, save_json)
from src.config import DEFAULT_SEED, BackendConfig, CircuitConfig, ExperimentConfig
from src.evaluation.metrics import classification_metrics
from src.evaluation.noise_analysis import export_noise_study, noise_figures, noise_tables
from src.evaluation.saved_results import protect_review
from src.models.noise import (DensityReadoutExecutor, NoiseExecution, NoiseSetting, build_gate_noise,
                              class_probabilities, noise_settings, readout_distribution)
from src.models.vqc import VQC
from src.noise_study import (benchmark_noise, load_noise_inputs, load_noise_predictions,
                             noise_download_bundle, noise_plan, run_noise_study)


def fixture(root):
    config = ExperimentConfig(circuit=CircuitConfig(reuploads=1, ansatz_reps=1),
                              backend=BackendConfig(shots=None, device="CPU", max_parallel_threads=1))
    rng = np.random.default_rng(DEFAULT_SEED)
    features, labels = rng.uniform(0, np.pi, (6, 4)), np.repeat([0, 1, 2], 2)
    ids = np.array([f"synthetic_{i}" for i in range(len(labels))])
    model, runs = VQC(config.circuit, config.backend), {}
    for role in ("original", "retrained", "guided", "uniform"):
        weights = rng.normal(0, 0.2, config.circuit.n_parameters)
        probabilities = model.predict_proba(features, weights)
        path, _ = create_run(root / "results/runs/synthetic", {"config": config.to_dict(), "role": role})
        save_arrays(path / "features.npz", test=features)
        save_arrays(path / "weights.npz", model=weights)
        save_arrays(path / "predictions.npz", model__test=probabilities, test_labels=labels, test_sample_ids=ids)
        save_json(path / "metrics.json", {"models": {"model": classification_metrics(labels, probabilities)}})
        (path / "history.csv").write_text("step\n0\n")
        finish_run(path, "complete")
        runs[role] = load_run(path)
    condition = {"config": config.to_dict(), "models": {"clean": runs["original"], "retrained_clean": runs["retrained"]},
                 "unlearned": {"clean_guided": runs["guided"], "clean_uniform": runs["uniform"]},
                 "manifest_path": root / "results/conditions/synthetic.json",
                 "poisoning": {"actual_fraction_all_training": 0.0, "actual_fraction_eligible": 0.0}}
    return SimpleNamespace(root=root, conditions={"clean": condition}, split_paths={0: "synthetic"},
                           report_path=root / "original_report", source_fingerprint="synthetic")


class NoiseStudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.review = fixture(self.root)
        self.models, self.bindings = load_noise_inputs(self.review, folds=(0,))
        self.execution = NoiseExecution("CPU", 12, 1, 1024)

    def test_noise_plan_deduplicates_zero_and_checks_parameters(self):
        settings = noise_settings()
        self.assertEqual(len(settings), 21)
        self.assertEqual(sum(s.level == 0 for s in settings), 1)
        table = noise_plan(self.models, self.bindings)
        self.assertEqual(len(table), 21)
        self.assertEqual(NoiseSetting("combined", 0.01).gate_setting(), NoiseSetting("depolarizing", 0.01))
        self.assertEqual(NoiseSetting("readout", 0.01).gate_setting(), NoiseSetting())
        with self.assertRaises(ValueError):
            noise_settings(levels=(0, 0.6))
        with self.assertRaises(ValueError):
            noise_plan(self.models, self.bindings, shots=(2048,))

    def test_channels_match_single_qubit_analytic_expectations(self):
        p = 0.2
        for family, expected in (("bit_flip", 1.0), ("phase_flip", (1 - 2*p)**2),
                                 ("depolarizing", (1-p)**2)):
            circuit = QuantumCircuit(1)
            circuit.h(0)
            circuit.ry(0, 0)
            circuit.barrier()
            circuit.ry(0, 0)
            circuit.save_expectation_value(Pauli("X"), [0], label="x")
            circuit.save_expectation_value(Pauli("Z"), [0], label="z")
            result = AerSimulator(method="density_matrix", noise_model=build_gate_noise(NoiseSetting(family, p))).run(
                                  circuit, shots=1, seed_simulator=DEFAULT_SEED).result()
            self.assertAlmostEqual(result.data()["x"], expected, places=12)
            self.assertAlmostEqual(result.data()["z"], 0.0, places=12)

    def test_joint_readout_order_and_correlated_shot_sampling(self):
        joint = np.zeros((2, 8))
        joint[0, 1], joint[1, 6] = 1, 1
        _, expectations = class_probabilities(joint)
        np.testing.assert_array_equal(expectations, [[-1, 1, 1], [1, -1, -1]])
        _, noisy = class_probabilities(joint, readout_error=0.1)
        np.testing.assert_allclose(noisy, 0.8 * expectations, atol=1e-14)
        np.testing.assert_allclose(readout_distribution(joint, 0.5), np.full((2, 8), 1/8))
        correlated = np.zeros((1, 8)); correlated[0, [0, 7]] = 0.5
        first, drawn = class_probabilities(correlated, shots=2048)
        np.testing.assert_array_equal(drawn[:, 0], drawn[:, 1])
        np.testing.assert_array_equal(drawn[:, 0], drawn[:, 2])
        second, _ = class_probabilities(correlated, shots=2048)
        np.testing.assert_array_equal(first, second)

    def test_zero_noise_joint_probabilities_reproduce_existing_vqc(self):
        item = next(iter(self.models.values()))
        executor = DensityReadoutExecutor(item.circuit, NoiseSetting(), self.execution)
        joint = executor.run(item.features, np.broadcast_to(item.run["weights"]["model"],
                                                           (len(item.features), item.circuit.n_parameters)))
        actual, _ = class_probabilities(joint)
        np.testing.assert_allclose(actual, item.run["predictions"]["model__test"], atol=1e-12, rtol=0)

    def test_density_benchmark_passes_and_reuses_compatible_cache(self):
        model = next(iter(self.models.values()))
        selected, table, _ = benchmark_noise(model, self.root, require_gpu_device=False, candidates=[self.execution])
        self.assertEqual(selected, self.execution)
        self.assertEqual(table.iloc[0].status, "passed")
        with patch("src.noise_study.DensityReadoutExecutor", side_effect=AssertionError("no new simulation")):
            second, _, _ = benchmark_noise(model, self.root, require_gpu_device=False, candidates=[self.execution])
        self.assertEqual(second, selected)

    def test_all_roles_use_matched_noisy_reference_and_preserve_saved_models(self):
        parents = {file: file.read_bytes() for model in self.models.values() for file in model.run["path"].rglob("*") if file.is_file()}
        with protect_review():
            index_file = run_noise_study(self.review, self.models, self.bindings, self.execution,
                levels=(0, 0.01), families=("depolarizing", "combined", "readout"), allow_execution=True)
            import src.training
            with self.assertRaisesRegex(RuntimeError, "blocks fitting"):
                src.training.fit_vqc()
        index, tables = noise_tables(index_file, self.root)
        self.assertTrue(index["complete"])
        self.assertEqual(index["density_evaluated"], 8)  # four models times two distinct gate channels
        self.assertEqual(len(index["prediction_records"]), 32)
        self.assertTrue(tables["coverage"].completed.all())
        self.assertEqual(len(tables["summary"]), 32)
        reference = tables["summary"].query("role == 'retrained'")
        np.testing.assert_array_equal(reference.full_kl_mean, 0)
        self.assertEqual(len(tables["unlearning_effects"]), 16)
        zero = tables["unlearning_effects"].query("family == 'none'")
        np.testing.assert_array_equal(zero.noise_interaction_retained_accuracy, 0)
        for file, original in parents.items():
            self.assertEqual(file.read_bytes(), original)
        # Re-running inference reuses density caches; no supervised fitting or optimizer continuation.
        with patch("src.noise_study.DensityReadoutExecutor", side_effect=AssertionError("no new simulation")):
            same = run_noise_study(self.review, self.models, self.bindings, self.execution,
                levels=(0, 0.01), families=("depolarizing", "combined", "readout"), allow_execution=True)
        self.assertEqual(same, index_file)
        self.assertEqual(load_json(same)["density_loaded"], 8)
        self.assertEqual(load_json(same)["previous_executions"][-1]["density_evaluated"], 8)

    def test_forced_evaluation_preserves_old_attempt_and_changed_noise_starts_separately(self):
        arguments = (self.review, self.models, self.bindings, self.execution)
        first = run_noise_study(*arguments, levels=(0, 0.01), families=("depolarizing",), shots=(None,), allow_execution=True)
        original = first.read_bytes()
        forced = run_noise_study(*arguments, levels=(0, 0.01), families=("depolarizing",), shots=(None,),
                                 allow_execution=True, force_evaluate=True)
        self.assertNotEqual(first, forced)
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual(load_json(forced)["density_evaluated"], 8)
        changed = run_noise_study(*arguments, levels=(0, 0.02), families=("depolarizing",), shots=(None,), allow_execution=True)
        self.assertNotEqual(first, changed)
        self.assertEqual(load_json(changed)["density_loaded"], 4)  # identical zero-noise controls are reused

    def test_interrupted_noise_evaluation_resumes_committed_batches(self):
        def interrupt(event):
            raise KeyboardInterrupt("synthetic interruption")
        with self.assertRaises(KeyboardInterrupt):
            run_noise_study(self.review, self.models, self.bindings, self.execution,
                 levels=(0, 0.01), families=("depolarizing",), shots=(None,), allow_execution=True, progress=interrupt)
        index_file = next((self.root / "results/noise_studies").glob("*/study.json"))
        self.assertFalse(load_json(index_file)["complete"])
        self.assertGreater(len(load_json(index_file)["density_records"]), 0)
        run_noise_study(self.review, self.models, self.bindings, self.execution,
                 levels=(0, 0.01), families=("depolarizing",), shots=(None,), allow_execution=True)
        self.assertTrue(load_json(index_file)["complete"])
        self.assertGreater(load_json(index_file)["density_loaded"], 0)

    def test_exports_are_portable_and_corrupt_predictions_are_rejected(self):
        index_file = run_noise_study(self.review, self.models, self.bindings, self.execution,
                 levels=(0, 0.01), families=("combined",), shots=(None,), allow_execution=True)
        index, tables = noise_tables(index_file, self.root)
        figures = noise_figures(tables)
        self.assertEqual(len(figures), 2)
        destination = export_noise_study(index_file, self.root, figures)
        for figure in figures.values():
            plt.close(figure)
        self.assertTrue((destination / "summary.csv").exists())
        self.assertEqual(len(list((destination / "figures").glob("*.png"))), 2)
        for name in ("src/noise_study.py", "src/models/noise.py", "src/evaluation/noise_analysis.py", "main_project.ipynb", "README.md"):
            file = self.root / name; file.parent.mkdir(parents=True, exist_ok=True); file.write_text("synthetic bundle fixture")
        bundle = noise_download_bundle(index_file, self.root)
        with ZipFile(bundle) as archive:
            self.assertIsNone(archive.testzip())
            self.assertTrue(any(name.endswith("joint.npz") for name in archive.namelist()))
            self.assertFalse(any(name.startswith("results/runs/") for name in archive.namelist()))
        prediction = self.root / index["prediction_records"][0]["file"]
        prediction.write_bytes(b"synthetic corruption")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            load_noise_predictions(index_file, self.root)
        with self.assertRaisesRegex(RuntimeError, "Enable RUN_NOISE_STUDY"):
            run_noise_study(self.review, self.models, self.bindings, self.execution)


if __name__ == "__main__":
    unittest.main()
