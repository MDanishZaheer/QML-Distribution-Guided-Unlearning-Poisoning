"""Review must fail closed rather than train when historical artifacts are missing."""

from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from src.config import ExperimentConfig
from src.artifacts import fingerprint
from src.evaluation.saved_results import SavedResultsReview, protect_review, require_review_flags


class SavedResultsReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.review = SavedResultsReview.__new__(SavedResultsReview)
        self.review.root = self.root
        self.review.runs = {}
        self.review.conditions = {}
        self.review.source_fingerprint = "original-code"

    def saved_run(self):
        path = self.root / "saved"
        path.mkdir()
        for name in ("config", "metrics"):
            (path / f"{name}.json").write_text("{}")
        np.savez(path / "weights.npz", model=np.zeros(2))
        np.savez(path / "predictions.npz", test_labels=np.array([0]))
        (path / "history.csv").write_text("step,objective\n0,1.0\n")
        manifest = {"status": "complete", "identity": {
            "code_fingerprint": "original-code", "environment": {"platform": "Linux"}},
            "artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}}
        (path / "manifest.json").write_text(json.dumps(manifest))
        return path

    def test_completed_linux_numeric_artifacts_load_without_environment_match(self):
        path = self.saved_run()
        run = self.review.load_saved_run(path)
        self.assertEqual(run["manifest"]["identity"]["environment"]["platform"], "Linux")
        self.assertIs(self.review.load_saved_run(path), run)

    def test_missing_or_corrupt_artifact_stops_without_creating_a_run(self):
        with self.assertRaises(FileNotFoundError):
            self.review.load_saved_run("missing")
        self.assertFalse((self.root / "missing").exists())
        path = self.saved_run()
        (path / "weights.npz").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.review.load_saved_run(path)
        self.assertFalse(self.review.runs)

    def test_incompatible_source_and_external_paths_are_rejected(self):
        path = self.saved_run()
        self.review.source_fingerprint = "another-code"
        with self.assertRaisesRegex(ValueError, "different source"):
            self.review.load_saved_run(path)
        with self.assertRaisesRegex(ValueError, "leaves the project"):
            self.review.local_path("../outside")

    def test_changed_condition_settings_never_fall_back_to_training(self):
        config = ExperimentConfig()
        with self.assertRaisesRegex(ValueError, "never trains"):
            self.review.condition(config, None)
        self.review.conditions[fingerprint(config.to_dict())] = {"split_fingerprint": "original"}
        class WrongSplit:
            def fingerprint(self):
                return "different"
        with self.assertRaisesRegex(ValueError, "different data"):
            self.review.condition(config, WrongSplit())

    def test_enabled_execution_flag_is_rejected(self):
        require_review_flags(FORCE_RETRAIN=False, RUN_UNLEARNING=False)
        with self.assertRaisesRegex(RuntimeError, "RUN_UNLEARNING"):
            require_review_flags(FORCE_RETRAIN=False, RUN_UNLEARNING=True)

    def test_saved_features_support_existing_pca_display_without_fitting(self):
        from src.evaluation.plots import plot_pca_variance
        import matplotlib.pyplot as plt
        path = self.root / "gold"
        path.mkdir()
        self.review.gold = {"path": path}
        parts, arrays = {}, {}
        for name in ("train", "validation", "test"):
            parts[name] = SimpleNamespace(sample_ids=np.array([name]), labels=np.array([0]))
            arrays.update({f"{name}_sample_ids": parts[name].sample_ids,
                           f"{name}_labels": parts[name].labels,
                           f"{name}_features": np.zeros((1, 4))})
        np.savez(path / "clean_features.npz", **arrays)
        diagnostics = {"explained_variance_ratio": [0.2, 0.1, 0.08, 0.05],
                       "total_explained_variance": 0.43, "n_fit_samples": 1}
        (path / "clean_preprocessing.json").write_text(json.dumps(diagnostics))
        with protect_review():
            features = self.review.gold_features(SimpleNamespace(**parts))
            figure = plot_pca_variance(features["preprocessing"])
            self.assertEqual(len(figure.axes), 1)
            plt.close(figure)

    def test_guard_blocks_execution_and_restores_functions_when_closed(self):
        import src.checkpoints as checkpoints
        import src.experiments as experiments
        import src.experiment_stages as stages
        from src.data.preprocessing import ImagePreprocessor
        from src.models.vqc import VQC
        original = experiments.fit_vqc
        with protect_review():
            calls = [experiments.fit_vqc, experiments.fit_classical_baselines,
                     stages.fit_vqc, stages.unlearn, stages.fit_classical_baselines,
                     checkpoints.retire_pilot_checkpoints, checkpoints.CheckpointSession().open,
                     ImagePreprocessor.fit, VQC.predict_proba]
            for function in calls:
                with self.subTest(function=function):
                    with self.assertRaisesRegex(RuntimeError, "no training has started"):
                        function()
        self.assertIs(experiments.fit_vqc, original)


if __name__ == "__main__":
    unittest.main()
