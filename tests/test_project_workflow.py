"""Local migration checks with synthetic artifacts, never research data/results."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import ast
import hashlib
import json
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from test_feature_randomization import fixture
from src import artifacts, project_workflow as workflow
from src.data.dataset import ImageDataset, DatasetSplits
from src.data.feature_randomization import UnitInputPreprocessor
from src.evaluation import fresh_analysis, ablation_analysis, noise_analysis, noisy_training_analysis
from src.fresh_study import FreshStudy, run_fresh_study
from src.models.parallel_aer import AerExecution
from src.models.vqc import VQC


class ProjectWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name).resolve()
        cls.splits, cls.config = fixture()
        # Like real MNIST IDs, the pool fixes one string dtype before subsetting.
        cls.splits = DatasetSplits(**{
            name: ImageDataset(part.images, part.labels, part.sample_ids.astype("U32"))
            for name in ("train", "validation", "test")
            for part in (getattr(cls.splits, name),)
        })
        cls.execution = AerExecution(device="CPU", workers=2, threads=1, circuit_batch_size=1024)
        cls.gold = workflow.train_gold_models(cls.root, {0: cls.splits}, cls.execution, config=cls.config)
        cls.gold_bytes = (cls.gold[0]["path"] / "weights.npz").read_bytes()
        cls.study, cls.index = run_fresh_study(cls.root, {0: cls.splits}, cls.execution,
                                              base=cls.config, rates=(.25, .5))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_gold_preview_is_reused_by_full_pipeline_without_overwrite(self):
        run = workflow.clean_runs(self.study)[0]
        self.assertEqual(run["path"], self.gold[0]["path"])
        self.assertEqual((run["path"] / "weights.npz").read_bytes(), self.gold_bytes)
        table = workflow.baseline_table(self.gold)
        self.assertEqual(table.model.tolist(), ["gold_vqc"])

    def test_poison_review_uses_saved_bytes_and_rejects_changed_inputs(self):
        key = "fold_0_feature_randomization_50pct"
        condition = self.study.conditions[key]
        directory = self.root / condition["annotated_dataset"]
        file = directory / "images.npz"
        original_bytes = file.read_bytes()
        with workflow.protect_saved_review(), patch(
            "src.data.feature_randomization.feature_variant", side_effect=AssertionError("Never regenerate")
        ):
            variant = workflow.restore_poison_variant(self.study, key, split=self.splits)
        self.assertEqual(variant.metadata(), condition["poisoning"])
        arrays = artifacts.load_arrays(file)
        changed = dict(arrays)
        changed["images"] = arrays["images"].copy()
        selected = int(np.flatnonzero(arrays["is_poisoned"])[0])
        changed["images"][selected, 0, 0] = np.nextafter(changed["images"][selected, 0, 0], np.inf)
        try:
            artifacts.save_arrays(file, **changed)
            with self.assertRaisesRegex(ValueError, "fingerprints"):
                workflow.restore_poison_variant(self.study, key, split=self.splits)
        finally:
            file.write_bytes(original_bytes)

    def test_portable_review_keeps_archived_identity_and_blocks_fitting(self):
        original = self.index.read_bytes()
        with workflow.protect_saved_review(RUN_PRIMARY_STUDY=False), \
             patch.object(artifacts, "source_fingerprint", return_value="different-review-source"):
            reviewed = workflow.load_completed_study(self.root, progress=None)
            self.assertEqual(reviewed.source_fingerprint, self.study.source_fingerprint)
            self.assertEqual(workflow.study_config(reviewed).to_dict(), self.config.to_dict())
            table = fresh_analysis.study_tables(reviewed.select_rates((.5,)))
            self.assertFalse(table["metrics"].empty)
            features, diagnostics, view = workflow.cached_fold_features(reviewed)
            self.assertEqual(features["test"].shape[1], 4)
            figure = fresh_analysis.diagnostic_figure(pd.DataFrame([
                {"attack": attack, "poison_rate": .5, "mean_input_l2": 0,
                 "mean_pca_l2": 0, "mean_angle_l2": 0}
                for attack in ("label_flip", "feature_randomization")
            ]))
            plt.close(figure)
            self.assertEqual(view.diagnostics(), diagnostics)
            with self.assertRaisesRegex(RuntimeError, "blocks fitting"):
                UnitInputPreprocessor().fit(self.splits.train.images)
            with self.assertRaisesRegex(RuntimeError, "blocks fitting"):
                VQC(self.config.circuit, self.config.backend).predict_proba(
                    features["test"], self.gold[0]["weights"]["model"])
        self.assertEqual(self.index.read_bytes(), original)

    def test_missing_and_ambiguous_studies_stop_without_training_or_writes(self):
        with TemporaryDirectory() as other:
            root = Path(other)
            with patch("src.fresh_study.run_fresh_study", side_effect=AssertionError("Never fit")):
                with self.assertRaisesRegex(ValueError, "No model has been trained"):
                    workflow.load_completed_study(root, progress=None)
            self.assertFalse((root / "results").exists())
        with self.assertRaisesRegex(RuntimeError, "requires these flags False"):
            workflow.protect_saved_review(RUN_PRIMARY_STUDY=True)

    def test_saved_split_restoration_checks_raw_fingerprints_without_fitting(self):
        parts = [self.splits.train, self.splits.validation, self.splits.test]
        raw = ImageDataset(np.concatenate([p.images for p in parts]),
                           np.concatenate([p.labels for p in parts]),
                           np.concatenate([p.sample_ids for p in parts]))
        with workflow.protect_saved_review(), patch.object(workflow, "load_mnist", return_value=(raw, None)):
            restored = workflow.restore_study_splits(self.study)
        self.assertEqual(restored[0].fingerprint(), self.splits.fingerprint())
        changed = raw.images.copy()
        changed[0, 0, 0] = (int(changed[0, 0, 0]) + 1) % 256
        foreign = ImageDataset(changed, raw.labels, raw.sample_ids)
        with patch.object(workflow, "load_mnist", return_value=(foreign, None)):
            with self.assertRaisesRegex(ValueError, "fingerprint|membership"):
                workflow.restore_study_splits(self.study)

    def test_partial_noisy_index_is_discovered_without_loading_optimizer_or_fitting(self):
        relative = "results/noisy_training/synthetic_partial/study.json"
        file = self.root / relative
        record = {"complete": False, "protocol": {
            "original_report": self.index.relative_to(self.root).as_posix(),
            "original_source": self.study.source_fingerprint, "plan": [{}, {}, {}, {}]},
            "supervised": {"synthetic_model": {"run": "not-read-by-inventory"}}, "unlearned": {}}
        artifacts.save_json(file, record)
        try:
            paths, table = workflow.extension_inventory(self.study)
            self.assertEqual(paths["noisy_training"], file)
            row = table.set_index("stage").loc["noisy_training"]
            self.assertEqual(row.saved_fits, 1)
            self.assertEqual(row.planned_fits, 4)
            self.assertFalse(row.complete)
            with self.assertRaisesRegex(ValueError, "leaves the project"):
                workflow.optional_study_index(self.study, "noisy_training", self.root.parent / "outside.json")
        finally:
            file.unlink()

    def test_cpu_fallback_and_explicit_unavailable_gpu(self):
        with patch("qiskit_aer.AerSimulator.available_devices", return_value=("CPU",)):
            execution = workflow.local_execution()
            self.assertEqual(execution.device, "CPU")
            self.assertLessEqual(execution.workers, 2)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                workflow.local_execution(device="GPU")
        with patch("qiskit_aer.AerSimulator.available_devices", return_value=("CPU", "GPU")), \
             patch.object(workflow, "gpu_sample", return_value={"memory.total": 6144, "memory.free": 1000}):
            execution = workflow.local_execution()
            self.assertEqual(execution.device, "GPU")
            self.assertLess(execution.memory_mb, 6144)

    def test_current_notebook_is_unexecuted_ordered_and_uses_local_paths(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "main_project.ipynb").read_text(encoding="utf-8"))
        sections, sources = [], []
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            sources.append(source)
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse(source)
                self.assertLessEqual(len(source.splitlines()), 28)
            elif source.startswith("## "):
                sections.append(int(source.split()[1].split(".")[0]))
        self.assertEqual(sections, list(range(1, 16)))
        self.assertEqual(notebook["cells"][-1]["source"], [])
        text = "\n".join(sources)
        self.assertNotIn("/content/", text)
        self.assertNotIn("colab", text.lower())
        self.assertIn("SAVED_RESULTS_ONLY = True", text)
        for flag in ("RUN_PRIMARY_STUDY", "RUN_ABLATIONS", "RUN_FIXED_WEIGHT_NOISE", "RUN_NOISY_TRAINING",
                     "FORCE_RETRAIN", "DOWNLOAD_MNIST", "USE_IBM_HARDWARE"):
            self.assertIn(flag + " = False", text)

    def test_export_reuses_loaded_tables_without_reading_predictions_twice(self):
        # A small explicit analysis payload exercises all three export paths.
        summary = pd.DataFrame([{"run_id": "synthetic", "status": "cached"}])
        coverage = pd.DataFrame([{"completed": True}])
        tables = {"summary": summary, "coverage": coverage}
        file = self.root / "synthetic_export/study.json"
        artifacts.save_json(file, {"complete": True, "protocol": {"plan": [{}]},
                                   "supervised": {}, "unlearned": {}})
        for module, exporter, loader in (
            (ablation_analysis, ablation_analysis.export_study, "study_tables"),
            (noise_analysis, noise_analysis.export_noise_study, "noise_tables"),
            (noisy_training_analysis, noisy_training_analysis.export_noisy_training, "noisy_training_tables"),
        ):
            with patch.object(module, loader, side_effect=AssertionError("Do not read twice")):
                destination = exporter(file, self.root, tables=tables,
                                       destination=self.root / "synthetic_export" / module.__name__)
            self.assertTrue((destination / "summary.csv").is_file())
            self.assertTrue((destination / "provenance.json").is_file())


if __name__ == "__main__":
    unittest.main()
