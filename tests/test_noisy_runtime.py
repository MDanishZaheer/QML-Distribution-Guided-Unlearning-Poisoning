"""Recovery instrumentation must leave scientific checkpoint identities unchanged."""

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np

from test_noisy_training import fixture
from src import artifacts, noisy_training, training, checkpoints
from src.models.noise import NoiseSetting
from src.models.parallel_aer import AerExecution
from src.noisy_training import load_training_problems, noisy_training_execution, run_noisy_training
from tools.noisy_training_runtime import (completed_fresh_index, prepare_noisy_execution,
    completed_extension_index, noisy_training_monitor, include_runtime_support, _ManifestCatalog)


class NoisyRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.review = fixture(self.root)
        self.problems, self.models, self.bindings = load_training_problems(self.review, folds=(0,))
        self.execution = AerExecution(device="CPU", workers=2, threads=1, circuit_batch_size=1024)
        self.noise = NoiseSetting("combined", .01)
        self.messages = []

    def run_pipeline(self, monitored=False):
        with noisy_training_execution(self.root):
            if monitored:
                with noisy_training_monitor(self.root, heartbeat_seconds=.02, emit=self.messages.append) as observer:
                    path = run_noisy_training(self.review, self.problems, self.models, self.bindings,
                        self.execution, setting=self.noise, allow_training=True)
                    audit = observer.audit
            else:
                path = run_noisy_training(self.review, self.problems, self.models, self.bindings,
                    self.execution, setting=self.noise, allow_training=True)
                audit = None
        return path, audit

    def test_existing_completed_fits_reuse_same_identity_and_bytes_with_monitor(self):
        path, _ = self.run_pipeline()
        runs = [p for stage in ("noisy_supervised", "noisy_unlearning")
                for directory in (self.root / "results/runs" / stage).glob("*") if directory.is_dir()
                for p in directory.rglob("*") if p.is_file()]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in runs}
        source = artifacts.source_fingerprint(self.root)
        with patch.object(training, "fit_vqc", side_effect=AssertionError("Completed model must not fit")), \
             patch("src.unlearning.methods.unlearn", side_effect=AssertionError("Completed model must not unlearn")):
            same, audit = self.run_pipeline(monitored=True)
        self.assertEqual(path, same)
        self.assertEqual(source, artifacts.source_fingerprint(self.root))
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in runs})
        record = artifacts.load_json(audit / "provenance.json")
        self.assertFalse(record["scientific_source_identity_changed"])
        self.assertFalse(record["optimizer_or_execution_settings_changed"])
        self.assertEqual(artifacts.load_json(audit / "final_progress.json")["completed_fits"], 4)
        self.assertTrue(any("START" in message for message in self.messages))
        self.assertTrue(any("loaded" in message for message in self.messages))

    def test_interrupted_old_fit_resumes_with_new_step_heartbeat_and_same_run_path(self):
        old_fit = training.fit_vqc
        def interrupted(*args, **kwargs):
            callback = kwargs.get("callback")
            def stop(row):
                if callback:
                    callback(row)
                if row["step"] == 1:
                    raise KeyboardInterrupt("synthetic earlier session")
            kwargs["callback"] = stop
            return old_fit(*args, **kwargs)
        with patch.object(training, "fit_vqc", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_pipeline()
        old = set((self.root / "results/runs/noisy_supervised").glob("*/manifest.json"))
        source = artifacts.source_fingerprint(self.root)
        path, audit = self.run_pipeline(monitored=True)
        self.assertEqual(source, artifacts.source_fingerprint(self.root))
        self.assertEqual(old, set((self.root / "results/runs/noisy_supervised").glob("*/manifest.json")))
        record = artifacts.load_json(path)
        self.assertTrue(all(row["status"] == "resumed" for row in record["supervised"].values()))
        self.assertTrue(any("STEP" in message for message in self.messages))
        self.assertTrue(any("HEARTBEAT" in message for message in self.messages))
        self.assertTrue((audit / "latest_progress.json").is_file())
        for item in record["supervised"].values():
            resume = artifacts.load_json(self.root / item["run"] / "checkpoints/model.resume.json")
            self.assertEqual(resume["step"], 2)

    def test_new_monitored_fits_have_identical_weights_to_uninstrumented_fit(self):
        path, audit = self.run_pipeline(monitored=True)
        observed = artifacts.load_json(path)
        with TemporaryDirectory() as other:
            root = Path(other).resolve()
            review = fixture(root)
            problems, models, bindings = load_training_problems(review, folds=(0,))
            with noisy_training_execution(root):
                original_path = run_noisy_training(review, problems, models, bindings, self.execution,
                    setting=self.noise, allow_training=True)
            original = artifacts.load_json(original_path)
            for group in ("supervised", "unlearned"):
                for key, row in observed[group].items():
                    actual = artifacts.load_arrays(self.root / row["run"] / "weights.npz")["model"]
                    expected = artifacts.load_arrays(root / original[group][key]["run"] / "weights.npz")["model"]
                    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        self.assertTrue(any("INITIAL STATE" in message for message in self.messages))
        self.assertTrue(any("AER CPU:" in message for message in self.messages))
        self.assertTrue(any("initializing Aer" in message for message in self.messages))
        self.assertGreater(len(artifacts.load_json(audit / "final_progress.json")["aer_services"]), 0)

    def test_manifest_catalog_reads_once_and_keeps_exact_matching(self):
        directory = self.root / "results/runs/noisy_supervised"
        for name, state, source in (("one", "complete", "a"), ("two", "failed", "a"), ("three", "retired", "a"), ("other", "complete", "b")):
            artifacts.save_json(directory / name / "manifest.json", {
                "status": state, "identity": {"source": source}, "created_utc": name})
        expected = checkpoints.matching_manifests(directory, {"source": "a"})
        catalog = _ManifestCatalog(self.root, self.messages.append)
        with patch.object(artifacts, "load_json", wraps=artifacts.load_json) as read:
            first = catalog(directory, {"source": "a"})
            count = read.call_count
            for _ in range(10):
                self.assertEqual(catalog(directory, {"source": "a"}), first)
            self.assertEqual(read.call_count, count)
        self.assertEqual(first, expected)
        self.assertEqual(catalog(directory, {"source": "foreign"}), [])

    def test_recorded_execution_is_pinned_and_passing_benchmark_not_repeated(self):
        path, _ = self.run_pipeline()
        index = artifacts.load_json(path)
        performance = self.root / "results/noisy_training/performance/synthetic"
        artifacts.save_json(performance / "benchmark.json", {
            "identity": {"source": artifacts.source_fingerprint(self.root),
                         "environment": noisy_training.execution_environment(), "noise": asdict(self.noise),
                         "original_problem": self.problems[0].original["path"].name},
            "selected": asdict(self.execution), "candidates": [{**asdict(self.execution), "status": "passed"}]})
        index["performance_directory"] = performance.relative_to(self.root).as_posix()
        artifacts.save_json(path, index)
        with patch.object(noisy_training, "benchmark_noisy_training", side_effect=AssertionError("Do not repeat benchmark")):
            execution, table, _ = prepare_noisy_execution(self.review, self.problems, self.models, self.bindings,
                self.noise, require_gpu_device=False)
        self.assertEqual(execution, self.execution)
        self.assertEqual(table.iloc[0].status, "passed")
        with patch.object(noisy_training, "execution_environment", return_value={"different": "runtime"}):
            with self.assertRaisesRegex(ValueError, "do not match"):
                prepare_noisy_execution(self.review, self.problems, self.models, self.bindings,
                    self.noise, require_gpu_device=False)

    def test_fresh_index_selection_is_load_only_and_ambiguous_selection_stops(self):
        def save(name, complete):
            path = self.root / "results/fresh_studies" / name / "study.json"
            artifacts.save_json(path, {"complete": complete, "protocol": {"label": "feature_randomization_v1"}})
            return path
        first = save("first", True)
        save("unfinished", False)
        self.assertEqual(completed_fresh_index(self.root), first)
        save("second", True)
        with self.assertRaisesRegex(ValueError, "No fitting was started"):
            completed_fresh_index(self.root)
        self.assertEqual(completed_fresh_index(self.root, first.relative_to(self.root)), first)
        with self.assertRaisesRegex(ValueError, "leaves the project"):
            completed_fresh_index(self.root, "../foreign/study.json")

    def test_portable_archive_carries_separate_helper_provenance(self):
        _, audit = self.run_pipeline(monitored=True)
        (self.root / "tools").mkdir()
        (self.root / "tools/noisy_training_runtime.py").write_text("# synthetic helper fixture\n")
        archive = self.root / "portable.zip"
        with ZipFile(archive, "w", ZIP_DEFLATED):
            pass
        include_runtime_support(archive, self.root)
        include_runtime_support(archive, self.root)
        with ZipFile(archive) as bundle:
            self.assertIsNone(bundle.testzip())
            names = bundle.namelist()
            self.assertEqual(names.count("tools/noisy_training_runtime.py"), 1)
            self.assertIn((audit / "provenance.json").relative_to(self.root).as_posix(), names)

    def test_completed_extensions_relink_without_running_or_using_other_parent(self):
        def save(name, complete=True, parent=None):
            path = self.root / "results/ablations" / name / "study.json"
            artifacts.save_json(path, {"complete": complete, "protocol": {
                "original_report": parent or self.review.report_path.relative_to(self.root).as_posix(),
                "original_source": self.review.source_fingerprint}})
            return path
        self.assertIsNone(completed_extension_index(self.review, "ablations"))
        save("other", parent="results/fresh_studies/different/study.json")
        save("unfinished", complete=False)
        first = save("one")
        self.assertEqual(completed_extension_index(self.review, "ablations"), first)
        save("two")
        with self.assertRaisesRegex(ValueError, "Select one completed"):
            completed_extension_index(self.review, "ablations")
        self.assertEqual(completed_extension_index(self.review, "ablations", first), first)


if __name__ == "__main__":
    unittest.main()
