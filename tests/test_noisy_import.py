"""Import/retraining boundary checks; no research data or model execution."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from zipfile import ZipFile

from src import artifacts, project_workflow as workflow, noisy_training
from src.models.noise import NoiseSetting
from src.models.parallel_aer import AerExecution
from src.models.noisy_training import NoisyBatchService
from tools.import_noisy_results import checked_members, source_digest


class NoisyImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def record(self, complete=True, engine="torch_exact_density_autograd_v1"):
        path = self.root / "saved/study.json"
        artifacts.save_json(path, {"complete": complete, "protocol": {"code_fingerprint": "archived-source",
            "execution": {"engine": engine, "device": "GPU"}}, "supervised": {"one": {}}, "unlearned": {},
            "service_resources": {"torch_runtime": {"gpu": "Tesla T4"}}})
        return path

    def test_completed_import_review_never_requests_fitting_without_force(self):
        path = self.record()
        original = path.read_bytes()
        self.assertFalse(workflow.noisy_fit_required(path, run=False, force=True))
        self.assertFalse(workflow.noisy_fit_required(path, run=True, force=False))
        self.assertTrue(workflow.noisy_fit_required(path, run=True, force=True))
        self.assertTrue(workflow.noisy_fit_required(None, run=True))
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue(workflow.noisy_fit_required(self.record(False), run=True))

    def test_provenance_retains_actual_backend_and_declares_future_aer(self):
        path = self.record()
        original = path.read_bytes()
        frame = workflow.noisy_provenance(path)
        self.assertEqual(frame.iloc[0].training_engine, "torch_exact_density_autograd_v1")
        self.assertEqual(frame.iloc[0].future_training_backend, "Qiskit Aer")
        self.assertEqual(path.read_bytes(), original)

    def test_forced_future_fit_uses_aer_benchmark_and_preserves_original_index(self):
        path = self.record()
        original = path.read_bytes()
        execution = AerExecution(device="CPU", workers=2)
        with patch.object(noisy_training, "benchmark_noisy_training", return_value=(execution, None, None)) as benchmark, \
             patch("tools.noisy_training_runtime.prepare_noisy_execution", side_effect=AssertionError("No foreign resume")):
            result = workflow.prepare_aer_noisy_execution(SimpleNamespace(root=self.root), ["synthetic"], {}, None,
                NoiseSetting("combined", .01), execution, selected_index=path, force=True)
        self.assertIsInstance(result[0], AerExecution)
        self.assertEqual(benchmark.call_args.kwargs["candidates"], [execution])
        self.assertEqual(NoisyBatchService.__module__, "src.models.noisy_training")
        self.assertEqual(path.read_bytes(), original)

    def test_partial_foreign_optimizer_cannot_resume_as_aer(self):
        path = self.record(False)
        with patch("tools.noisy_training_runtime.prepare_noisy_execution", side_effect=AssertionError("No conversion")):
            with self.assertRaisesRegex(ValueError, "another engine"):
                workflow.prepare_aer_noisy_execution(SimpleNamespace(root=self.root), [], {}, None,
                    NoiseSetting("combined", .01), AerExecution(device="CPU"), selected_index=path)

    def test_source_fingerprint_is_verified_without_importing_archive_code(self):
        (self.root / "src").mkdir()
        (self.root / "src/danger.py").write_text("raise AssertionError('must never execute')\n")
        (self.root / "requirements.txt").write_text("synthetic-only\n")
        path = self.root / "source.zip"
        with ZipFile(path, "w") as archive:
            archive.write(self.root / "src/danger.py", "src/danger.py")
            archive.write(self.root / "requirements.txt", "requirements.txt")
        self.assertEqual(source_digest(path), artifacts.source_fingerprint(self.root))

    def test_unsafe_zip_members_and_case_duplicates_are_rejected(self):
        for names in (("../outside",), ("/absolute",), ("a\\b",), ("C:/drive",), ("a.json", "A.json")):
            buffer = BytesIO()
            with ZipFile(buffer, "w") as archive:
                for name in names:
                    archive.writestr(name, "synthetic")
            buffer.seek(0)
            if names == ("a\\b",):
                # Windows ZipInfo normalizes separators when writing. Build a
                # literal untrusted backslash member in both ZIP headers.
                buffer = BytesIO(buffer.getvalue().replace(b"a/b", b"a\\b"))
            with ZipFile(buffer) as archive, self.assertRaises(ValueError):
                checked_members(archive)


if __name__ == "__main__":
    unittest.main()
