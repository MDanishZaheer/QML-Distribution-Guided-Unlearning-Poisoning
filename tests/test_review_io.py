"""The storage read optimization must preserve strict saved-input validation."""

from hashlib import sha256
from pathlib import Path
from threading import Barrier
from time import monotonic
import json
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from src.evaluation.review_io import build_review_input_archive, prepare_review_inputs
from src.evaluation.saved_results import SavedResultsReview, protect_review


class ReviewInputIOTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "project"
        self.root.mkdir()
        self.cache = self.root.parent / "local_cache"
        self.review = SavedResultsReview.__new__(SavedResultsReview)
        self.review.root = self.root
        self.review.input_cache = None
        self.review.source_fingerprint = "historical-source"
        self.review.runs = {}
        self.review.conditions = {}
        self.review.split_paths = {}
        self.review.report_path = self.root / "results/tables/original"
        self.review.report_path.mkdir(parents=True)
        (self.review.report_path / "provenance.json").write_text("{}")
        self.review._started = monotonic()
        directory = self.root / "results/runs/original"
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}")
        (directory / "metrics.json").write_text("{}")
        (directory / "history.csv").write_text("step,loss\n1,0.1\n")
        np.savez(directory / "weights.npz", model=np.array([0.25]))
        np.savez(directory / "predictions.npz", test_labels=np.array([0]))
        original = self.root / "src/original.py"
        original.parent.mkdir()
        original.write_text("# historical training source\n")
        with ZipFile(directory / "source.zip", "w") as archive:
            archive.write(original, "src/original.py")
        manifest = {"status": "complete", "identity": {
            "code_fingerprint": "historical-source", "environment": {"platform": "Linux"}},
            "artifact_sha256": {file.name: sha256(file.read_bytes()).hexdigest()
                                for file in directory.iterdir()}}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        run = self.review.load_saved_run(directory)
        self.review.gold = self.review.classical = run
        self.report = self.review.report_path.relative_to(self.root).as_posix()
        self.run = directory.relative_to(self.root).as_posix()

    def prepare(self, **kwargs):
        return prepare_review_inputs(self.root, self.report, gold_run=self.run,
                                     classical_run=self.run, cache_directory=self.cache, **kwargs)

    def test_archive_uses_local_bytes_preserves_public_paths_and_checks_every_hash(self):
        build_review_input_archive(self.review)
        messages = []
        cache = self.prepare(progress=messages.append)
        self.review.input_cache = cache
        self.review.runs.clear()
        with self.assertRaisesRegex(ValueError, "leaves the project root"):
            self.review._read_path(self.root / "../outside")
        with self.assertRaisesRegex(ValueError, "leaves the project root"):
            self.review._read_path(self.root.parent / "outside")
        with protect_review():
            run = self.review.load_saved_run(self.run)
            self.review._check_original_source()
        np.testing.assert_array_equal(run["weights"]["model"], [0.25])
        self.assertEqual(run["path"], self.root / self.run)
        self.assertEqual(run["manifest"]["identity"]["environment"]["platform"], "Linux")
        self.assertTrue(any("still save to storage" in message for message in messages))
        self.assertEqual(self.prepare(), cache)
        (cache / self.run / "weights.npz").write_bytes(b"corrupt local input")
        self.review.runs.clear()
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.review.load_saved_run(self.run)
        self.assertFalse(self.review.runs)  # Never fall back to fitting or the storage copy.

    def test_current_original_source_is_checked_even_with_local_cache(self):
        build_review_input_archive(self.review)
        self.review.input_cache = self.prepare()
        (self.root / "src/original.py").write_text("# changed scientific implementation\n")
        with self.assertRaisesRegex(ValueError, "Original experiment source changed"):
            self.review._check_original_source()

    def test_corrupt_or_wrong_selection_archive_stops(self):
        archive = build_review_input_archive(self.review)
        with archive.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.prepare()
        build_review_input_archive(self.review)
        with self.assertRaisesRegex(ValueError, "different report/baseline"):
            prepare_review_inputs(self.root, self.report, gold_run="different",
                                  classical_run=self.run, cache_directory=self.cache)

    def test_missing_archive_falls_back_but_incomplete_archive_stops(self):
        messages = []
        self.assertIsNone(self.prepare(progress=messages.append))
        self.assertIn("Archive not present", messages[-1])
        archive = build_review_input_archive(self.review)
        archive.unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()

    def test_concurrent_loading_deduplicates_paths_and_reports_progress(self):
        messages = []
        self.review.progress = messages.append
        barrier = Barrier(2)
        paths = [self.root / "run1", self.root / "run2"]
        def reader(path):
            barrier.wait(timeout=5)
            return {"path": path}
        self.review.runs.clear()
        with patch.object(self.review, "_read_run", side_effect=reader):
            self.review._preload_runs([*paths, paths[0]], workers=2)
        self.assertEqual(set(self.review.runs), set(paths))
        self.assertTrue(any("Verified 2/2" in message for message in messages))

    def test_cache_cannot_be_on_drive_and_archive_paths_cannot_escape_it(self):
        archive = build_review_input_archive(self.review)
        with self.assertRaisesRegex(ValueError, "outside the storage project"):
            prepare_review_inputs(self.root, self.report, gold_run=self.run,
                                  classical_run=self.run, cache_directory=self.root / "cache")
        with ZipFile(archive, "w") as malicious:
            malicious.writestr("../outside", "invalid")
        sidecar = archive.with_suffix(".json")
        metadata = json.loads(sidecar.read_text())
        metadata.update(archive_sha256=sha256(archive.read_bytes()).hexdigest(), file_count=1)
        sidecar.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "invalid file path"):
            self.prepare()
        self.assertFalse((self.cache / "outside").exists())


if __name__ == "__main__":
    unittest.main()
