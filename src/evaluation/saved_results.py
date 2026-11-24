"""Read completed experiments across machines without fitting or resuming models.

Historical training identities remain unchanged. Explicit report/run paths select
the experiment; current environment matching is deliberately not used for review.
"""

from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd

from src.artifacts import fingerprint, load_arrays, load_json, load_run
from src.data.dataset import DatasetSplits


def require_review_flags(**flags):
    enabled = [name for name, value in flags.items() if value]
    if enabled:
        raise RuntimeError("Saved-results review requires these flags False: " + ", ".join(enabled))


def _blocked(*args, **kwargs):
    raise RuntimeError("Saved-results review blocks fitting, simulation, cleanup, and checkpoint execution. "
                       "Load the completed result instead; no training has started.")


def protect_review():
    """Keep this returned guard alive until kernel restart (or explicitly close it)."""
    targets = [
        "src.checkpoints.CheckpointSession.open", "src.checkpoints.retire_pilot_checkpoints",
        "src.artifacts.create_run", "src.data.dataset.download_mnist",
        "src.data.preprocessing.ImagePreprocessor.fit",
        "src.training.fit_vqc", "src.unlearning.methods.unlearn",
        "src.models.classical.fit_classical_baselines",
        "src.experiments.fit_vqc", "src.experiments.unlearn",
        "src.experiments.fit_classical_baselines",
        "src.experiment_stages.fit_vqc", "src.experiment_stages.unlearn",
        "src.experiment_stages.fit_classical_baselines",
        "src.models.vqc.VQC.predict_proba",
    ]
    guard = ExitStack()
    try:
        for target in targets:
            guard.enter_context(patch(target, _blocked))
    except BaseException:
        guard.close()
        raise
    return guard


class SavedResultsReview:
    """Strict, read-only loading of one complete, explicitly selected report."""

    def __init__(self, root, report, *, gold_run, classical_run,
                 input_cache=None, io_workers=1, progress=None):
        self.root = Path(root).resolve()
        self.input_cache = Path(input_cache).resolve() if input_cache is not None else None
        self.progress = progress
        self._started = monotonic()
        self._notify("Reading saved report and condition manifests (no training).")
        self.report_path = self.local_path(report)
        self.provenance = self._read_json(self.report_path / "provenance.json")
        if not self.provenance.get("main_experiment_complete"):
            raise ValueError("Select a complete saved experiment for review.")
        self.source_fingerprint = self.provenance["source_fingerprint"]
        self.runs = {}
        self.conditions = {}
        self.split_paths = {}
        payloads = {}
        for number, relative in enumerate(self.provenance["condition_manifests"], start=1):
            payloads[relative] = self._read_json(self.local_path(relative))
            if number % 10 == 0:
                self._notify(f"Read {number}/{len(self.provenance['condition_manifests'])} condition manifests.")
        links = {link for payload in payloads.values() for group in ("models", "unlearned")
                 for link in payload[group].values()}
        self._preload_runs(links, io_workers)
        self._notify("Validating condition, split, parent and reference links.")
        for relative in self.provenance["condition_manifests"]:
            path = self.local_path(relative)
            payload = payloads[relative]
            if path.name != f"condition_{fingerprint(payload)[:16]}.json":
                raise ValueError(f"Condition manifest content changed: {relative}")
            condition = {**payload, "manifest_path": path}
            for group, required in (
                ("models", {"clean", "poisoned", "retrained_clean", "retrained_poisoned"}),
                ("unlearned", {f"{case}_{method}" for case in ("clean", "poisoned")
                               for method in payload["config"]["unlearning"]["methods"]}),
            ):
                if set(payload[group]) != required:
                    raise ValueError(f"Missing or unexpected saved roles: {relative}/{group}")
                condition[group] = {role: self.load_saved_run(link) for role, link in payload[group].items()}
            key = fingerprint(payload["config"])
            if key in self.conditions:
                raise ValueError("Duplicate saved condition configuration.")
            self.conditions[key] = condition
            fold = payload["config"]["data"]["fold"]
            split_path = self.local_path(payload["split_manifest"])
            if fold in self.split_paths and self.split_paths[fold] != split_path:
                raise ValueError("Conflicting split manifests within one fold.")
            self.split_paths[fold] = split_path
            if self._read_json(split_path)["fingerprint"] != payload["split_fingerprint"]:
                raise ValueError("Condition and saved split fingerprint disagree.")
            for run in condition["models"].values():
                if run["manifest"]["identity"]["split_fingerprint"] != payload["split_fingerprint"]:
                    raise ValueError("A linked model belongs to a different split.")
            for role, run in condition["unlearned"].items():
                case, method = role.split("_", 1)
                identity = run["manifest"]["identity"]
                if (identity["parent_run"] != condition["models"][case]["path"].name
                        or identity["reference_run"] != condition["models"][f"retrained_{case}"]["path"].name
                        or identity["method"] != method):
                    raise ValueError("Unlearning checkpoint has different parent/reference/method links.")
            saved_test = self._read_json(split_path)["test"]
            for run in [*condition["models"].values(), *condition["unlearned"].values()]:
                np.testing.assert_array_equal(run["predictions"]["test_sample_ids"], saved_test["sample_ids"])
                np.testing.assert_array_equal(run["predictions"]["test_labels"], saved_test["labels"])
        if {run["path"].name for run in self.runs.values()} != set(self.provenance["source_run_ids"]):
            raise ValueError("Saved report and linked model runs disagree.")
        if sorted(self.split_paths) != sorted(self.provenance["planned_folds"]):
            raise ValueError("Saved report does not cover every planned fold.")
        self.stage_run_count = len(self.runs)
        self._notify("Verifying saved gold/classical baselines.")
        self.gold = self.load_saved_run(gold_run)
        self.classical = self.load_saved_run(classical_run)
        self._validate_baselines()
        self._notify("Comparing current original source files with the archived training source.")
        self._check_original_source()
        self._notify(f"Saved-input verification complete: {self.stage_run_count} model stages, "
                     f"{len(self.conditions)} conditions.")

    def _notify(self, message):
        progress = getattr(self, "progress", None)
        if progress is not None:
            progress(f"[{monotonic() - self._started:.0f}s] {message}")

    def _read_path(self, path):
        """Use local input bytes while preserving public project/output paths."""
        cache = getattr(self, "input_cache", None)
        if cache is None:
            return self.local_path(path)
        # Map logical project paths without stat/readlink calls on the storage file.
        # The cached bytes are local; resolve and contain that local path instead.
        logical = self.root / path
        try:
            relative = logical.relative_to(self.root)
        except ValueError:
            raise ValueError("Saved artifact path leaves the project root.") from None
        if ".." in relative.parts:
            raise ValueError("Saved artifact path leaves the project root.")
        local = (cache / relative).resolve()
        if not local.is_relative_to(cache):
            raise ValueError("Saved input path leaves the local cache.")
        return local

    def _read_json(self, path):
        return load_json(self._read_path(path))

    def _read_run(self, path):
        run = load_run(self._read_path(path))  # Every original artifact SHA is checked.
        if run["manifest"]["identity"]["code_fingerprint"] != self.source_fingerprint:
            raise ValueError("A linked run was produced by a different source version.")
        run["path"] = path  # Scientific identities and new checkpoints retain storage paths.
        return run

    def _preload_runs(self, links, workers):
        if workers < 1:
            raise ValueError("Review I/O workers must be at least one.")
        paths = sorted({self.local_path(link) for link in links})
        total = len(paths)
        self._notify(f"Checking {total} unique model stages with {workers} file-reader(s).")
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="saved-input")
        pending = {executor.submit(self._read_run, path): path for path in paths}
        last_update = monotonic()
        complete = False
        try:
            while pending:
                done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    path = pending.pop(future)
                    self.runs[path] = future.result()
                now = monotonic()
                if not done or now - last_update >= 5 or not pending:
                    current = next(iter(pending.values()), None)
                    reading = f" Reading {current.name}." if current is not None else ""
                    self._notify(f"Verified {total - len(pending)}/{total} model stages.{reading}")
                    last_update = now
            complete = True
        finally:
            executor.shutdown(wait=complete, cancel_futures=True)

    def local_path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Saved artifact path leaves the project root.")
        return path

    def load_saved_run(self, relative):
        path = self.local_path(relative)
        if path not in self.runs:
            self.runs[path] = self._read_run(path)
        return self.runs[path]

    def _validate_baselines(self):
        identities = [run["manifest"]["identity"] for run in (self.gold, self.classical)]
        if [identity["stage"] for identity in identities] != ["clean_vqc", "classical_cv"]:
            raise ValueError("The selected baseline run roles are incorrect.")
        for identity in identities:
            if identity["config"]["data"] != self.provenance["data_config"]:
                raise ValueError("Baseline data configuration differs from the report.")
            if identity["config"]["seed"] != self.provenance["seed"]:
                raise ValueError("Baseline seed differs from the report.")
        folds = sorted(self.split_paths)
        fingerprints = [self._read_json(self.split_paths[fold])["fingerprint"] for fold in folds]
        selected = self.provenance["selected_fold"]
        if identities[0]["fold_fingerprints"] != [fingerprints[folds.index(selected)]]:
            raise ValueError("Gold model split does not match the selected fold.")
        if identities[1]["fold_fingerprints"] != fingerprints:
            raise ValueError("Classical baseline splits do not match the report.")
        for condition in self.conditions.values():
            if condition["config"]["data"]["fold"] == selected:
                imported = condition["models"]["clean"]["manifest"]["identity"]["imported_gold"]
                if imported != self.gold["path"].name:
                    raise ValueError("Selected gold model is not the one linked by these conditions.")

    def _check_original_source(self):
        # New review helpers may be added, but original analysis/training code must
        # still match its archived bytes. Never relabel historical source identities.
        with ZipFile(self._read_path(self.gold["path"] / "source.zip")) as archive:
            for name in archive.namelist():
                if self.local_path(name).read_bytes() != archive.read(name):
                    raise ValueError(f"Original experiment source changed: {name}")

    def restore_splits(self, training_pool):
        lookup = {sample: i for i, sample in enumerate(training_pool.sample_ids)}
        restored = []
        for fold in sorted(self.split_paths):
            saved = self._read_json(self.split_paths[fold])
            parts = {}
            for name in ("train", "validation", "test"):
                part = training_pool.subset([lookup[sample] for sample in saved[name]["sample_ids"]])
                np.testing.assert_array_equal(part.labels, saved[name]["labels"])
                parts[name] = part
            splits = DatasetSplits(**parts)
            if splits.fingerprint() != saved["fingerprint"]:
                raise ValueError("Saved split no longer matches the original image data.")
            restored.append(splits)
        return restored

    def gold_features(self, splits):
        saved = load_arrays(self._read_path(self.gold["path"] / "clean_features.npz"))
        result = {}
        for name in ("train", "validation", "test"):
            part = getattr(splits, name)
            np.testing.assert_array_equal(saved[f"{name}_sample_ids"], part.sample_ids)
            np.testing.assert_array_equal(saved[f"{name}_labels"], part.labels)
            result[name] = saved[f"{name}_features"]
        diagnostics = self._read_json(self.gold["path"] / "clean_preprocessing.json")
        # Only the numeric diagnostics needed by the existing PCA display. This
        # is deliberately not an executable cross-version sklearn transformer.
        result["preprocessing"] = SimpleNamespace(
            fit_sample_ids_=saved["train_sample_ids"],
            pca=SimpleNamespace(explained_variance_ratio_=np.asarray(diagnostics["explained_variance_ratio"])),
            diagnostics=lambda: dict(diagnostics),
        )
        return result

    def condition(self, config, splits, *, include_unlearned=True):
        key = fingerprint(config.to_dict())
        if key not in self.conditions:
            raise ValueError("No saved condition matches these settings. Review never trains missing models.")
        saved = self.conditions[key]
        if saved["split_fingerprint"] != splits.fingerprint():
            raise ValueError("Requested condition uses different data/splits.")
        return {**saved, "models": dict(saved["models"]),
                "unlearned": dict(saved["unlearned"]) if include_unlearned else {}}

    def annotated_path(self, poison_config, fold):
        matches = [condition for condition in self.conditions.values()
                   if condition["config"]["data"]["fold"] == fold
                   and fingerprint(condition["config"]["poison"]) == fingerprint(poison_config)]
        if len(matches) != 1:
            raise ValueError("No unique saved annotation dataset matches the selected condition.")
        path = self.local_path(matches[0]["annotated_dataset"])
        if load_json(path / "metadata.json") != matches[0]["poisoning"]:
            raise ValueError("Saved annotation metadata differs from the condition manifest.")
        return path

    def table(self, name):
        return pd.read_csv(self._read_path(self.report_path / f"{name}.csv"))
