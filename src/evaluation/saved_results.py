"""Read completed experiments across machines without fitting or resuming models.

Historical training identities remain unchanged. Explicit report/run paths select
the experiment; current environment matching is deliberately not used for review.
"""

from contextlib import ExitStack
from pathlib import Path
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

    def __init__(self, root, report, *, gold_run, classical_run):
        self.root = Path(root).resolve()
        self.report_path = self.local_path(report)
        self.provenance = load_json(self.report_path / "provenance.json")
        if not self.provenance.get("main_experiment_complete"):
            raise ValueError("Select a complete saved experiment for review.")
        self.source_fingerprint = self.provenance["source_fingerprint"]
        self.runs = {}
        self.conditions = {}
        self.split_paths = {}
        for relative in self.provenance["condition_manifests"]:
            path = self.local_path(relative)
            payload = load_json(path)
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
            if load_json(split_path)["fingerprint"] != payload["split_fingerprint"]:
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
            saved_test = load_json(split_path)["test"]
            for run in [*condition["models"].values(), *condition["unlearned"].values()]:
                np.testing.assert_array_equal(run["predictions"]["test_sample_ids"], saved_test["sample_ids"])
                np.testing.assert_array_equal(run["predictions"]["test_labels"], saved_test["labels"])
        if {run["path"].name for run in self.runs.values()} != set(self.provenance["source_run_ids"]):
            raise ValueError("Saved report and linked model runs disagree.")
        if sorted(self.split_paths) != sorted(self.provenance["planned_folds"]):
            raise ValueError("Saved report does not cover every planned fold.")
        self.stage_run_count = len(self.runs)
        self.gold = self.load_saved_run(gold_run)
        self.classical = self.load_saved_run(classical_run)
        self._validate_baselines()
        self._check_original_source()

    def local_path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Saved artifact path leaves the project root.")
        return path

    def load_saved_run(self, relative):
        path = self.local_path(relative)
        if path not in self.runs:
            run = load_run(path)  # Checks completion and every artifact checksum.
            if run["manifest"]["identity"]["code_fingerprint"] != self.source_fingerprint:
                raise ValueError("A linked run was produced by a different source version.")
            self.runs[path] = run
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
        fingerprints = [load_json(self.split_paths[fold])["fingerprint"] for fold in folds]
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
        with ZipFile(self.gold["path"] / "source.zip") as archive:
            for name in archive.namelist():
                if self.local_path(name).read_bytes() != archive.read(name):
                    raise ValueError(f"Original experiment source changed: {name}")

    def restore_splits(self, training_pool):
        lookup = {sample: i for i, sample in enumerate(training_pool.sample_ids)}
        restored = []
        for fold in sorted(self.split_paths):
            saved = load_json(self.split_paths[fold])
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
        saved = load_arrays(self.gold["path"] / "clean_features.npz")
        result = {}
        for name in ("train", "validation", "test"):
            part = getattr(splits, name)
            np.testing.assert_array_equal(saved[f"{name}_sample_ids"], part.sample_ids)
            np.testing.assert_array_equal(saved[f"{name}_labels"], part.labels)
            result[name] = saved[f"{name}_features"]
        diagnostics = load_json(self.gold["path"] / "clean_preprocessing.json")
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
        return pd.read_csv(self.local_path(self.report_path / f"{name}.csv"))
