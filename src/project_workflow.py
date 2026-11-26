"""Local notebook helpers for the current protocol and portable saved results.

Saved review never fits a transformer, trains a model, or generates predictions.
Training entry points are explicit; archived source/environment identities are
kept intact when their completed results are read on a different machine.
"""

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import os

import numpy as np
import pandas as pd

from src import artifacts
from src.config import DEFAULT_SEED
from src.data.dataset import DatasetSplits, load_mnist, balanced_subset, make_cv_splits
from src.data.feature_randomization import experiment_config, SignedFeatureDataset
from src.data.poisoning import PoisonedDataset
from src.evaluation.saved_results import protect_review, require_review_flags, _blocked
from src.execution_resources import memory_budget_mb
from src.fresh_study import FreshStudy, base_config
from src.models.parallel_aer import AerExecution, gpu_sample
from tools.noisy_training_runtime import completed_fresh_index, completed_extension_index


STUDY_LABEL = "feature_randomization_v1"
STUDY_FOLDS = (0, 1, 2, 3, 4)


def protect_saved_review(**execution_flags):
    """Keep this guard alive until kernel restart; all execution flags must be False."""
    require_review_flags(**execution_flags)
    guard = ExitStack()
    try:
        guard.enter_context(protect_review())
        for target in (
            "src.data.feature_randomization.UnitInputPreprocessor.fit",
            "src.fresh_study.run_fresh_study", "src.ablations.run_study",
            "src.noise_study.run_noise_study", "src.noisy_training.run_noisy_training",
            "src.models.backends.CircuitExecutor.run",
            "src.models.parallel_aer.AerBatchService.__init__",
            "src.models.noisy_training.NoisyBatchService.__init__",
            "sklearn.decomposition.PCA.fit", "sklearn.decomposition.PCA.fit_transform",
            "sklearn.preprocessing.MinMaxScaler.fit",
        ):
            guard.enter_context(patch(target, _blocked))
    except BaseException:
        guard.close()
        raise
    return guard


def load_completed_study(root, selected=None, *, label=STUDY_LABEL, progress=print):
    """Select one complete study. A missing cache always stops without fitting."""
    root = Path(root).resolve()
    try:
        path = completed_fresh_index(root, selected, label=label)
    except ValueError as error:
        raise ValueError(
            "No unique completed study is selected. Restore the saved results with "
            "their relative results/ and data/ paths, or explicitly select STUDY_INDEX. "
            "For a new experiment, restart the kernel and enable training mode. "
            "No model has been trained. Details: " + str(error)
        ) from error
    return FreshStudy(root, path, progress=progress)


def local_splits(root, config=None, *, folds=STUDY_FOLDS, download=False):
    """New-experiment data selection; saved review restores exact recorded IDs instead."""
    config = config or base_config()
    train_pool, _ = load_mnist(Path(root) / "data/raw", download=download)
    pool = balanced_subset(train_pool, config.data.train_per_class, config.seed)
    splits = make_cv_splits(pool, config.data.n_folds, config.data.validation_fraction, config.seed)
    return {fold: splits[fold] for fold in folds}


def restore_study_splits(study):
    """Reconstruct raw images by recorded IDs and verify labels and fingerprints."""
    raw, _ = load_mnist(study.root / "data/raw", download=False)
    lookup = {str(value): i for i, value in enumerate(raw.sample_ids)}
    restored = {}
    for fold, path in study.split_paths.items():
        record = artifacts.load_json(path)
        parts = {}
        for name in ("train", "validation", "test"):
            saved = record[name]
            try:
                indices = np.array([lookup[str(value)] for value in saved["sample_ids"]])
            except KeyError as error:
                raise ValueError("Saved MNIST record is absent from the raw training archive.") from error
            parts[name] = raw.subset(indices)
            np.testing.assert_array_equal(parts[name].labels, saved["labels"])
        split = DatasetSplits(**parts)
        if split.fingerprint() != record["fingerprint"]:
            raise ValueError("Raw images or saved split membership differ from the experiment.")
        restored[fold] = split
    return restored


def study_config(study, fold=0):
    """Use recorded scientific settings when displaying a saved study."""
    return experiment_config(study.conditions[f"fold_{fold}_clean"]["config"])


def restore_poison_variant(study, condition_name, *, split=None):
    """Load the exact saved training inputs without regenerating random vectors.

    Floating-point normalization may differ by a few ULPs between platforms.
    The saved arrays, fingerprints, labels, and annotation rows are authoritative.
    This read-only path does not relax any training/optimizer identity checks.
    """
    condition = study.conditions[condition_name]
    config = experiment_config(condition["config"])
    split = restore_study_splits(study)[config.data.fold] if split is None else split
    if split.fingerprint() != condition["split_fingerprint"]:
        raise ValueError("Saved poison preview belongs to a different split.")
    directory = study._read_path(study.root / condition["annotated_dataset"])
    metadata = artifacts.load_json(directory / "metadata.json")
    if metadata != condition["poisoning"]:
        raise ValueError("Saved poisoning metadata differs from the condition.")
    arrays = artifacts.load_arrays(directory / "images.npz")
    original = split.train
    np.testing.assert_array_equal(arrays["sample_ids"], original.sample_ids)
    np.testing.assert_array_equal(arrays["original_labels"], original.labels)
    mask = arrays["is_poisoned"]
    if mask.dtype != np.bool_ or mask.shape != original.labels.shape:
        raise ValueError("Saved poison membership must be a matching boolean vector.")
    dataset_class = SignedFeatureDataset if config.poison.kind == "feature_randomization" else type(original)
    observed = dataset_class(arrays["images"], arrays["observed_labels"], arrays["sample_ids"])
    eligible_class = config.poison.source_class if config.poison.kind == "label_flip" else config.poison.target_class
    eligible = original.labels == eligible_class
    if np.any(mask & ~eligible):
        raise ValueError("Saved poison membership includes an ineligible digit.")
    variant = PoisonedDataset(original, observed, mask, config.poison, int(eligible.sum()))
    if variant.metadata() != metadata:
        raise ValueError("Saved poisoned-data bytes do not match their recorded fingerprints.")
    pd.testing.assert_frame_equal(pd.read_csv(directory / "annotations.csv"), variant.manifest(), check_dtype=False)
    parent = condition["models"]["poisoned"]
    expected = original.fingerprint() if config.poison.kind == "none" else observed.fingerprint()
    if parent["manifest"]["identity"]["train_fingerprint"] != expected:
        raise ValueError("Saved original model was trained on different poisoned inputs.")
    return variant


def clean_runs(study):
    return {fold: study.conditions[f"fold_{fold}_clean"]["models"]["clean"]
            for fold in sorted(study.split_paths)}


def classical_table(study):
    return baseline_table(clean_runs(study))


def baseline_table(runs):
    rows = []
    for fold, run in runs.items():
        for model, values in run["metrics"]["models"].items():
            rows.append({"fold": fold, "model": "gold_vqc" if model == "model" else model,
                         "run_id": run["path"].name,
                         **{key: value for key, value in values.items() if np.isscalar(value)}})
    return pd.DataFrame(rows)


def train_gold_models(root, splits, execution, *, config=None, label=STUDY_LABEL, progress=None):
    """Fit the clean originals first; the full pipeline reuses these exact cache keys.

    The notebook skips this preview for forced attempts; the full runner then
    creates one new attempt rather than fitting the gold models twice.
    """
    from src import experiment_stages as stages
    from src.ablation_conditions import _parallel
    from src.checkpoints import CheckpointSession
    from src.fresh_study import fresh_execution
    from src.models.parallel_aer import AerBatchService
    config = config or base_config()
    session = CheckpointSession(force_retrain=False)
    jobs = []
    with AerBatchService(config.circuit, config.backend, execution) as service:
        with fresh_execution(root, service, label):
            for fold, split in sorted(splits.items()):
                settings = replace(config, data=replace(config.data, fold=fold))
                def train(cancel, fold=fold, split=split, settings=settings):
                    def report(row):
                        if cancel.is_set():
                            raise InterruptedError("Stopped after a saved gold-model update.")
                        if progress and (row.get("step") in (0, 1, settings.training.steps)
                                         or row.get("step", 1) % 10 == 0):
                            progress({**row, "fold": fold, "condition": "clean", "role": "original",
                                      "updates": settings.training.steps})
                    return stages._supervised(settings, split, split.train, False,
                        allow_training=True, root=root, role="poisoned", checkpoints=session, callback=report)
                jobs.append((fold, train))
            return _parallel(jobs, execution.workers, None)


def cached_fold_features(study, fold=0):
    run = clean_runs(study)[fold]
    features = artifacts.load_arrays(run["path"] / "features.npz")
    diagnostics = artifacts.load_json(run["path"] / "preprocessing.json")
    # The variance plot needs only these saved numbers, never an unpickled estimator.
    variance_view = SimpleNamespace(diagnostics=lambda: diagnostics)
    return features, diagnostics, variance_view


def optional_study_index(study, stage, selected=None):
    """Discover compatible saved extensions, including a partially completed noise fit."""
    if stage in ("ablations", "noise_studies"):
        return completed_extension_index(study, stage, selected)
    if stage != "noisy_training":
        raise ValueError("Unknown study stage.")
    paths = ([study.root / selected] if selected is not None else
             sorted((study.root / "results/noisy_training").glob("*/study.json")))
    matches = []
    for path in paths:
        path = study._read_path(path)
        record = artifacts.load_json(path)
        protocol = record.get("protocol", {})
        if (protocol.get("original_report") == study.report_path.relative_to(study.root).as_posix()
                and protocol.get("original_source") == study.source_fingerprint):
            matches.append(path)
    if len(matches) > 1 or (selected is not None and len(matches) != 1):
        raise ValueError("Select one noisy-training index belonging to this study.")
    return matches[0] if matches else None


def extension_inventory(study, selections=None):
    """Read three small indexes rather than recursively scanning every checkpoint."""
    rows, paths = [], {}
    for stage in ("ablations", "noise_studies", "noisy_training"):
        path = optional_study_index(study, stage, (selections or {}).get(stage))
        paths[stage] = path
        row = {"stage": stage, "index": None, "complete": False, "status": "not available"}
        if path is not None:
            record = artifacts.load_json(path)
            row.update(index=path.relative_to(study.root).as_posix(),
                       complete=bool(record.get("complete")),
                       status="complete" if record.get("complete") else "partial; no fitting in review")
            if stage == "noisy_training":
                row["saved_fits"] = len(record.get("supervised", {})) + len(record.get("unlearned", {}))
                row["planned_fits"] = len(record["protocol"]["plan"])
        rows.append(row)
    return paths, pd.DataFrame(rows)


def local_execution(*, device="auto", workers=None):
    """Conservative parallel local execution, with explicit CPU fallback.

    VRAM is a memory limit, not a utilization target. Benchmark new settings
    before changing execution for a long fit; incomplete identities stay strict.
    """
    from qiskit_aer import AerSimulator
    available = tuple(AerSimulator().available_devices())
    if device not in ("auto", "CPU", "GPU"):
        raise ValueError("device must be auto, CPU, or GPU.")
    selected = "GPU" if device == "auto" and "GPU" in available else "CPU" if device == "auto" else device
    if selected not in available:
        raise ValueError(f"Aer device {selected} is unavailable: {available}")
    cpu_count = os.cpu_count() or 1
    workers = min(cpu_count, 8 if selected == "GPU" else 2) if workers is None else workers
    sample = gpu_sample() if selected == "GPU" else {}
    memory = memory_budget_mb(sample, fraction=.75) if sample.get("memory.total") else 4096
    return AerExecution(device=selected, workers=workers, threads=1,
                        circuit_batch_size=2048, memory_mb=memory,
                        runtime_binding=False, gpu_parameter_batching=False)


def review_directory(study):
    identity = artifacts.fingerprint({"study": study.report_path.relative_to(study.root).as_posix(),
                                      "training_source": study.source_fingerprint,
                                      "review_source": artifacts.source_fingerprint()})[:16]
    return study.root / "results/reviews" / identity


def saved_input_diagnostics(study):
    path = study.report_path.parent / "analysis/input_diagnostics.csv"
    if not path.is_file() or path.stat().st_size <= 1:
        return pd.DataFrame()
    return pd.read_csv(path)


def provenance_table(study, environment):
    protocol = study.index["protocol"]
    return pd.DataFrame([
        {"role": "original training", "source": study.source_fingerprint,
         "environment": protocol["environment"], "execution": protocol["execution"]},
        {"role": "current review", "source": artifacts.source_fingerprint(),
         "environment": environment, "execution": "saved arrays only; no predictions"},
    ])


def noisy_fit_required(index, *, run=False, force=False):
    """Review a completed imported study; fit only an enabled missing/new attempt."""
    if not run:
        return False
    return bool(force or index is None or not artifacts.load_json(index).get("complete"))


def prepare_aer_noisy_execution(study, problems, historical, bindings, setting, execution,
                                *, selected_index=None, force=False):
    """Explicit Aer execution for future fits, with strict same-backend continuation.

    Completed imported results remain read-only. A forced Aer experiment uses
    a new study/run identity; a foreign optimizer is never converted to Aer.
    """
    from src.noisy_training import benchmark_noisy_training
    from tools.noisy_training_runtime import prepare_noisy_execution
    if force:
        return benchmark_noisy_training(problems[0], setting, study.root,
            candidates=[execution], minimum_workers=execution.workers,
            require_gpu_device=execution.device == "GPU", probe_steps=1)
    if selected_index is not None:
        record = artifacts.load_json(selected_index)
        engine = record.get("protocol", {}).get("execution", {}).get("engine", "Aer")
        if engine != "Aer":
            raise ValueError("The selected optimizer was produced by another engine. "
                             "Review its completed results, or enable FORCE_NOISY_TRAINING "
                             "to create a separate Aer attempt. No state was converted.")
    return prepare_noisy_execution(study, problems, historical, bindings, setting,
        selected_index=selected_index, require_gpu_device=execution.device == "GPU")


def noisy_provenance(index):
    """Describe actual archived training separately from the future execution code."""
    record = artifacts.load_json(index)
    protocol = record["protocol"]
    runtime = record.get("service_resources", {}).get("torch_runtime", {})
    return pd.DataFrame([{
        "study": Path(index).parent.name, "complete": record["complete"],
        "supervised_fits": len(record["supervised"]), "unlearning_fits": len(record["unlearned"]),
        "training_engine": protocol["execution"].get("engine", "Qiskit Aer"),
        "training_device": runtime.get("gpu", protocol["execution"]["device"]),
        "training_source": protocol["code_fingerprint"],
        "future_training_backend": "Qiskit Aer",
    }])
