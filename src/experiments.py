"""Notebook-callable Phase 1 experiments. Nothing executes on import."""

from dataclasses import asdict, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.artifacts import (create_run, environment_info, finish_run, load_run, save_arrays,
                           save_json, save_poisoned_dataset, save_split_manifest,
                           save_training_result, source_fingerprint, save_source_snapshot, fingerprint,
                           save_fold_artifacts)
from src.config import DEFAULT_SEED, ExperimentConfig, PROJECT_ROOT
from src.checkpoints import CheckpointSession
from src.data.dataset import DatasetSplits, prepare_splits
from src.data.poisoning import apply_poisoning, forget_mask
from src.data.preprocessing import ImagePreprocessor, prepare_fold_features
from src.evaluation.metrics import (classification_metrics, compare_to_retraining,
                                   forgetting_changes)
from src.models.classical import aligned_probabilities, fit_classical_baselines
from src.models.vqc import VQC
from src.training import fit_vqc
from src.unlearning.methods import unlearn
from src.unlearning.targets import prepare_references


def _stage_identity(stage, folds, data_config, seed, **settings):
    # Include only settings used in this stage. The clean condition also lets
    # the existing artifact writer give the run its usual readable name.
    return {
        "stage": stage,
        "config": {"seed": seed, "data": asdict(data_config),
                   "poison": {"kind": "none", "rate": 0.0}, **settings},
        "fold_fingerprints": [fold.fingerprint() for fold in folds],
        "code_fingerprint": source_fingerprint(),
        "environment": environment_info(),
    }


def run_classical_cv(folds, *, data_config, seed=DEFAULT_SEED, reuse=True, root=PROJECT_ROOT,
                     checkpoints=None):
    """Evaluate four fixed classical models on every fixed outer fold.

    Each fold gets a separately fitted PCA/scaler. All models use only that
    fold's training records; the held-out test fold never selects settings.
    SVM calibration uses inner splits of the training partition. Its PCA is
    shared within that partition; the outer validation/test records stay out.
    """
    if data_config.split_strategy != "stratified_kfold" or len(folds) != data_config.n_folds:
        raise ValueError("Classical CV requires all configured stratified folds.")
    root = Path(root).resolve()
    directory = root / "results/runs/classical_cv"
    identity = _stage_identity("classical_cv", folds, data_config, seed)
    checkpoints = checkpoints if checkpoints is not None else CheckpointSession()
    path, cached = checkpoints.open(directory, identity, allow_execution=True, reuse=reuse,
                                     slot={"role": "classical_cv"})
    if cached is not None:
        return cached
    try:
        save_source_snapshot(path / "source.zip")
        scores, fold_metrics, predictions, split_files = [], {}, {}, {}
        for fold_index, splits in enumerate(folds):
            prefix = f"fold_{fold_index}"
            features = prepare_fold_features(splits, seed=data_config.seed)
            split_files[prefix] = save_fold_artifacts(path, splits, features, root, prefix)
            models = fit_classical_baselines(features["train"], splits.train.labels, seed=seed)
            joblib.dump(models, path / f"{prefix}_models.joblib")
            predictions[f"{prefix}__test_sample_ids"] = splits.test.sample_ids
            predictions[f"{prefix}__test_labels"] = splits.test.labels
            fold_metrics[str(fold_index)] = {}
            for name, model in models.items():
                probabilities = aligned_probabilities(model, features["test"])
                metrics = classification_metrics(splits.test.labels, probabilities)
                fold_metrics[str(fold_index)][name] = metrics
                predictions[f"{prefix}__{name}__test"] = probabilities
                scores.append({"fold": fold_index, "seed": seed, "model": name,
                               "n_test": len(splits.test), "accuracy": metrics["accuracy"],
                               "retained_accuracy": metrics["retained_accuracy"],
                               "digit_2_recall": metrics["forget_recall"]})
        pd.DataFrame(scores).to_csv(path / "scores.csv", index=False)
        save_json(path / "datasets.json", {"split_manifests": split_files,
                                            "official_test_pool_used": False})
        save_json(path / "metrics.json", {"folds": fold_metrics})
        save_arrays(path / "predictions.npz", **predictions)
        # These empty, explicitly named artifacts keep load_run() compatible;
        # classical estimators have no quantum weights or iterative VQC history.
        save_arrays(path / "weights.npz")
        pd.DataFrame(columns=["model", "step"]).to_csv(path / "history.csv", index=False)
        finish_run(path, "complete", scope="clean classical baselines on fixed folds")
    except Exception as error:
        finish_run(path, "failed", error=f"{type(error).__name__}: {error}")
        raise
    return load_run(path)


def run_clean_vqc(splits, *, data_config, circuit_config, backend_config,
                     training_config, seed=DEFAULT_SEED, allow_training=False, reuse=True,
                     callback=None, root=PROJECT_ROOT, checkpoints=None):
    """Load the clean VQC (gold reference) run, or train it only with allow_training=True.

    Return None when no saved result exists and training is disabled. This
    makes opening/running the notebook safe from accidental long training.
    Checkpoints use validation loss; the test fold is evaluated only afterward.
    """
    if data_config.split_strategy != "stratified_kfold":
        raise ValueError("The clean VQC baseline uses a selected fixed stratified fold.")
    if backend_config.use_ibm_hardware or any(backend_config.noise_rates):
        raise ValueError("The clean VQC baseline uses local Aer without device noise (exact or sampled).")
    root = Path(root).resolve()
    directory = root / "results/runs/clean_vqc"
    identity = _stage_identity(
        "clean_vqc", [splits], data_config, seed,
        circuit=asdict(circuit_config), backend=asdict(backend_config),
        training=asdict(training_config),
    )
    checkpoints = checkpoints if checkpoints is not None else CheckpointSession()
    path, cached = checkpoints.open(directory, identity, allow_execution=allow_training, reuse=reuse,
                                     slot={"role": "clean", "fold": data_config.fold})
    if cached is not None:
        return cached
    if path is None:
        return None
    try:
        save_source_snapshot(path / "source.zip")
        features = prepare_fold_features(splits, seed=data_config.seed)
        split_file = save_fold_artifacts(path, splits, features, root, "clean")
        model = VQC(circuit_config, backend_config, seed=seed)
        training = fit_vqc(
            model, features["train"], splits.train.labels,
            features["validation"], splits.validation.labels,
            config=training_config, seed=seed, callback=callback,
            checkpoint_path=path / "checkpoints/clean.resume.json",
        )
        save_training_result(path, "clean", training)
        # The initialization is regenerated by fit_vqc from the saved seed.
        probabilities = model.predict_proba(features["test"], training.weights)
        metrics = classification_metrics(splits.test.labels, probabilities)
        save_arrays(path / "weights.npz", clean=training.weights)
        save_arrays(path / "predictions.npz", test_sample_ids=splits.test.sample_ids,
                    test_labels=splits.test.labels, clean__test=probabilities)
        save_json(path / "metrics.json", {"models": {"clean": metrics}})
        save_json(path / "resources.json", model.metadata())
        save_json(path / "datasets.json", {"split_manifest": split_file,
                                            "official_test_pool_used": False})
        training.history.assign(model="clean").to_csv(path / "history.csv", index=False)
        finish_run(path, "complete", scope="one clean VQC on one outer fold",
                   best_step=training.best_step, best_validation_loss=training.best_score)
    except BaseException as error:
        finish_run(path, "failed", error=f"{type(error).__name__}: {error}")
        raise
    return load_run(path)


def run_experiment(config: ExperimentConfig | None = None, *, root: Path | str = PROJECT_ROOT,
                   splits: DatasetSplits | None = None, download: bool = False,
                   reuse: bool = True, callback=None) -> dict:
    """Train/evaluate one condition, seed, and fold; save and return all outputs.

    Downloading and training require an explicit call. Inject splits for tests or
    a preselected protocol. Their content fingerprint is part of the run identity.
    Results include clean/poisoned models, exact-record retained-data retraining references, and each
    configured unlearning method. This can be expensive with 72 shifted parameters.
    """
    config = config or ExperimentConfig()
    root = Path(root)
    dataset_source = "caller_supplied" if splits is not None else "MNIST"
    splits = splits or prepare_splits(config.data, root, download=download)
    poisoned = apply_poisoning(splits.train, config.poison)
    identity = {"config": config.to_dict(), "data_fingerprint": splits.fingerprint(),
                "code_fingerprint": source_fingerprint(), "environment": environment_info()}
    path, cached = create_run(root / "results/runs", identity, reuse=reuse)
    if cached:
        return load_run(path)

    def report(stage, **information):
        if callback is not None:
            callback({"stage": stage, **information})

    try:
        save_source_snapshot(path / "source.zip")
        split_path = save_split_manifest(splits, root / "data/splits")
        annotated_path = save_poisoned_dataset(poisoned, root / "data/annotated")
        save_json(path / "datasets.json", {"split_manifest": str(split_path),
                                           "annotated_dataset": str(annotated_path),
                                           "poisoning": poisoned.metadata()})
        train, validation, test = splits.train, splits.validation, splits.test
        forget_class = config.unlearning.forget_class
        clean_forget = train.labels == forget_class
        poisoned_forget = forget_mask(poisoned, forget_class, config.unlearning.forget_definition)
        masks = {"clean": clean_forget, "poisoned": poisoned_forget}
        for name, mask in masks.items():
            if not mask.any() or mask.all():
                raise ValueError(f"{name} requires nonempty forget and retain sets.")
        if not (validation.labels != forget_class).any():
            raise ValueError("Retained-data retraining needs retained validation examples.")
        shared = None
        if config.data.preprocessing == "shared_clean_train":
            shared = ImagePreprocessor(config.data.seed).fit(train.images, train.sample_ids)

        bundles, transformers, weights, histories, resources = {}, {}, {}, [], {}
        feature_arrays = {"test_sample_ids": test.sample_ids, "test_labels": test.labels}
        predictions = {"test_sample_ids": test.sample_ids, "test_labels": test.labels,
                       "train_sample_ids": train.sample_ids, "train_original_labels": train.labels,
                       "train_observed_labels": poisoned.observed.labels,
                       "clean_forget_mask": clean_forget, "poisoned_forget_mask": poisoned_forget}
        metrics = {"models": {}, "unlearning_changes": {}, "retraining_comparisons": {},
                   "poisoning": poisoned.metadata(), "targets": {}}

        def record_model(name, model, transformer, outcome, mask=None):
            save_training_result(path, name, outcome)
            weights[name] = outcome.weights
            histories.append(outcome.history.assign(model=name))
            test_probabilities = model.predict_proba(transformer.transform(test.images), outcome.weights)
            predictions[f"{name}__test"] = test_probabilities
            metrics["models"][name] = classification_metrics(test.labels, test_probabilities, forget_class)
            if mask is not None:
                # Evaluate the designated training forget IDs using their CLEAN input images.
                forget_probabilities = model.predict_proba(transformer.transform(train.images[mask]), outcome.weights)
                predictions[f"{name}__forget_train_clean_inputs"] = forget_probabilities
                predictions[f"{name}__forget_train_ids"] = train.sample_ids[mask]
                metrics["models"][name]["designated_forget_train_clean_input_confidence"] = float(
                    forget_probabilities[:, forget_class].mean())
            resources[name] = model.metadata()
            report("evaluated", model=name, metrics=metrics["models"][name])

        def train_case(name, images, labels, ids, *, retained_only=False, mask=None):
            report("training", model=name)
            transformer = shared or ImagePreprocessor(config.data.seed).fit(images, ids)
            features = transformer.transform(images)
            validation_mask = validation.labels != forget_class if retained_only else np.ones(len(validation), dtype=bool)
            feature_arrays.update({f"{name}__train": features, f"{name}__train_labels": labels,
                                   f"{name}__train_ids": ids,
                                   f"{name}__validation": transformer.transform(validation.images[validation_mask]),
                                   f"{name}__validation_labels": validation.labels[validation_mask],
                                   f"{name}__validation_ids": validation.sample_ids[validation_mask],
                                   f"{name}__test": transformer.transform(test.images)})
            model = VQC(config.circuit, config.backend, config.seed)
            outcome = fit_vqc(
                model, features, labels,
                transformer.transform(validation.images[validation_mask]), validation.labels[validation_mask],
                config.training, config.seed,
                callback=lambda row: report("training_progress", model=name, **row),
            )
            transformers[name] = transformer
            bundles[name] = {"model": model, "transformer": transformer, "features": features,
                             "labels": labels, "outcome": outcome}
            record_model(name, model, transformer, outcome, mask)

        train_case("clean", train.images, train.labels, train.sample_ids, mask=clean_forget)
        train_case("poisoned", poisoned.observed.images, poisoned.observed.labels,
                   train.sample_ids, mask=poisoned_forget)
        train_case("retrained_clean", train.images[~clean_forget], train.labels[~clean_forget],
                   train.sample_ids[~clean_forget], retained_only=True)
        if np.array_equal(clean_forget, poisoned_forget):
            # The same clean retained IDs imply the same deterministic retraining problem.
            bundles["retrained_poisoned"] = bundles["retrained_clean"]
            transformers["retrained_poisoned"] = transformers["retrained_clean"]
            weights["retrained_poisoned"] = weights["retrained_clean"].copy()
            predictions["retrained_poisoned__test"] = predictions["retrained_clean__test"].copy()
            metrics["models"]["retrained_poisoned"] = dict(metrics["models"]["retrained_clean"])
            resources["retrained_poisoned"] = {"reused_model": "retrained_clean", "additional_circuit_evaluations": 0}
            for key, value in list(feature_arrays.items()):
                if key.startswith("retrained_clean__"):
                    feature_arrays[key.replace("retrained_clean__", "retrained_poisoned__")] = value
        else:
            train_case("retrained_poisoned", train.images[~poisoned_forget], train.labels[~poisoned_forget],
                       train.sample_ids[~poisoned_forget], retained_only=True)
        save_arrays(path / "retraining_ids.npz", clean=train.sample_ids[~clean_forget],
                    poisoned=train.sample_ids[~poisoned_forget])

        if config.label_shuffle_control:
            shuffled_labels = np.random.default_rng(config.seed).permutation(train.labels)
            save_arrays(path / "label_shuffle.npz", sample_ids=train.sample_ids, labels=shuffled_labels)
            train_case("label_shuffle", train.images, shuffled_labels, train.sample_ids)

        classical_models = {}
        if config.classical_baselines:
            for case in ("clean", "poisoned"):
                bundle = bundles[case]
                classical = fit_classical_baselines(bundle["features"], bundle["labels"], config.seed)
                classical_models[case] = classical
                for name, model in classical.items():
                    key = f"{case}_{name}"
                    probabilities = aligned_probabilities(model, bundle["transformer"].transform(test.images))
                    predictions[f"{key}__test"] = probabilities
                    metrics["models"][key] = classification_metrics(test.labels, probabilities, forget_class)
            joblib.dump(classical_models, path / "classical_models.joblib")

        for case in ("clean", "poisoned"):
            bundle, mask = bundles[case], masks[case]
            model = bundle["model"]
            model.executor.reset_seed(config.seed + 1000)
            reference_stats = dict(model.executor.stats)
            references = prepare_references(model, bundle["features"], bundle["labels"], mask,
                                            bundle["outcome"].weights, config.unlearning, config.seed)
            resources[f"{case}_references"] = {
                "stage_resources": {key: model.executor.stats[key] - reference_stats[key] for key in reference_stats}}
            save_arrays(path / "references" / f"{case}.npz",
                        forget_ids=train.sample_ids[references.forget_indices],
                        anchor_ids=train.sample_ids[references.anchor_indices],
                        calibration_ids=train.sample_ids[references.calibration_indices],
                        anchor_probabilities=references.anchor_probabilities,
                        calibration_probabilities=references.calibration_probabilities,
                        original_weights=references.original_weights)
            reference_probabilities = predictions[f"retrained_{case}__test"]
            metrics["retraining_comparisons"][case] = compare_to_retraining(
                test.labels, reference_probabilities, predictions[f"{case}__test"], forget_class)
            for method in config.unlearning.methods:
                name = f"{case}_{method}"
                report("unlearning", model=name)
                before_stats = dict(model.executor.stats)
                outcome, target = unlearn(
                    model, bundle["features"], references, config.unlearning, method, config.seed + 2000,
                    callback=lambda row, current=name: report("unlearning_progress", model=current, **row),
                )
                record_model(name, model, bundle["transformer"], outcome, mask)
                resources[name]["stage_resources"] = {
                    key: model.executor.stats[key] - before_stats[key] for key in before_stats}
                transformers[name] = bundle["transformer"]
                metrics["targets"][name] = target.tolist()
                metrics["unlearning_changes"][name] = forgetting_changes(
                    metrics["models"][case], metrics["models"][name])
                metrics["retraining_comparisons"][name] = compare_to_retraining(
                    test.labels, reference_probabilities, predictions[f"{name}__test"], forget_class)

        metrics["poisoning_damage_before_unlearning"] = forgetting_changes(
            metrics["models"]["clean"], metrics["models"]["poisoned"])
        metrics["additional_unlearning_effect_under_poisoning"] = {}
        for method in config.unlearning.methods:
            normal = metrics["unlearning_changes"][f"clean_{method}"]
            poisoned_change = metrics["unlearning_changes"][f"poisoned_{method}"]
            metrics["additional_unlearning_effect_under_poisoning"][method] = {
                key: None if normal[key] is None or poisoned_change[key] is None
                else poisoned_change[key] - normal[key] for key in normal}

        joblib.dump(transformers, path / "preprocessing.joblib")
        save_json(path / "preprocessing.json", {name: value.diagnostics() for name, value in transformers.items()})
        save_arrays(path / "weights.npz", **weights)
        save_arrays(path / "features.npz", **feature_arrays)
        save_arrays(path / "predictions.npz", **predictions)
        save_json(path / "metrics.json", metrics)
        save_json(path / "resources.json", resources)
        pd.concat(histories, ignore_index=True).to_csv(path / "history.csv", index=False)
        finish_run(path, "complete", dataset_source=dataset_source)
        report("complete", path=str(path))
    except BaseException as error:
        finish_run(path, "failed", error_type=type(error).__name__, error=str(error))
        raise
    return load_run(path)


def results_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        config = result["config"]
        # Keep ablations/noise/preprocessing protocols separate while combining
        # optimization seeds and folds. Data and poisoning seeds remain explicit.
        protocol = {"configuration": {**config, "seed": None, "data": {**config["data"], "fold": None}},
                    "code_fingerprint": result["manifest"]["identity"]["code_fingerprint"],
                    "environment": result["manifest"]["identity"]["environment"]}
        for model, values in result["metrics"]["models"].items():
            row = {"run_id": result["path"].name, "condition": config["poison"]["kind"],
                   "poison_rate": config["poison"]["rate"], "seed": config["seed"],
                   "fold": config["data"]["fold"], "model": model,
                   "protocol_id": fingerprint(protocol)[:16],
                   "alpha": config["unlearning"]["alpha"], "lambda": config["unlearning"]["regularization"],
                   "beta": config["unlearning"]["beta"], "anchor_fraction": config["unlearning"]["anchor_fraction"],
                   "shots": config["backend"]["shots"], "preprocessing": config["data"]["preprocessing"]}
            row.update({key: value for key, value in values.items() if not isinstance(value, (dict, list))})
            row.update({key: value for key, value in result["metrics"]["retraining_comparisons"].get(model, {}).items()
                        if isinstance(value, (int, float))})
            rows.append(row)
    return pd.DataFrame(rows)


def run_study(configurations, *, root=PROJECT_ROOT, splits=None, download=False, reuse=True,
              callback=None) -> pd.DataFrame:
    """Explicit configurations cover rates, seeds, folds, and alpha/lambda/beta ablations.

    Keep DataConfig.seed fixed to pair the data across optimization seeds. Do not
    inject one split when you intend different fold configurations.
    """
    configurations = list(configurations)
    if not configurations:
        raise ValueError("Provide at least one experiment configuration.")
    if splits is not None and any(config.data != configurations[0].data for config in configurations):
        raise ValueError("One injected split cannot represent multiple different data/fold configurations.")
    results = [run_experiment(config, root=root, splits=splits, download=download,
                              reuse=reuse, callback=callback) for config in configurations]
    table = results_table(results)
    destination = Path(root) / "results/tables"
    destination.mkdir(parents=True, exist_ok=True)
    name = fingerprint([result["path"].name for result in results])[:16]
    table.to_csv(destination / f"study_{name}.csv", index=False)
    return table


def restore_vqc(run_directory, model_name: str, *, backend_override=None):
    """Return (VQC, fitted_preprocessor, weights) from a TRUSTED local run.

    This explicit operation loads joblib preprocessing objects. Only use runs
    created by this project or another trusted source. No training occurs.
    """
    result = load_run(run_directory)
    config = ExperimentConfig.from_dict(result["config"])
    if model_name not in result["weights"]:
        raise KeyError(f"No quantum checkpoint named {model_name}.")
    transformers = joblib.load(Path(run_directory) / "preprocessing.joblib")
    model = VQC(config.circuit, backend_override or config.backend, config.seed)
    return model, transformers[model_name], result["weights"][model_name].copy()


def evaluate_noise_sweep(circuit_config, backend_config, features, labels, weights,
                         noise_levels=(0.0, 0.001, 0.005, 0.01), seeds=(DEFAULT_SEED,),
                         forget_class=2) -> pd.DataFrame:
    """Evaluate FIXED trained weights under device noise; does not retrain models.

    Each level sets p1=p and p2=2p. The original readout-error setting is retained.
    Use finite shots for the course-facing sweep; exact mode remains explicitly labeled.
    """
    if backend_config.use_ibm_hardware:
        raise ValueError("Synthetic noise sweeps use local AerSimulator.")
    rows = []
    for level in noise_levels:
        for seed in seeds:
            backend = replace(backend_config, one_qubit_error=level, two_qubit_error=2 * level)
            model = VQC(circuit_config, backend, seed)
            metrics = classification_metrics(labels, model.predict_proba(features, weights), forget_class)
            rows.append({"noise_level": level, "seed": seed,
                         "expectation_mode": "exact" if backend.shots is None else "sampled",
                         "actual_device": model.executor.actual_device,
                         **{key: value for key, value in metrics.items() if not isinstance(value, (dict, list))}})
    return pd.DataFrame(rows)
