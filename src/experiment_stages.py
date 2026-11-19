"""Checkpointed notebook stages: supervised models first, then class unlearning.

Model caches describe the actual fitting problem, so identical clean retained
records are trained once across poison rates. Condition dictionaries retain the
attack settings and exact model links. Nothing runs on import.
"""

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.artifacts import (environment_info, finish_run,
                           fingerprint, load_arrays, load_json, load_run, save_arrays,
                           save_json, save_poisoned_dataset, save_source_snapshot,
                           save_split_manifest, save_training_result, source_fingerprint)
from src.config import ExperimentConfig, PROJECT_ROOT
from src.checkpoints import CheckpointSession
from src.data.dataset import ImageDataset
from src.data.poisoning import apply_poisoning, forget_mask
from src.data.preprocessing import ImagePreprocessor, normalize_images
from src.evaluation.metrics import classification_metrics, compare_to_retraining, forgetting_changes
from src.models.classical import aligned_probabilities, fit_classical_baselines
from src.models.vqc import VQC
from src.training import fit_vqc
from src.unlearning.methods import unlearn
from src.unlearning.targets import UnlearningReferences, prepare_references


def _fit_config(config):
    # Attack rate and unlearning settings do not affect an identical supervised fit.
    return {"seed": config.seed, "data": asdict(config.data),
            "poison": {"kind": "none", "rate": 0.0},
            "circuit": asdict(config.circuit), "backend": asdict(config.backend),
            "training": asdict(config.training), "classical_baselines": config.classical_baselines}


def _identity(stage, config, **details):
    return {"stage": stage, "config": config, "code_fingerprint": source_fingerprint(),
            "environment": environment_info(), **details}


def _report(callback, stage, role, **values):
    if callback is not None:
        callback({"stage": stage, "model": role, **values})


def _check_gold(gold_run, splits, config):
    """Accept an explicitly supplied historical gold model only after strict checks."""
    gold = load_run(gold_run["path"])
    identity = gold["manifest"]["identity"]
    if identity.get("stage") != "clean_vqc":
        raise ValueError("gold_run must be a completed Section 7 clean_vqc run.")
    if identity.get("fold_fingerprints") != [splits.fingerprint()]:
        raise ValueError("Gold reference and current condition use different data/splits.")
    for key in ("seed", "data", "circuit", "backend", "training"):
        if fingerprint(gold["config"][key]) != fingerprint(_fit_config(config)[key]):
            raise ValueError(f"Gold reference has incompatible {key} settings.")
    np.testing.assert_array_equal(gold["predictions"]["test_sample_ids"], splits.test.sample_ids)
    np.testing.assert_array_equal(gold["predictions"]["test_labels"], splits.test.labels)
    return gold


def _supervised(config, splits, training_data, retained_only, *, gold_run=None,
                allow_training=False, reuse=True, root=PROJECT_ROOT, callback=None, role="model",
                checkpoints=None):
    root = Path(root).resolve()
    checkpoints = checkpoints if checkpoints is not None else CheckpointSession()
    if checkpoints.force_retrain and not checkpoints.owns(gold_run):
        gold_run = None  # Forced training cannot import an old gold model.
    forget_class = config.unlearning.forget_class
    validation = splits.validation.subset(splits.validation.labels != forget_class) if retained_only else splits.validation
    gold = _check_gold(gold_run, splits, config) if gold_run is not None else None
    identity = _identity("supervised_model", _fit_config(config),
                         split_fingerprint=splits.fingerprint(), train_fingerprint=training_data.fingerprint(),
                         validation_fingerprint=validation.fingerprint(), forget_class=forget_class,
                         imported_gold=None if gold is None else gold["path"].name)
    directory = root / "results/runs/supervised_models"
    slot = {"role": role, "fold": config.data.fold,
            "poison": asdict(config.poison) if "poisoned" in role else None}
    path, cached = checkpoints.open(directory, identity, allow_execution=allow_training,
                                     reuse=reuse, slot=slot)
    if cached is not None:
        _report(callback, "loaded_checkpoint", role)
        return cached
    if path is None:
        return None
    try:
        save_source_snapshot(path / "source.zip")
        fit_data = splits.train if config.data.preprocessing == "shared_clean_train" else training_data
        if gold is None:
            processor = ImagePreprocessor(config.data.seed).fit(fit_data.images, fit_data.sample_ids)
        else:
            processor = joblib.load(gold["path"] / "clean_preprocessing.joblib")
            np.testing.assert_array_equal(processor.fit_sample_ids_, splits.train.sample_ids)
        features = {"train": processor.transform(training_data.images),
                    "validation": processor.transform(validation.images),
                    "test": processor.transform(splits.test.images)}
        progress_file = path / "checkpoints/model.resume.json"
        stage = "resuming" if progress_file.exists() else "training"
        _report(callback, stage if gold is None else "importing_gold", role)
        if gold is None:
            model = VQC(config.circuit, config.backend, config.seed)
            outcome = fit_vqc(model, features["train"], training_data.labels,
                              features["validation"], validation.labels, config.training, config.seed,
                              callback=lambda row: _report(callback, "training_progress", role, **row),
                              checkpoint_path=progress_file)
            save_training_result(path, "model", outcome)
            weights, history = outcome.weights, outcome.history
            probabilities = model.predict_proba(features["test"], weights)
            resources = model.metadata()
            resources["training_seconds"] = float(history.elapsed_seconds.max())
        else:
            weights = gold["weights"]["clean"].copy()
            probabilities = gold["predictions"]["clean__test"].copy()
            history = gold["history"].drop(columns="model", errors="ignore")
            resources = load_json(gold["path"] / "resources.json")
            resources.update(imported_from=gold["path"].name, additional_circuit_evaluations=0,
                             training_seconds=float(history.elapsed_seconds.max()))
            checkpoint = load_arrays(gold["path"] / "checkpoints/clean.npz")
            save_arrays(path / "checkpoints/model.npz", **checkpoint)
            save_json(path / "checkpoints/model.json", load_json(gold["path"] / "checkpoints/clean.json"))
        predictions = {"test_sample_ids": splits.test.sample_ids, "test_labels": splits.test.labels,
                       "model__test": probabilities}
        metrics = {"models": {"model": classification_metrics(splits.test.labels, probabilities, forget_class)}}
        if config.classical_baselines:
            classical = fit_classical_baselines(features["train"], training_data.labels, config.seed)
            joblib.dump(classical, path / "classical_models.joblib")
            for name, estimator in classical.items():
                probabilities = aligned_probabilities(estimator, features["test"])
                predictions[f"{name}__test"] = probabilities
                metrics["models"][name] = classification_metrics(splits.test.labels, probabilities, forget_class)
        joblib.dump(processor, path / "preprocessing.joblib")
        save_json(path / "preprocessing.json", processor.diagnostics())
        save_arrays(path / "features.npz", **features, train_ids=training_data.sample_ids,
                    train_labels=training_data.labels, validation_ids=validation.sample_ids,
                    validation_labels=validation.labels)
        save_arrays(path / "weights.npz", model=weights)
        save_arrays(path / "predictions.npz", **predictions)
        save_json(path / "metrics.json", metrics)
        save_json(path / "resources.json", resources)
        history.assign(model="model").to_csv(path / "history.csv", index=False)
        finish_run(path, "complete")
    except BaseException as error:
        finish_run(path, "failed", error=f"{type(error).__name__}: {error}")
        raise
    return load_run(path)


def train_condition(config: ExperimentConfig, splits, *, gold_run=None, allow_training=False,
                    reuse=True, root=PROJECT_ROOT, callback=None, checkpoints=None):
    """Train/load original and matched retraining models; never perform unlearning.

    The caller's split is used as supplied. Raw originals stay unchanged. The
    retained reference uses clean images/labels of exactly the non-forget IDs.
    """
    if config.backend.use_ibm_hardware:
        raise ValueError("Notebook training stages use local Aer; hardware is evaluation-only.")
    checkpoints = checkpoints if checkpoints is not None else CheckpointSession()
    variant = apply_poisoning(splits.train, config.poison)
    masks = {"clean": splits.train.labels == config.unlearning.forget_class,
             "poisoned": forget_mask(variant, config.unlearning.forget_class, config.unlearning.forget_definition)}
    if any(not mask.any() or mask.all() for mask in masks.values()):
        raise ValueError("Every condition needs nonempty forget and retained training sets.")
    models = {}
    models["clean"] = _supervised(config, splits, splits.train, False, gold_run=gold_run,
                                  allow_training=allow_training, reuse=reuse, root=root, callback=callback,
                                  role="clean", checkpoints=checkpoints)
    if variant.observed.fingerprint() == splits.train.fingerprint() or config.poison.kind == "none":
        models["poisoned"] = models["clean"]
    else:
        models["poisoned"] = _supervised(config, splits, variant.observed, False,
                                         allow_training=allow_training, reuse=reuse, root=root, callback=callback,
                                         role="poisoned", checkpoints=checkpoints)
    for case, mask in masks.items():
        if case == "poisoned" and np.array_equal(mask, masks["clean"]):
            models["retrained_poisoned"] = models["retrained_clean"]
            continue
        role = f"retrained_{case}"
        models[role] = _supervised(config, splits, splits.train.subset(~mask), True,
                                   allow_training=allow_training, reuse=reuse, root=root, callback=callback,
                                   role=role, checkpoints=checkpoints)
    if config.label_shuffle_control:
        shuffled = ImageDataset(splits.train.images, np.random.default_rng(config.seed).permutation(splits.train.labels),
                                splits.train.sample_ids)
        models["label_shuffle"] = _supervised(config, splits, shuffled, False, allow_training=allow_training,
                                              reuse=reuse, root=root, callback=callback, role="label_shuffle",
                                              checkpoints=checkpoints)
    condition = {"config": config.to_dict(), "split_fingerprint": splits.fingerprint(),
                 "poisoning": variant.metadata(), "models": models, "unlearned": {}}
    if all(run is not None for run in models.values()):
        _save_condition(condition, splits, variant, root)
    return condition


def _save_condition(condition, splits, variant, root):
    root = Path(root).resolve()
    split_path = save_split_manifest(splits, root / "data/splits")
    variant_path = save_poisoned_dataset(variant, root / "data/annotated")
    links = {group: {role: run["path"].resolve().relative_to(root).as_posix()
                     for role, run in condition[group].items() if run is not None}
             for group in ("models", "unlearned")}
    payload = {key: condition[key] for key in ("config", "split_fingerprint", "poisoning")}
    payload.update(**links, split_manifest=split_path.relative_to(root).as_posix(),
                   annotated_dataset=variant_path.relative_to(root).as_posix())
    destination = root / "results/conditions" / f"condition_{fingerprint(payload)[:16]}.json"
    save_json(destination, payload)
    condition["manifest_path"] = destination


def _unlearn_stage(parent, reference, config, splits, original_data, mask, method, *,
                    allow_execution=False, reuse=True, root=PROJECT_ROOT, callback=None, role="model",
                    checkpoints=None):
    checkpoints = checkpoints if checkpoints is not None else CheckpointSession()
    # Hash-verified parents and their record identities prevent stale/mismatched reuse.
    parent, reference = load_run(parent["path"]), load_run(reference["path"])
    if parent["manifest"]["identity"]["train_fingerprint"] != original_data.fingerprint():
        raise ValueError("Unlearning parent does not match the prepared training variant.")
    for run in (parent, reference):
        if run["manifest"]["identity"]["split_fingerprint"] != splits.fingerprint():
            raise ValueError("Unlearning inputs belong to different splits.")
        if fingerprint(run["config"]) != fingerprint(_fit_config(config)):
            raise ValueError("Unlearning settings do not match the supervised model configuration.")
    arrays = load_arrays(parent["path"] / "features.npz")
    reference_arrays = load_arrays(reference["path"] / "features.npz")
    np.testing.assert_array_equal(arrays["train_ids"], splits.train.sample_ids)
    np.testing.assert_array_equal(reference_arrays["train_ids"], splits.train.sample_ids[~mask])
    for run in (parent, reference):
        np.testing.assert_array_equal(run["predictions"]["test_sample_ids"], splits.test.sample_ids)
        np.testing.assert_array_equal(run["predictions"]["test_labels"], splits.test.labels)
    settings = asdict(config.unlearning)
    settings.pop("methods")  # Each method has its own independently reusable checkpoint.
    identity = _identity("class_unlearning", {**_fit_config(config), "unlearning": settings},
                         parent_run=parent["path"].name, reference_run=reference["path"].name,
                         method=method, forget_ids=splits.train.sample_ids[mask].tolist())
    directory = Path(root).resolve() / "results/runs/unlearning"
    slot = {"role": role, "fold": config.data.fold,
            "poison": asdict(config.poison) if role.startswith("poisoned") else None,
            "unlearning": settings}
    path, cached = checkpoints.open(directory, identity, allow_execution=allow_execution,
                                     reuse=reuse, slot=slot)
    if cached is not None:
        _report(callback, "loaded_checkpoint", role)
        return cached
    if path is None:
        return None
    try:
        save_source_snapshot(path / "source.zip")
        processor = joblib.load(parent["path"] / "preprocessing.joblib")
        model = VQC(config.circuit, config.backend, config.seed)
        model.executor.reset_seed(config.seed + 1000)
        reference_file = path / "training_references.json"
        if reference_file.exists():
            saved = load_json(reference_file)
            references = UnlearningReferences(**{key: np.asarray(value)
                                                  for key, value in saved["references"].items()})
            model.executor.stats = saved["resources"]
        else:
            references = prepare_references(model, arrays["train"], arrays["train_labels"], mask,
                                            parent["weights"]["model"], config.unlearning, config.seed)
            save_json(reference_file, {"references": asdict(references), "resources": model.executor.stats})
        progress_file = path / "checkpoints/model.resume.json"
        _report(callback, "resuming_unlearning" if progress_file.exists() else "unlearning", role)
        outcome, target = unlearn(model, arrays["train"], references, config.unlearning, method,
                                  config.seed + 2000,
                                  callback=lambda row: _report(callback, "unlearning_progress", role, **row),
                                  checkpoint_path=progress_file)
        probabilities = model.predict_proba(arrays["test"], outcome.weights)
        metrics = classification_metrics(splits.test.labels, probabilities, config.unlearning.forget_class)
        anchors = references.anchor_indices
        audit = {"n_forget": len(references.forget_indices), "n_anchors": len(anchors),
                 "n_calibration": len(references.calibration_indices),
                 "anchor_accuracy_original_labels": float(np.mean(references.anchor_probabilities.argmax(1) == splits.train.labels[anchors])),
                 "anchor_accuracy_observed_labels": float(np.mean(references.anchor_probabilities.argmax(1) == arrays["train_labels"][anchors]))}
        save_json(path / "metrics.json", {"models": {"model": metrics}, "target": target.tolist(),
                  "reference_audit": audit,
                  "changes": forgetting_changes(parent["metrics"]["models"]["model"], metrics),
                  "retraining_comparison": compare_to_retraining(splits.test.labels, reference["predictions"]["model__test"], probabilities,
                                                                  config.unlearning.forget_class)})
        save_arrays(path / "references.npz", forget_ids=splits.train.sample_ids[references.forget_indices],
                    anchor_ids=splits.train.sample_ids[anchors],
                    calibration_ids=splits.train.sample_ids[references.calibration_indices],
                    anchor_probabilities=references.anchor_probabilities,
                    calibration_probabilities=references.calibration_probabilities,
                    original_weights=references.original_weights, target=target)
        save_training_result(path, "model", outcome)
        save_arrays(path / "weights.npz", model=outcome.weights)
        save_arrays(path / "predictions.npz", test_sample_ids=splits.test.sample_ids,
                    test_labels=splits.test.labels, model__test=probabilities)
        joblib.dump(processor, path / "preprocessing.joblib")
        save_json(path / "preprocessing.json", processor.diagnostics())
        resources = model.metadata()
        resources["training_seconds"] = float(outcome.history.elapsed_seconds.max())
        save_json(path / "resources.json", resources)
        outcome.history.assign(model="model").to_csv(path / "history.csv", index=False)
        finish_run(path, "complete")
    except BaseException as error:
        finish_run(path, "failed", error=f"{type(error).__name__}: {error}")
        raise
    return load_run(path)


def unlearn_condition(condition, splits, *, config=None, allow_execution=False,
                      reuse=True, root=PROJECT_ROOT, callback=None, checkpoints=None):
    """Run/load each method from its original checkpoint, never sequentially."""
    config = config or ExperimentConfig.from_dict(condition["config"])
    checkpoints = checkpoints if checkpoints is not None else CheckpointSession()
    if fingerprint(_fit_config(config)) != fingerprint(_fit_config(ExperimentConfig.from_dict(condition["config"]))):
        raise ValueError("Changing supervised settings requires a matching training condition.")
    if asdict(config.poison) != asdict(ExperimentConfig.from_dict(condition["config"]).poison):
        raise ValueError("Changing poisoning requires a matching training condition.")
    if any(run is None for run in condition["models"].values()):
        return {**condition, "unlearned": {}}
    result = {**condition, "config": config.to_dict(), "unlearned": {}}
    variant = apply_poisoning(splits.train, config.poison)
    for case in ("clean", "poisoned"):
        data = splits.train if case == "clean" else variant.observed
        mask = (splits.train.labels == config.unlearning.forget_class if case == "clean" else
                forget_mask(variant, config.unlearning.forget_class, config.unlearning.forget_definition))
        # The clean control shares exactly the original clean model and records.
        if config.poison.kind == "none":
            data = splits.train
        for method in config.unlearning.methods:
            role = f"{case}_{method}"
            result["unlearned"][role] = _unlearn_stage(
                condition["models"][case], condition["models"][f"retrained_{case}"],
                config, splits, data, mask, method, allow_execution=allow_execution,
                reuse=reuse, root=root, callback=callback, role=role, checkpoints=checkpoints)
    if all(run is not None for run in result["unlearned"].values()):
        _save_condition(result, splits, variant, root)
    return result


def restore_stage_model(run, backend_override=None):
    """Restore trusted local stage artifacts for optional new evaluations."""
    run = load_run(run["path"])
    config = ExperimentConfig.from_dict(run["config"])
    return (VQC(config.circuit, backend_override or config.backend, config.seed),
            joblib.load(run["path"] / "preprocessing.joblib"), run["weights"]["model"].copy())


def poison_feature_effect(variants, processor):
    """Paired changes through a FROZEN clean representation; no fitting/training.

    Projection energy is a geometric diagnostic, not classification information.
    Poisoned models may refit PCA separately under the selected protocol.
    """
    rows = []
    for name, variant in variants.items():
        selected = variant.is_poisoned
        if not selected.any():
            continue
        original, observed = variant.original.images[selected], variant.observed.images[selected]
        delta = normalize_images(observed) - normalize_images(original)
        projected = delta @ processor.pca.components_.T
        energy = np.sum(delta**2, axis=1)
        valid = energy > 0
        angle_delta = processor.transform(observed) - processor.transform(original)
        rows.append({"condition": name, "n_selected": int(selected.sum()),
                     "mean_pixel_l2": float(np.linalg.norm(delta, axis=1).mean()),
                     "mean_pca_l2": float(np.linalg.norm(projected, axis=1).mean()),
                     "mean_angle_l2": float(np.linalg.norm(angle_delta, axis=1).mean()),
                     "mean_projected_energy_fraction": float(np.mean(np.sum(projected[valid]**2, axis=1) / energy[valid])) if valid.any() else np.nan,
                     "unchanged_angle_records": int(np.all(np.isclose(angle_delta, 0, atol=1e-12), axis=1).sum())})
    return pd.DataFrame(rows)
