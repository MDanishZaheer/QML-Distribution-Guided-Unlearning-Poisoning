"""Prepare newly authorized 25% conditions, outside historical runs.

The archived supervised/unlearning routines are reused unchanged. A narrow scope
routes their new checkpoints to separate directories and supplies the shared Aer
service. No clean models or historical conditions are trained or overwritten.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from threading import Event, Lock
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import artifacts, experiment_stages as stages
from src.ablations import execution_environment, problem_from_runs
from src.checkpoints import CheckpointSession
from src.checkpoint_storage import CheckpointMirror
from src.config import DEFAULT_SEED, ExperimentConfig
from src.data.poisoning import apply_poisoning, forget_mask
from src.models.parallel_aer import AerBatchService, GpuMonitor


SUPPORT_STAGES = ("ablation_inputs_supervised", "ablation_inputs_unlearning")


def quarter_configs(review, folds, attack="label_flip"):
    """Use the historical per-fold protocol, with exactly 25% of the eligible class."""
    if attack not in ("label_flip", "pixel_patch"):
        raise ValueError("Select label_flip or the existing corner pixel_patch.")
    folds = tuple(folds)
    if not folds or len(set(folds)) != len(folds) or not set(folds).issubset(review.split_paths):
        raise ValueError("Choose unique folds from the saved five-fold experiment.")
    result = {}
    for fold in folds:
        clean = [c for c in review.conditions.values() if c["config"]["data"]["fold"] == fold
                 and c["config"]["poison"]["kind"] == "none"]
        if len(clean) != 1:
            raise ValueError(f"Missing/ambiguous clean protocol for fold {fold}.")
        base = ExperimentConfig.from_dict(clean[0]["config"])
        if base.seed != DEFAULT_SEED or base.backend.shots is not None or any(base.backend.noise_rates):
            raise ValueError("The extra condition requires seed 6304 and noiseless exact expectations.")
        result[fold] = replace(base, classical_baselines=False,
                               poison=replace(base.poison, kind=attack, rate=0.25,
                                              source_class=0, target_class=2, seed=DEFAULT_SEED),
                               unlearning=replace(base.unlearning, methods=("guided", "uniform")))
    return result


def quarter_plan(review, folds, attack="label_flip"):
    rows = []
    for fold, config in quarter_configs(review, folds, attack).items():
        for role, steps in (("poisoned", config.training.steps),
                            ("retrained_poisoned", config.training.steps),
                            ("poisoned_guided", config.unlearning.training.steps),
                            ("poisoned_uniform", config.unlearning.training.steps)):
            if attack == "pixel_patch" and role == "retrained_poisoned":
                continue  # Pixel patches do not change forget membership; reuse the clean retained reference.
            rows.append({"fold": fold, "condition": f"{attack}_25pct", "role": role,
                         "updates": steps, "seed": DEFAULT_SEED})
    return pd.DataFrame(rows)


def _blocked(*args, **kwargs):
    raise RuntimeError("Only the separate 25% condition preparation is authorized in this scope.")


@contextmanager
def quarter_execution(root, service, review_guard=None, *, attack="label_flip", review=None):
    """Allow only new-condition stages, with local Adam/RNG saves backed up to storage."""
    from src.evaluation.saved_results import protect_review
    import src.checkpoints as checkpoints
    root = Path(root).resolve()
    if review_guard is not None:
        review_guard.close()
    try:
        with ExitStack() as stack:
            original_open = CheckpointSession.open
            original_fit, original_unlearn = stages.fit_vqc, stages.unlearn
            original_create = artifacts.create_run
            lock = Lock()
            mapping = {root / "results/runs" / name: root / "results/runs" / target
                       for name, target in zip(("supervised_models", "unlearning"), SUPPORT_STAGES)}
            allowed = set(mapping.values())
            execution = {"condition": f"{attack}_25pct", "execution": asdict(service.execution),
                         "runtime_environment": execution_environment()}

            def redirected_open(session, directory, identity, **kwargs):
                directory = Path(directory).resolve()
                slot = kwargs.get("slot") or {}
                poison = slot.get("poison") or {}
                if (directory not in mapping or poison.get("kind") != attack
                        or poison.get("rate") != 0.25 or poison.get("seed") != DEFAULT_SEED
                        or slot.get("role") not in ("poisoned", "retrained_poisoned",
                                                    "poisoned_guided", "poisoned_uniform")):
                    return _blocked()
                identity = {**identity, "new_condition_execution": execution}
                with lock:
                    return original_open(session, mapping[directory], identity, **kwargs)

            def restricted_create(directory, *args, **kwargs):
                if Path(directory).resolve() not in allowed:
                    return _blocked()
                return original_create(directory, *args, **kwargs)

            def model_factory(circuit, backend, seed=DEFAULT_SEED):
                if circuit != service.circuit_config or backend != service.backend_config or seed != DEFAULT_SEED:
                    return _blocked()
                return service.model(seed)

            def mirrored(function):
                def execute(model, *args, checkpoint_path, callback=None, **kwargs):
                    if getattr(model.executor, "service", None) is not service:
                        return _blocked()
                    if not Path(checkpoint_path).resolve().is_relative_to(root / "results/runs"):
                        return _blocked()
                    if not any(Path(checkpoint_path).resolve().is_relative_to(directory) for directory in allowed):
                        return _blocked()
                    mirror = CheckpointMirror(checkpoint_path)
                    def report(row):
                        if row["step"] == 0:
                            mirror.flush_now()
                        if mirror.error is not None:
                            raise RuntimeError("storage checkpoint backup failed.") from mirror.error
                        if callback is not None:
                            callback(row)
                    with mirror as local:
                        return function(model, *args, checkpoint_path=local, callback=report, **kwargs)
                return execute

            for target in ("src.experiments.run_clean_vqc", "src.experiments.run_classical_cv",
                           "src.experiments.fit_vqc", "src.experiments.unlearn",
                           "src.experiment_stages.train_condition", "src.experiment_stages.unlearn_condition",
                           "src.models.classical.fit_classical_baselines", "src.experiments.fit_classical_baselines",
                           "src.experiment_stages.fit_classical_baselines", "src.data.dataset.download_mnist",
                           "src.checkpoints.retire_pilot_checkpoints", "src.training.fit_vqc",
                           "src.unlearning.methods.unlearn"):
                stack.enter_context(patch(target, _blocked))
            stack.enter_context(patch.object(CheckpointSession, "open", redirected_open))
            stack.enter_context(patch.object(artifacts, "create_run", restricted_create))
            stack.enter_context(patch.object(checkpoints, "create_run", restricted_create))
            stack.enter_context(patch.object(stages, "VQC", model_factory))
            stack.enter_context(patch.object(stages, "fit_vqc", mirrored(original_fit)))
            stack.enter_context(patch.object(stages, "unlearn", mirrored(original_unlearn)))
            if attack == "pixel_patch":
                # The historical quantum reference also fitted auxiliary classical
                # estimators. That flag does not affect its VQC; compare every
                # scientific VQC setting strictly and retain its original provenance.
                original_fingerprint = stages.fingerprint
                def vqc_fingerprint(value):
                    if isinstance(value, dict) and {"seed", "data", "circuit", "backend", "training", "classical_baselines"}.issubset(value):
                        value = {key: item for key, item in value.items() if key != "classical_baselines"}
                    return original_fingerprint(value)
                stack.enter_context(patch.object(stages, "fingerprint", vqc_fingerprint))
            if review is not None:
                historical = getattr(review, "runs", {})
                original_load = stages.load_run
                original_arrays = stages.load_arrays
                def reviewed_run(path):
                    resolved = Path(path).resolve()
                    return historical[resolved] if resolved in historical else original_load(path)
                def reviewed_arrays(path):
                    if Path(path).parent.resolve() in historical:
                        path = review._read_path(path)
                    return original_arrays(path)
                stack.enter_context(patch.object(stages, "load_run", reviewed_run))
                stack.enter_context(patch.object(stages, "load_arrays", reviewed_arrays))
            # Each independent PCA fit owns its arrays; avoid BLAS threads per worker oversubscribing the CPU.
            stack.enter_context(threadpool_limits(limits=1, user_api="blas"))
            yield
    finally:
        if review_guard is not None:
            review_guard.enter_context(protect_review())


def _parallel(jobs, workers, progress):
    cancellation = Event()
    def run(key, function):
        if cancellation.is_set():
            raise InterruptedError("Pending new-condition work was cancelled.")
        return key, function(cancellation)
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, key, function) for key, function in jobs]
        try:
            for future in as_completed(futures):
                key, value = future.result()
                results[key] = value
                if progress is not None:
                    progress({"stage": "condition_preparation", "completed": len(results),
                              "planned": len(jobs), "problem": str(key)})
        except BaseException:
            cancellation.set()
            for future in futures:
                future.cancel()
            raise
    return results


def prepare_quarter_problems(review, splits, execution, *, folds=(0, 1, 2, 3, 4),
                             force_retrain=False, progress=None, review_guard=None, attack="label_flip"):
    """Prepare one exact-25% attack; reuse the pixel condition's exact clean reference."""
    root = review.root
    configs = quarter_configs(review, folds, attack)
    variants = {}
    reused = {}
    for fold, config in configs.items():
        partition = splits[fold]
        clean = next(c for c in review.conditions.values()
                     if c["config"]["data"]["fold"] == fold and c["config"]["poison"]["kind"] == "none")
        if partition.fingerprint() != clean["split_fingerprint"]:
            raise ValueError("25% preparation must reuse the exact saved split.")
        variants[fold] = apply_poisoning(partition.train, config.poison)
        larger = apply_poisoning(partition.train, replace(config.poison, rate=0.5))
        if not np.isin(np.flatnonzero(variants[fold].is_poisoned), np.flatnonzero(larger.is_poisoned)).all():
            raise ValueError("25% selections are not nested in the historical 50% protocol.")
        if attack == "pixel_patch":
            reference = clean["models"]["retrained_clean"]
            mask = forget_mask(variants[fold], config.unlearning.forget_class, config.unlearning.forget_definition)
            retained = partition.train.subset(~mask)
            if reference["manifest"]["identity"]["train_fingerprint"] != retained.fingerprint():
                raise ValueError("Pixel reference does not match the exact retained records.")
            saved = artifacts.load_arrays(review._read_path(reference["path"] / "features.npz"))
            np.testing.assert_array_equal(saved["train_ids"], retained.sample_ids)
            np.testing.assert_array_equal(saved["train_labels"], retained.labels)
            reused[fold, "retrained_poisoned"] = reference
    protocol = {"version": 1, "condition": f"{attack}_25pct", "seed": DEFAULT_SEED,
                "original_report": review.report_path.relative_to(root).as_posix(),
                "configs": {str(fold): config.to_dict() for fold, config in configs.items()},
                "splits": {str(fold): splits[fold].fingerprint() for fold in configs},
                "code_fingerprint": artifacts.source_fingerprint(root), "environment": execution_environment(),
                "execution": asdict(execution)}
    if force_retrain:
        protocol["forced_attempt"] = uuid4().hex
    destination = root / "results/ablations/condition_inputs" / artifacts.fingerprint(protocol)[:16]
    index_file = destination / "study.json"
    index = {"protocol": protocol, "complete": False, "conditions": {},
             "planned_supervised_fits": (1 if attack == "pixel_patch" else 2) * len(configs),
             "planned_default_unlearning_fits": 2 * len(configs)}
    artifacts.save_json(index_file, index)
    session = CheckpointSession(force_retrain=force_retrain)
    with AerBatchService(next(iter(configs.values())).circuit, next(iter(configs.values())).backend, execution) as service, GpuMonitor() as monitor:
        with quarter_execution(root, service, review_guard, attack=attack, review=review):
            jobs = []
            def callback(cancel, fold, role, updates):
                def check(row):
                    if cancel.is_set():
                        raise InterruptedError("25% preparation stopped at a saved update.")
                    step = row.get("step")
                    if progress is not None and step is not None and (step in (0, 1, updates) or step % 10 == 0):
                        progress({"stage": row["stage"], "fold": fold, "model": role,
                                  "step": step, "updates": updates})
                return check
            for role in ("poisoned", "retrained_poisoned"):
                if attack == "pixel_patch" and role == "retrained_poisoned":
                    continue
                for fold, config in configs.items():
                    variant = variants[fold]
                    mask = forget_mask(variant, config.unlearning.forget_class, config.unlearning.forget_definition)
                    training_data = variant.observed if role == "poisoned" else splits[fold].train.subset(~mask)
                    def supervised(cancel, fold=fold, config=config, data=training_data, role=role):
                        return stages._supervised(config, splits[fold], data, role == "retrained_poisoned",
                                                  allow_training=True, root=root, role=role, checkpoints=session,
                                                  callback=callback(cancel, fold, role, config.training.steps))
                    jobs.append(((fold, role), supervised))
            trained = _parallel(jobs, execution.workers, progress)
            trained.update(reused)
            jobs = []
            for method in ("guided", "uniform"):
                for fold, config in configs.items():
                    variant = variants[fold]
                    mask = forget_mask(variant, config.unlearning.forget_class, config.unlearning.forget_definition)
                    def unlearning(cancel, fold=fold, config=config, variant=variant, mask=mask, method=method):
                        return stages._unlearn_stage(trained[fold, "poisoned"], trained[fold, "retrained_poisoned"],
                            config, splits[fold], variant.observed, mask, method, allow_execution=True, root=root,
                            role=f"poisoned_{method}", checkpoints=session,
                            callback=callback(cancel, fold, f"poisoned_{method}", config.unlearning.training.steps))
                    jobs.append(((fold, method), unlearning))
            unlearned = _parallel(jobs, execution.workers, progress)
        pd.DataFrame(monitor.rows).to_csv(destination / "gpu_cpu_telemetry.csv", index=False)
    problems = []
    for fold, config in configs.items():
        parent, reference = trained[fold, "poisoned"], trained[fold, "retrained_poisoned"]
        guided, uniform = unlearned[fold, "guided"], unlearned[fold, "uniform"]
        problems.append(problem_from_runs(f"{attack}_25pct", fold, config, parent, reference, guided, uniform,
                                          baseline_origin="prepared_25pct"))
        annotation = artifacts.save_poisoned_dataset(variants[fold], root / "data/annotated")
        index["conditions"][str(fold)] = {
            "config": config.to_dict(), "split_fingerprint": splits[fold].fingerprint(),
            "annotated_dataset": annotation.relative_to(root).as_posix(),
            "poisoning": variants[fold].metadata(),
            "retained_reference_reused": attack == "pixel_patch",
            "runs": {role: run["path"].relative_to(root).as_posix() for role, run in (
                ("poisoned", parent), ("retrained_poisoned", reference), ("guided", guided), ("uniform", uniform))}}
    index["complete"] = True
    artifacts.save_json(index_file, index)
    return problems, index_file
